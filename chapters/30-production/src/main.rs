//! Inference in production, on SmolLM2-135M: layer-by-layer debugging
//! against PyTorch, metrics from a loaded engine, capacity planning, and
//! what retries do around an outage.
//!
//! Run with: cargo run --release -p ch30-production [debug|metrics|capacity|outage]
//! (the first three need the model: ./tools/download_model.sh)

use ch07_threads::SpinPool;
use ch09_safetensors::SafeTensors;
use ch14_kv_cache::{DenseF32, Layer, Matrix, Model};
use ch16_real_model::{
    DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16, load_f32, model_dir,
};
use ch17_profiling::map_matrices;
use ch19_4bit::{Q4Matrix, Scheme};
use ch21_engine_thread::EngineHandle;
use ch23_continuous_batching::PackedBf16;
use ch24_paged_kv_cache::PagedConfig;
use ch25_scheduling::{
    Planned, Policy, Report, SchedulerConfig, Slo, poisson_arrivals, run_closed, run_open, spawn,
};
use ch28_parallelism::to_f32;
use ch30_production::capacity::{erlang_c, mean_wait, replicas_for};
use ch30_production::metrics::Registry;
use ch30_production::outage::{Retry, Scenario, Second, simulate};
use ch30_production::trace::{Diff, diff, first_jump, trace_last};
use std::path::Path;
use std::time::Duration;

const TEXT: &str = include_str!("../../15-sampling/data/pride-and-prejudice-1-6.txt");

fn main() {
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    let dir = model_dir();
    let have_model = dir.join("model.safetensors").exists();
    if !have_model && part != "outage" {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh (or run the `outage` part alone)",
            dir.display()
        );
    }
    if have_model && run("debug") {
        debug(&dir);
    }
    if have_model && run("metrics") {
        metrics(&dir);
    }
    if have_model && run("capacity") {
        capacity(&dir);
    }
    if run("outage") {
        outage();
    }
}

/// Chapter 16's PyTorch reference: the prompt, the hidden state entering
/// every layer at its last position, then after the final norm.
fn reference() -> (Vec<u32>, Vec<Vec<f32>>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../16-real-model/fixtures/reference.safetensors");
    let bytes = std::fs::read(path).expect("chapter 16's fixture");
    let st = SafeTensors::parse(&bytes).expect("safetensors");
    let ids: Vec<u32> = serde_json::from_str(&st.metadata()["prompt_ids"]).expect("prompt ids");
    let hidden = st
        .tensor("hidden_last")
        .expect("hidden states")
        .to_f32_vec()
        .expect("f32");
    let h = hidden.len() / 31;
    (ids, hidden.chunks_exact(h).map(<[f32]>::to_vec).collect())
}

/// SmolLM2-135M with every layer's matrices quantized to 4 bits and back
/// (the embedding untouched): the weights a 4-bit model computes with.
fn four_bit(dir: &Path) -> Model<DenseF32> {
    let (config, embed, layers, final_norm, lm_head) = load_f32(dir).expect("model").0.into_parts();
    let q = |w: DenseF32| {
        let q4 = Q4Matrix::new(
            w.values(),
            w.rows(),
            w.cols(),
            64,
            Scheme::MinMaxSearch,
            false,
        );
        DenseF32::new(&to_f32(&q4), w.rows(), w.cols())
    };
    let layers = layers
        .into_iter()
        .map(|l| Layer {
            attn_norm: l.attn_norm,
            wq: q(l.wq),
            wk: q(l.wk),
            wv: q(l.wv),
            wo: q(l.wo),
            mlp_norm: l.mlp_norm,
            w_gate: q(l.w_gate),
            w_up: q(l.w_up),
            w_down: q(l.w_down),
        })
        .collect();
    Model::new(config, embed, layers, final_norm, lm_head)
}

fn print_diffs(name: &str, diffs: &[Diff]) {
    let rows: Vec<usize> = vec![0, 1, 5, 10, 12, 13, 14, 29, 30];
    let cells: Vec<String> = rows
        .iter()
        .map(|&i| format!("{:>8.1e}", diffs[i].relative))
        .collect();
    let jump = match first_jump(diffs, 10.0, 1e-4) {
        Some(30) => "the final norm".to_string(),
        Some(i) => format!("input of layer {i}"),
        None => "none".to_string(),
    };
    println!("   {name:<26} {}   {jump}", cells.join(""));
}

/// Part 1: where two implementations part.
fn debug(dir: &Path) {
    println!("== 1. hidden state entering each layer (last prompt position), relative difference");
    let (ids, pytorch) = reference();
    let mut pool = SpinPool::with_all_cores();
    let ours = trace_last(&load_f32(dir).expect("model").0, &mut pool, &ids);
    println!(
        "   layer:                            0       1       5      10      12      13      14      29   final   first jump"
    );
    let against = |rows: &[Vec<f32>], base: &[Vec<f32>]| -> Vec<Diff> {
        rows.iter().zip(base).map(|(a, b)| diff(a, b)).collect()
    };
    let vs_pytorch = against(&ours, &pytorch);
    print_diffs("ours vs PyTorch", &vs_pytorch);
    let worst = vs_pytorch.iter().map(|d| d.cosine).fold(1.0, f64::min);
    // A plausible bug: layer 12's MLP norm weights not loaded (left at 1).
    let mut buggy = load_f32(dir).expect("model").0;
    buggy.layers[12].mlp_norm.fill(1.0);
    let bug = trace_last(&buggy, &mut pool, &ids);
    print_diffs("layer 12 norm bug vs ours", &against(&bug, &ours));
    let quantized = trace_last(&four_bit(dir), &mut pool, &ids);
    print_diffs("4-bit weights vs ours", &against(&quantized, &ours));
    println!("   (ours vs PyTorch: lowest cosine similarity {worst:.7})");
    // What each version would answer.
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let model = load_f32(dir).expect("model").0;
    let mut next = |rows: &[Vec<f32>]| -> String {
        let mut logits = vec![0.0; model.config.vocab_size];
        model
            .lm_head()
            .matmul(&mut pool, &rows[30], &mut logits, 1, &mut Vec::new());
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
        format!("{:?}", tok.decode(&[best as u32]))
    };
    println!(
        "   next token: PyTorch {}, ours {}, with the bug {}, 4-bit {}",
        next(&pytorch),
        next(&ours),
        next(&bug),
        next(&quantized)
    );
    println!();
}

fn packed(dir: &Path) -> Model<PackedBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        |w: DenseBf16| PackedBf16::new(w.values(), w.rows(), w.cols()),
    )
}

/// Chapter 25's engine, configured as that chapter recommends.
fn engine(dir: &Path) -> EngineHandle {
    spawn(
        packed(dir),
        SchedulerConfig {
            paged: PagedConfig {
                threads: 4,
                max_batch: 16,
                context: 2048,
                max_step_tokens: 128,
                num_blocks: 1024,
                block_size: 16,
                prefix_caching: false,
            },
            policy: Policy::ShortestFirst,
            max_waiting: Some(3),
        },
    )
    .0
}

fn chat(tok: &Tokenizer, at: &[Duration]) -> Vec<Planned> {
    let words: Vec<&str> = TEXT.split_whitespace().collect();
    at.iter()
        .enumerate()
        .map(|(i, &at)| {
            let n = 20 + (i * 7) % 41;
            let start = (i * 397) % (words.len() - n);
            let question = format!(
                "Summarize this passage in one sentence:\n\n{}",
                words[start..start + n].join(" ")
            );
            Planned {
                at,
                prompt: tok.encode(&chat_prompt(&[Message {
                    role: "user",
                    content: &question,
                }])),
                max_tokens: 64,
            }
        })
        .collect()
}

/// Part 2: what a server would export at `/metrics`, and what its
/// histograms say about percentiles.
fn metrics(dir: &Path) {
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let handle = engine(dir);
    let plan = chat(&tok, &poisson_arrivals(3.5, 70, 21));
    println!("== 2. metrics from 70 chat requests at 3.5 per second (chapter 25's engine and SLO)");
    let traces = run_open(&handle, &plan);
    // Prometheus's default buckets, and buckets chosen for this service.
    let default = [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ];
    let chosen = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0];
    let mut r = Registry::default();
    for t in &traces {
        let status = if t.rejected { "429" } else { "200" };
        r.inc(
            "requests_total",
            &format!("status=\"{status}\""),
            1.0,
            "Requests by HTTP status.",
        );
        if let Some(ttft) = t.ttft() {
            r.inc(
                "generated_tokens_total",
                "",
                t.tokens.len() as f64,
                "Tokens generated.",
            );
            let s = ttft.as_secs_f64();
            r.observe(
                "ttft_seconds",
                &default,
                s,
                "Time to first token (default buckets).",
            );
            r.observe(
                "ttft_chosen_seconds",
                &chosen,
                s,
                "Time to first token (chosen buckets).",
            );
        }
    }
    let exposition = r.exposition();
    for line in exposition.lines().filter(|l| {
        l.starts_with("requests_total")
            || l.starts_with("generated")
            || l.starts_with("ttft_seconds")
    }) {
        println!("   {line}");
    }
    let report = Report::new(
        &traces,
        &Slo {
            ttft: Duration::from_secs(1),
            tpot: Duration::from_millis(100),
        },
    );
    println!("   TTFT percentile     exact   default buckets   chosen buckets");
    for (q, exact) in [
        (0.5, report.ttft.p50),
        (0.9, report.ttft.p90),
        (0.99, report.ttft.p99),
    ] {
        println!(
            "   p{:<17} {:>6.0} ms {:>14.0} ms {:>14.0} ms",
            (q * 100.0) as u32,
            exact.as_secs_f64() * 1e3,
            r.histogram("ttft_seconds").expect("observed").quantile(q) * 1e3,
            r.histogram("ttft_chosen_seconds")
                .expect("observed")
                .quantile(q)
                * 1e3
        );
    }
    println!(
        "   ({} served, {} rejected; {:.0}% met the SLO)",
        report.served,
        report.rejected,
        100.0 * report.attainment()
    );
    println!();
}

/// Part 3: how many replicas for a forecast peak.
fn capacity(dir: &Path) {
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let handle = engine(dir);
    let plan = chat(&tok, &vec![Duration::ZERO; 64]);
    let traces = run_closed(&handle, 16, &plan, Duration::ZERO);
    let r = Report::new(
        &traces,
        &Slo {
            ttft: Duration::from_secs(1),
            tpot: Duration::from_millis(100),
        },
    );
    let mu = r.served as f64 / r.elapsed.as_secs_f64();
    let lambda = 40.0;
    println!(
        "== 3. capacity: one replica (this machine) completes {mu:.2} chat requests per second at saturation"
    );
    println!("   forecast peak: {lambda} requests per second (M/M/c model, not measured)");
    println!("   replicas   utilization   P(wait)   mean wait");
    let first = (lambda / mu).floor() as usize + 1;
    for c in first..first + 7 {
        println!(
            "   {c:>8} {:>12.0}% {:>8.0}% {:>9.0} ms",
            100.0 * lambda / (c as f64 * mu),
            100.0 * erlang_c(c, lambda, mu),
            1e3 * mean_wait(c, lambda, mu)
        );
    }
    let need = replicas_for(lambda, mu, 0.1, 0.2);
    println!(
        "   for a mean wait under 100 ms and P(wait) under 20%: {need} replicas, {} with one spare",
        need + 1
    );
    println!();
}

/// Part 4: a 5-second stall, under four policies.
fn outage() {
    println!(
        "== 4. a 5-second stall at 10 s: 80 requests/s, capacity 100/s, 1 s client timeout, 4 attempts (simulated)"
    );
    println!(
        "   policy                       good requests per second       wasted   lost   longest"
    );
    println!(
        "                              5-10 s  15-20 s  20-30 s  30-40 s  work            queue"
    );
    let base = Scenario {
        rate: 80.0,
        capacity: 100.0,
        timeout: 1.0,
        max_attempts: 4,
        retry: Retry::Immediately,
        drop_expired: false,
        max_queue: None,
        outage: (10.0, 15.0),
        seconds: 40,
        seed: 1,
    };
    let backoff = Retry::Backoff {
        base: 0.5,
        cap: 8.0,
    };
    let scenarios = [
        ("retry at once", base.clone()),
        (
            "backoff with jitter",
            Scenario {
                retry: backoff,
                ..base.clone()
            },
        ),
        (
            "drop expired requests",
            Scenario {
                drop_expired: true,
                ..base.clone()
            },
        ),
        (
            "all three, queue limit 100",
            Scenario {
                retry: backoff,
                drop_expired: true,
                max_queue: Some(100),
                ..base.clone()
            },
        ),
    ];
    for (name, s) in scenarios {
        let run = simulate(&s);
        let rate = |from: usize, to: usize| {
            run[from..to].iter().map(|x| x.good).sum::<usize>() as f64 / (to - from) as f64
        };
        let total = |f: fn(&Second) -> usize| run.iter().map(f).sum::<usize>();
        println!(
            "   {name:<26} {:>6.0} {:>8.0} {:>8.0} {:>8.0} {:>8} {:>6} {:>8}",
            rate(5, 10),
            rate(15, 20),
            rate(20, 30),
            rate(30, 40),
            total(|x| x.wasted),
            total(|x| x.lost),
            run.iter().map(|x| x.queue).max().unwrap_or(0)
        );
    }
    println!();
}
