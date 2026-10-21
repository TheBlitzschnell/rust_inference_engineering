//! Scheduling and SLOs on SmolLM2-135M-Instruct: latency against load,
//! the step budget (chunked prefill), the order of service, and admission
//! control under overload.
//!
//! Run with: cargo run --release -p ch25-scheduling [load|chunks|policy|overload]
//! (needs the model: ./tools/download_model.sh)

use ch14_kv_cache::{Matrix, Model};
use ch16_real_model::{
    DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16, model_dir,
};
use ch17_profiling::map_matrices;
use ch21_engine_thread::EngineHandle;
use ch23_continuous_batching::PackedBf16;
use ch24_paged_kv_cache::PagedConfig;
use ch25_scheduling::{
    Planned, Policy, Report, SchedulerConfig, Slo, Stats, Trace, percentile, poisson_arrivals,
    run_closed, run_open, spawn,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const THREADS: usize = 4;
const TEXT: &str = include_str!("../../15-sampling/data/pride-and-prejudice-1-6.txt");

/// The service-level objective used throughout: the first token within a
/// second, then at least 10 tokens per second on average.
const SLO: Slo = Slo {
    ttft: Duration::from_secs(1),
    tpot: Duration::from_millis(100),
};

fn main() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("load") {
        load_curve(&dir, &tok);
    }
    if run("chunks") {
        chunks(&dir, &tok);
    }
    if run("policy") {
        policy(&dir, &tok);
    }
    if run("overload") {
        overload(&dir, &tok);
    }
}

fn load(dir: &Path) -> Model<PackedBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        |w: DenseBf16| PackedBf16::new(w.values(), w.rows(), w.cols()),
    )
}

/// An engine with room for 16 requests (chapter 23: throughput stops
/// growing there on this machine) and 16,384 positions of KV cache.
fn engine(
    dir: &Path,
    max_step_tokens: usize,
    policy: Policy,
    max_waiting: Option<usize>,
) -> (EngineHandle, Arc<Stats>) {
    let (handle, _thread, stats) = spawn(
        load(dir),
        SchedulerConfig {
            paged: PagedConfig {
                threads: THREADS,
                max_batch: 16,
                context: 2048,
                max_step_tokens,
                num_blocks: 1024,
                block_size: 16,
                prefix_caching: false,
            },
            policy,
            max_waiting,
        },
    );
    (handle, stats)
}

/// A request to summarize `words` words of the novel, starting near word
/// `at` (wrapped so that the passage always has `words` words).
fn summarize(tok: &Tokenizer, at: usize, words: usize) -> Vec<u32> {
    let all: Vec<&str> = TEXT.split_whitespace().collect();
    let start = at % (all.len() - words);
    let passage = &all[start..start + words];
    let question = format!(
        "Summarize this passage in one sentence:\n\n{}",
        passage.join(" ")
    );
    tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: &question,
    }]))
}

/// Chat requests: passages of 20 to 60 words (prompts of about 50 to 100
/// tokens), 64 tokens of answer each, arriving at the given times.
fn chat(tok: &Tokenizer, at: &[Duration]) -> Vec<Planned> {
    at.iter()
        .enumerate()
        .map(|(i, &at)| Planned {
            at,
            prompt: summarize(tok, (i * 397) % 9000, 20 + (i * 7) % 41),
            max_tokens: 64,
        })
        .collect()
}

fn ms(d: Duration) -> String {
    format!("{:.0} ms", d.as_secs_f64() * 1e3)
}

fn secs(d: Duration) -> String {
    format!("{:.1} s", d.as_secs_f64())
}

/// Prints one row of part 1's table.
fn load_row(name: &str, r: &Report) {
    println!(
        "   {name:<17} {:>5.2} {:>7.0} {:>9} {:>7} {:>7} {:>8} {:>6} {:>6.0}% {:>6.2}",
        r.served as f64 / r.elapsed.as_secs_f64(),
        r.throughput(),
        ms(r.ttft.p50),
        ms(r.ttft.p90),
        ms(r.ttft.max),
        ms(r.tpot.p50),
        ms(r.tpot.p90),
        100.0 * r.attainment(),
        r.good as f64 / r.elapsed.as_secs_f64()
    );
}

/// Part 1: the latency of chat requests as the load grows, from clients
/// that wait for each answer (closed loop) and from independent arrivals
/// (open loop, Poisson).
fn load_curve(dir: &Path, tok: &Tokenizer) {
    println!("== 1. chat requests (prompts of 50-100 tokens, 64 tokens out), step budget 128");
    println!("      SLO: first token within 1 s, then at most 100 ms per token on average");
    println!(
        "                    req/s tokens/s  TTFT p50     p90     max  TPOT p50    p90 met SLO goodput"
    );
    for users in [1, 16] {
        let (engine, _) = engine(dir, 128, Policy::Fcfs, None);
        let plan = chat(tok, &vec![Duration::ZERO; 4 * users]);
        let traces = run_closed(&engine, users, &plan, Duration::ZERO);
        load_row(
            &format!("closed, {users} user{}", if users == 1 { "" } else { "s" }),
            &Report::new(&traces, &SLO),
        );
    }
    for rate in [1.0, 2.0, 3.0, 4.0, 5.0] {
        let (engine, _) = engine(dir, 128, Policy::Fcfs, None);
        // 12 seconds of arrivals.
        let plan = chat(tok, &poisson_arrivals(rate, (12.0 * rate) as usize, 7));
        let traces = run_open(&engine, &plan);
        load_row(&format!("open, {rate:.0}/s"), &Report::new(&traces, &SLO));
    }
    println!();
}

/// Part 2: the step budget. Eight requests stream long answers while three
/// long prompts arrive; how long do the streams stall, and how long do the
/// long prompts wait for their first token?
fn chunks(dir: &Path, tok: &Tokenizer) {
    println!("== 2. eight streams of 200 tokens; three 1,050-token prompts arrive at 1, 3 and 5 s");
    println!("   step budget   gaps between the streams' tokens   long prompts' TTFT   all done");
    println!("                    p50      p99      max             p50      max");
    let mut plan: Vec<Planned> = (0..8)
        .map(|i| Planned {
            at: Duration::from_millis(50 * i as u64),
            prompt: summarize(tok, 2000 + i * 500, 15),
            max_tokens: 200,
        })
        .collect();
    for (i, s) in [1, 3, 5].into_iter().enumerate() {
        plan.push(Planned {
            at: Duration::from_secs(s),
            prompt: summarize(tok, 1000 + i * 2000, 800),
            max_tokens: 16,
        });
    }
    for budget in [32, 64, 128, 256, 2048] {
        let (engine, _) = engine(dir, budget, Policy::Fcfs, None);
        let traces = run_open(&engine, &plan);
        let (streams, long) = traces.split_at(8);
        let gaps = Report::new(streams, &SLO).itl;
        let ttft = Report::new(long, &SLO).ttft;
        let done = traces.iter().map(|t| t.sent + t.done).max().unwrap();
        println!(
            "   {budget:>6} tokens  {:>8} {:>8} {:>8}        {:>8} {:>8}   {:>8}",
            ms(gaps.p50),
            ms(gaps.p99),
            ms(gaps.max),
            ms(ttft.p50),
            ms(ttft.max),
            secs(done)
        );
    }
    println!();
}

/// Chat requests with Poisson arrivals, every 10th with a 1,050-token
/// prompt instead.
fn mixed(tok: &Tokenizer, rate: f64, n: usize) -> (Vec<Planned>, Vec<bool>) {
    let mut plan = chat(tok, &poisson_arrivals(rate, n, 11));
    let mut long = vec![false; n];
    for i in (4..n).step_by(10) {
        plan[i].prompt = summarize(tok, i * 300, 800);
        long[i] = true;
    }
    (plan, long)
}

/// Part 3: first come, first served against shortest prompt first.
fn policy(dir: &Path, tok: &Tokenizer) {
    let (plan, long) = mixed(tok, 1.5, 40);
    println!(
        "== 3. 40 requests at 1.5 per second; {} of them have 1,050-token prompts",
        long.iter().filter(|&&l| l).count()
    );
    println!("                       short prompts' TTFT         long prompts' TTFT   met SLO");
    println!("                          p50      p90      max         p50      max");
    for (name, policy, budget) in [
        ("FCFS, budget 512", Policy::Fcfs, 512),
        ("FCFS, budget 128", Policy::Fcfs, 128),
        ("shortest first, 128", Policy::ShortestFirst, 128),
    ] {
        let (engine, _) = engine(dir, budget, policy, None);
        let traces = run_open(&engine, &plan);
        let pick = |want: bool| -> Vec<Trace> {
            traces
                .iter()
                .zip(&long)
                .filter(|&(_, &l)| l == want)
                .map(|(t, _)| t.clone())
                .collect()
        };
        let short = Report::new(&pick(false), &SLO);
        let longs = Report::new(&pick(true), &SLO);
        let all = Report::new(&traces, &SLO);
        println!(
            "   {name:<20} {:>8} {:>8} {:>8}    {:>8} {:>8}   {:>6.0}%",
            ms(short.ttft.p50),
            ms(short.ttft.p90),
            ms(short.ttft.max),
            ms(longs.ttft.p50),
            ms(longs.ttft.max),
            100.0 * all.attainment()
        );
    }
    println!();
}

/// Part 4: more requests than the engine can serve, with and without a
/// limit on the queue.
fn overload(dir: &Path, tok: &Tokenizer) {
    let (rate, n) = (6.0, 90);
    println!("== 4. overload: {n} chat requests at {rate:.0} per second, about 1.7 times capacity");
    let plan = chat(tok, &poisson_arrivals(rate, n, 13));
    for max_waiting in [None, Some(3)] {
        let (engine, stats) = engine(dir, 128, Policy::Fcfs, max_waiting);
        let traces = run_open(&engine, &plan);
        let r = Report::new(&traces, &SLO);
        let name = max_waiting.map_or("no limit".to_string(), |n| format!("at most {n} waiting"));
        println!(
            "   {name}: {} served, {} rejected, longest queue {}; {} met the SLO; all done after {}",
            r.served,
            r.rejected,
            Stats::get(&stats.peak_waiting),
            r.good,
            secs(r.elapsed)
        );
        println!(
            "      served: TTFT p50 {}, p90 {}, max {}; TPOT p50 {}, p90 {}",
            ms(r.ttft.p50),
            ms(r.ttft.p90),
            ms(r.ttft.max),
            ms(r.tpot.p50),
            ms(r.tpot.p90)
        );
        // TTFT of the served requests, by when they arrived.
        let medians: Vec<String> = traces
            .chunks(n / 5)
            .map(|fifth| {
                let mut ttft: Vec<Duration> = fifth.iter().filter_map(Trace::ttft).collect();
                ttft.sort_unstable();
                format!("{:>7}", ms(percentile(&ttft, 50.0)))
            })
            .collect();
        println!(
            "      median TTFT by fifth of arrivals: {}",
            medians.join("  ")
        );
    }
    println!();
}
