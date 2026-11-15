//! More than one device, simulated with threads (one core each) on
//! SmolLM2-135M: collectives, tensor, pipeline and expert parallelism, and
//! estimates for real interconnects.
//!
//! Run with: cargo run --release -p ch28-parallelism [collectives|tensor|pipeline|experts|estimates]
//! (needs the model: ./tools/download_model.sh)

use ch02_numbers::Bf16;
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch16_real_model::{
    DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16, model_dir,
};
use ch17_profiling::map_matrices;
use ch23_continuous_batching::PackedBf16;
use ch28_parallelism::{Comm, Moe, forward_ep, moe, pp, run, shard};
use std::path::Path;
use std::time::{Duration, Instant};

const TEXT: &str = include_str!("../../15-sampling/data/pride-and-prejudice-1-6.txt");

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
    let run_part = |name: &str| part.is_empty() || part == name;
    if run_part("collectives") {
        collectives();
    }
    if run_part("tensor") {
        tensor(&dir, &tok);
    }
    if run_part("pipeline") {
        pipeline(&dir, &tok);
    }
    if run_part("experts") {
        experts(&dir, &tok);
    }
    if run_part("estimates") {
        estimates();
    }
}

fn ms(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}

/// Part 1: all-reduce, ring against naive, on 2 and 4 devices.
fn collectives() {
    println!("== 1. all-reduce of one buffer per device (threads; a transfer is a memory copy)");
    println!(
        "   devices   buffer       ring: time   sent by each      naive: time   sent by device 0"
    );
    for devices in [2, 4] {
        for floats in [576, 65_536, 4_194_304] {
            let reps = if floats > 1_000_000 { 5 } else { 50 };
            let time = |ring: bool| {
                let results = run(devices, |c: &Comm| {
                    let mut x = vec![1.0f32; floats];
                    let start = Instant::now();
                    for _ in 0..reps {
                        if ring {
                            c.all_reduce(&mut x);
                        } else {
                            c.all_reduce_naive(&mut x);
                        }
                    }
                    (start.elapsed() / reps, c.stats().bytes / reps as usize)
                });
                let slowest = results.iter().map(|r| r.0).max().unwrap_or_default();
                let most = results.iter().map(|r| r.1).max().unwrap_or_default();
                (slowest, most)
            };
            let (ring, ring_bytes) = time(true);
            let (naive, naive_bytes) = time(false);
            println!(
                "   {devices:>7} {:>9} KB {:>13} {:>12} KB {:>16} {:>15} KB",
                floats * 4 / 1024,
                ms(ring),
                ring_bytes / 1024,
                ms(naive),
                naive_bytes / 1024
            );
        }
    }
    println!();
}

fn packed(w: &[f32], rows: usize, cols: usize) -> PackedBf16 {
    let values: Vec<Bf16> = w.iter().map(|&x| Bf16::from_f32(x)).collect();
    PackedBf16::new(&values, rows, cols)
}

fn bf16_model(dir: &Path) -> Model<DenseBf16> {
    load_bf16(dir, Placement::Mapped).expect("model").0
}

fn chat(tok: &Tokenizer, question: &str) -> Vec<u32> {
    tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: question,
    }]))
}

/// Part 2: tensor parallelism over 1 and 3 devices of one core each.
fn tensor(dir: &Path, tok: &Tokenizer) {
    println!(
        "== 2. tensor parallelism, SmolLM2-135M, one core per device: 64 tokens after a 30-token prompt"
    );
    let full = bf16_model(dir);
    let prompt = chat(tok, "What is the capital of France?");
    let new = 64;
    let mut reference: Option<(Vec<u32>, Vec<f32>)> = None;
    println!(
        "   devices   per token   in communication   messages   bytes sent, per device and token"
    );
    for devices in [1, 3] {
        let results = run(devices, |comm: &Comm| {
            let mine = shard(&full, comm, packed);
            let mut pool = SpinPool::new(1);
            let mut cache = KvCache::new(&mine.config, 256);
            let mut scratch = Scratch::new(&mine.config, 64, 256);
            let mut logits = mine
                .forward_last(&mut pool, &prompt, &mut cache, &mut scratch)
                .to_vec();
            let first_logits = logits.clone();
            let before = comm.stats();
            let start = Instant::now();
            let mut out = Vec::new();
            for _ in 0..new {
                let t = argmax(&logits);
                out.push(t);
                logits.copy_from_slice(mine.forward_last(
                    &mut pool,
                    &[t],
                    &mut cache,
                    &mut scratch,
                ));
            }
            let elapsed = start.elapsed();
            let after = comm.stats();
            (
                out,
                first_logits,
                elapsed,
                after.time.saturating_sub(before.time),
                after.messages - before.messages,
                after.bytes - before.bytes,
            )
        });
        let (out, first, elapsed, comm_time, messages, bytes) = &results[0];
        let n = new as u32;
        println!(
            "   {devices:>7} {:>11} {:>18} {:>10} {:>12} B",
            ms(*elapsed / n),
            ms(*comm_time / n),
            messages / new,
            bytes / new
        );
        match &reference {
            None => reference = Some((out.clone(), first.clone())),
            Some((want, logits)) => {
                let diff = first
                    .iter()
                    .zip(logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "   3 devices against 1: largest logit difference {diff:.1e}; same {new} tokens: {}",
                    out == want
                );
            }
        }
    }
    println!();
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Part 3: pipeline parallelism over 1 to 3 stages of one core each, with
/// 1 or 3 sequences in flight.
fn pipeline(dir: &Path, tok: &Tokenizer) {
    println!(
        "== 3. pipeline parallelism, SmolLM2-135M, one core per stage: 32 new tokens per sequence"
    );
    let model = map_matrices(bf16_model(dir), |w: DenseBf16| {
        PackedBf16::new(w.values(), w.rows(), w.cols())
    });
    let questions = [
        "What is the capital of France?",
        "Name three colors.",
        "Why is the sea salty?",
    ];
    let prompts: Vec<Vec<u32>> = questions.iter().map(|q| chat(tok, q)).collect();
    println!(
        "   stages   sequences   decoding: tokens/s   between a sequence's tokens   stages busy"
    );
    let mut reference = None;
    for sequences in [1, 3] {
        for stages in [1, 2, 3] {
            let r = pp::generate(&model, stages, &prompts[..sequences], 32);
            let (rate, gap) = r.decode();
            let busy: Vec<String> = r
                .busy
                .iter()
                .map(|b| format!("{:.0}%", 100.0 * b.as_secs_f64() / r.elapsed.as_secs_f64()))
                .collect();
            println!(
                "   {stages:>6} {sequences:>11} {rate:>20.1} {:>29} {:>13}",
                ms(gap),
                busy.join(" ")
            );
            if sequences == 3 {
                match &reference {
                    None => reference = Some(r.outputs),
                    Some(want) => assert_eq!(&r.outputs, want, "the stages changed the answers"),
                }
            }
        }
    }
    println!("   (the answers are the same with 1, 2 and 3 stages)");
    println!();
}

/// Part 4: a mixture-of-experts layer over 4 devices.
fn experts(dir: &Path, tok: &Tokenizer) {
    let (experts, top_k, ffn, tokens, devices) = (8, 2, 768, 1024, 4);
    println!(
        "== 4. an MoE layer: {experts} experts of width {ffn}, top-{top_k}, on {tokens} token embeddings of the novel"
    );
    let model = bf16_model(dir);
    let h = model.config.hidden_size;
    let ids = tok.encode(TEXT);
    let mut x = vec![0.0; tokens * h];
    for (row, &id) in x.chunks_exact_mut(h).zip(&ids) {
        model.embed.row_to_f32(id as usize, row);
    }
    let moe = Moe::random(h, ffn, experts, top_k, 11);
    let mut pool = SpinPool::new(1);
    let start = Instant::now();
    let (_, load) = moe.forward(&mut pool, &x, tokens, usize::MAX);
    let alone = start.elapsed();
    println!(
        "   tokens per expert: {load:?} (an even split: {})",
        tokens * top_k / experts
    );
    println!("   one device: {}", ms(alone));
    println!(
        "   capacity factor   dropped   time    busiest device's experts   all-to-all, sent per device"
    );
    let even = tokens * top_k / experts;
    for factor in [1.0, 1.25, 2.0, f64::INFINITY] {
        let capacity = if factor.is_finite() {
            (factor * even as f64).ceil() as usize
        } else {
            usize::MAX
        };
        let start = Instant::now();
        let parts = run(devices, |comm: &Comm| {
            let rows = moe::block(tokens, comm.rank(), devices);
            let mut pool = SpinPool::new(1);
            let (_, stats) = forward_ep(
                &moe,
                comm,
                &mut pool,
                &x[rows.start * h..rows.end * h],
                capacity,
            );
            (stats, comm.stats().bytes)
        });
        let elapsed = start.elapsed();
        let dropped: usize = parts.iter().map(|(s, _)| s.dropped).sum();
        let busiest = parts
            .iter()
            .map(|(s, _)| s.expert_time)
            .max()
            .unwrap_or_default();
        let bytes = parts.iter().map(|p| p.1).max().unwrap_or_default();
        println!(
            "   {:>15} {:>8.1}% {:>9} {:>26} {:>23} KB",
            if factor.is_finite() {
                format!("{factor}")
            } else {
                "none".to_string()
            },
            100.0 * dropped as f64 / (tokens * top_k) as f64,
            ms(elapsed),
            ms(busiest),
            bytes / 1024
        );
    }
    println!();
}

/// A link between devices, for the latency-bandwidth model.
struct Link {
    name: &'static str,
    /// Assumed time per all-reduce step (software and hardware latency).
    latency_us: f64,
    /// Assumed bandwidth per device and direction.
    gb_per_s: f64,
}

/// Ring all-reduce of `bytes` over `n` devices: `2 (n - 1)` steps, each
/// paying the latency and moving `bytes / n`.
fn ring_time(link: &Link, n: usize, bytes: f64) -> f64 {
    let steps = 2.0 * (n as f64 - 1.0);
    steps * (link.latency_us * 1e-6 + bytes / n as f64 / (link.gb_per_s * 1e9))
}

/// Part 5: what tensor parallelism's all-reduces would cost on real links.
fn estimates() {
    println!(
        "== 5. estimated communication per decoded token for tensor parallelism (not measured)"
    );
    let links = [
        Link {
            name: "NVLink-class",
            latency_us: 1.0,
            gb_per_s: 400.0,
        },
        Link {
            name: "PCIe-class",
            latency_us: 5.0,
            gb_per_s: 25.0,
        },
        Link {
            name: "100 Gb Ethernet",
            latency_us: 10.0,
            gb_per_s: 12.0,
        },
    ];
    // (name, layers, hidden size, devices, bf16 weight bytes)
    let models = [
        ("SmolLM2-135M on 3 devices", 30, 576, 3, 0.27e9),
        ("a 70B model on 8 devices", 80, 8192, 8, 140e9),
    ];
    for (name, layers, hidden, n, _) in models {
        let bytes = hidden as f64 * 2.0; // one token's activations in bf16
        let count = 2 * layers;
        print!(
            "   {name}: {count} all-reduces of {:.0} KB:",
            bytes / 1024.0
        );
        for link in &links {
            print!(
                "  {} {:.2} ms",
                link.name,
                count as f64 * ring_time(link, n, bytes) * 1e3
            );
        }
        println!();
    }
    println!();
}
