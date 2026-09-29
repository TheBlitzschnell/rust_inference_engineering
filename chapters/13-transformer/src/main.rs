//! Counts SmolLM2's parameters and FLOPs, then generates from a
//! randomly initialized model of the same shape without any caching.
//!
//! Run with: cargo run --release -p ch13-transformer

use ch07_threads::SpinPool;
use ch13_transformer::{Config, Weights, generate_without_cache};

fn main() {
    let c = Config::smollm2_135m();
    breakdown(&c);
    no_cache_generation(&c);
}

/// Part 1: where SmolLM2's parameters and FLOPs are.
fn breakdown(c: &Config) {
    let h = c.hidden_size;
    let embed = c.vocab_size * h;
    let attn = h * c.q_dim() + 2 * h * c.kv_dim() + c.q_dim() * h;
    let mlp = 3 * h * c.intermediate_size;
    let norms = 2 * h;
    let total = c.param_count();
    let pct = |n: usize| 100.0 * n as f64 / total as f64;
    println!("== 1. SmolLM2-135M: {total} parameters");
    println!(
        "   embedding (also the output layer): {embed:>11}  {:>5.1}%",
        pct(embed)
    );
    println!(
        "   attention, {} layers x {attn:>9}: {:>11}  {:>5.1}%",
        c.num_layers,
        c.num_layers * attn,
        pct(c.num_layers * attn)
    );
    println!(
        "   MLP,       {} layers x {mlp:>9}: {:>11}  {:>5.1}%",
        c.num_layers,
        c.num_layers * mlp,
        pct(c.num_layers * mlp)
    );
    println!(
        "   norms:                             {:>11}  {:>5.1}%",
        c.num_layers * norms + h,
        pct(c.num_layers * norms + h)
    );
    for context in [0, 1_000, 8_000] {
        println!(
            "   FLOPs for one token with {context:>5} tokens of context: {:.3} GFLOP",
            c.flops_per_token(context) / 1e9
        );
    }
    println!();
}

/// Part 2: generating without a cache repeats all the earlier work.
fn no_cache_generation(c: &Config) {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let mut pool = SpinPool::new(cores);
    let start = std::time::Instant::now();
    let weights = Weights::random(c, 7);
    println!(
        "== 2. greedy generation with random SmolLM2-shaped weights, no cache ({cores} threads)"
    );
    println!("   (building random weights took {:.1?})", start.elapsed());
    let prompt: Vec<u32> = (0..32)
        .map(|i| (i * 131 + 7) % c.vocab_size as u32)
        .collect();
    let mut total_positions = 0;
    let mut first = None;
    let out = generate_without_cache(&weights, &mut pool, &prompt, 32, |len, t| {
        total_positions += len;
        let base = *first.get_or_insert(t);
        if len % 4 == 0 || len == 32 {
            println!(
                "   step {:>2}: ran the model on {len:>2} tokens in {t:>8.1?} ({:.1}x the first step)",
                len - 31,
                t.as_secs_f64() / base.as_secs_f64()
            );
        }
    });
    println!(
        "   generated {} tokens; the model processed {total_positions} token positions to do it",
        out.len() - prompt.len()
    );
}
