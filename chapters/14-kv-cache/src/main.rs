//! Measures what the KV cache costs (memory) and what it buys (speed), on a
//! randomly initialized model of SmolLM2-135M's shape.
//!
//! Run with: cargo run --release -p ch14-kv-cache

use ch07_threads::SpinPool;
use ch14_kv_cache::{Config, DenseF32, KvCache, Model, Scratch, argmax, generate_greedy};
use std::time::{Duration, Instant};

fn main() {
    memory_table();

    let config = Config::smollm2_135m();
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let mut pool = SpinPool::new(cores);
    let reference = ch13_transformer::Weights::random(&config, 7);
    let model = Model::from_reference(&reference);
    println!("(random SmolLM2-135M-shaped weights, f32, {cores} threads)\n");

    cache_versus_no_cache(&model, &reference, &mut pool);
    drop(reference);
    prefill_versus_decode(&model, &mut pool);
    chunk_sizes(&model, &mut pool);
    decode_versus_context(&model, &mut pool);
}

/// Part 1: KV cache size for some real model shapes.
fn memory_table() {
    println!("== 1. KV cache size in bf16 (2 bytes per value)");
    println!(
        "   {:<26} {:>6} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "model", "layers", "kv heads", "head dim", "per token", "4k tokens", "128k tokens"
    );
    // (name, layers, kv heads, head dim)
    let models = [
        ("SmolLM2-135M", 30, 3, 64),
        ("SmolLM2-360M", 32, 5, 64),
        ("Llama-3.1-8B", 32, 8, 128),
        ("Llama-3.1-8B without GQA", 32, 32, 128),
        ("Llama-3.1-70B", 80, 8, 128),
    ];
    for (name, layers, kv_heads, head_dim) in models {
        let per_token = 2 * layers * kv_heads * head_dim * 2;
        println!(
            "   {name:<26} {layers:>6} {kv_heads:>8} {head_dim:>8} {:>12} {:>12} {:>12}",
            human(per_token),
            human(per_token * 4096),
            human(per_token * 131_072)
        );
    }
    println!();
}

/// Part 2: the same greedy generation with and without the cache.
fn cache_versus_no_cache(
    model: &Model<DenseF32>,
    reference: &ch13_transformer::Weights,
    pool: &mut SpinPool,
) {
    let c = &model.config;
    let prompt = prompt(32, c);
    let new_tokens = 32;

    let start = Instant::now();
    let uncached =
        ch13_transformer::generate_without_cache(reference, pool, &prompt, new_tokens, |_, _| {});
    let no_cache_time = start.elapsed();

    let mut cache = KvCache::new(c, 64);
    let mut scratch = Scratch::new(c, 64, 64);
    let mut steps = Vec::new();
    let start = Instant::now();
    let cached = generate_greedy(
        model,
        pool,
        &prompt,
        new_tokens,
        &mut cache,
        &mut scratch,
        |_, t| steps.push(t),
    );
    let cache_time = start.elapsed();

    println!("== 2. 32-token prompt, 32 new tokens, greedy");
    println!("   without a cache (chapter 13): {no_cache_time:>9.1?}");
    println!(
        "   with the KV cache:            {cache_time:>9.1?}  ({:.1}x faster)",
        no_cache_time.as_secs_f64() / cache_time.as_secs_f64()
    );
    let first = steps[0];
    let decode = median(&mut steps[1..]);
    println!(
        "     first token (prefill of 32): {first:>8.1?}   later tokens (decode): median {decode:.1?}"
    );
    println!("   same tokens generated: {}", cached == uncached);
    println!();
}

/// Part 3: processing tokens in a batch (prefill) versus one at a time.
fn prefill_versus_decode(model: &Model<DenseF32>, pool: &mut SpinPool) {
    let c = &model.config;
    let weight_bytes = model.weight_bytes_per_token();
    let flops_per_token = c.flops_per_token(0);
    let mut cache = KvCache::new(c, 1024);
    let mut scratch = Scratch::new(c, 512, 1024);

    println!("== 3. prefill versus decode");
    for n in [32, 128, 512] {
        let prompt = prompt(n, c);
        let t = best_of(if n > 128 { 1 } else { 3 }, || {
            cache.clear();
            model.forward_last(pool, &prompt, &mut cache, &mut scratch);
        });
        println!(
            "   prefill {n:>3} tokens: {:>8.1?}  {:>6.0} tokens/s  {:>5.1} GFLOP/s",
            t,
            n as f64 / t.as_secs_f64(),
            n as f64 * flops_per_token / t.as_secs_f64() / 1e9
        );
    }
    let t = decode_step_time(model, pool, &mut cache, &mut scratch, 32);
    println!(
        "   decode, 1 token:     {:>8.1?}  {:>6.0} tokens/s  {:>5.1} GFLOP/s  {:.1} GB/s of weights",
        t,
        1.0 / t.as_secs_f64(),
        flops_per_token / t.as_secs_f64() / 1e9,
        weight_bytes as f64 / t.as_secs_f64() / 1e9
    );
    println!(
        "   (each token needs {:.2} GFLOP and reads {} of weights)",
        flops_per_token / 1e9,
        human(weight_bytes)
    );
    println!();
}

/// Part 4: a 256-token prompt, prefilled in chunks of different sizes.
fn chunk_sizes(model: &Model<DenseF32>, pool: &mut SpinPool) {
    let c = &model.config;
    let prompt = prompt(256, c);
    let mut cache = KvCache::new(c, 256);
    println!("== 4. prefill of 256 tokens with different chunk sizes");
    for chunk in [1, 4, 16, 64, 256] {
        let mut scratch = Scratch::new(c, chunk, 256);
        let t = best_of(2, || {
            cache.clear();
            model.forward_last(pool, &prompt, &mut cache, &mut scratch);
        });
        println!(
            "   chunk {chunk:>3}: {:>8.1?}  {:>6.0} tokens/s  scratch {}",
            t,
            256.0 / t.as_secs_f64(),
            human(scratch.bytes())
        );
    }
    println!();
}

/// Part 5: the cost of one decode step as the context grows.
fn decode_versus_context(model: &Model<DenseF32>, pool: &mut SpinPool) {
    let c = &model.config;
    let max = 2048;
    let text = prompt(max, c);
    let mut cache = KvCache::new(c, max + 64);
    let mut scratch = Scratch::new(c, 256, max + 64);
    println!("== 5. one decode step at different context lengths");
    let mut base = None;
    for context in [16, 256, 1024, 2048] {
        // Extend the sequence already in the cache up to `context` tokens:
        // only the new part is processed.
        model.forward_last(pool, &text[cache.len()..context], &mut cache, &mut scratch);
        let t = decode_step_time(model, pool, &mut cache, &mut scratch, 32);
        let base = *base.get_or_insert(t);
        println!(
            "   context {context:>4}: {:>7.1?} per token ({:.2}x)  cache holds {:>8} (f32)",
            t,
            t.as_secs_f64() / base.as_secs_f64(),
            human(cache.len() * 2 * c.num_layers * c.kv_dim() * 4)
        );
    }
}

/// Median time of `steps` single-token decode steps, continuing the sequence
/// in `cache` (which is rolled back afterwards).
fn decode_step_time(
    model: &Model<DenseF32>,
    pool: &mut SpinPool,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    steps: usize,
) -> Duration {
    if cache.is_empty() {
        model.forward_last(pool, &[1], cache, scratch);
    }
    let len = cache.len();
    let mut next = 1;
    let mut times: Vec<Duration> = (0..steps)
        .map(|_| {
            let start = Instant::now();
            next = argmax(model.forward_last(pool, &[next], cache, scratch));
            start.elapsed()
        })
        .collect();
    cache.truncate(len);
    median(&mut times)
}

/// A deterministic "prompt" of `n` token ids.
fn prompt(n: usize, c: &Config) -> Vec<u32> {
    (0..n)
        .map(|i| ((i * 131 + 7) % c.vocab_size) as u32)
        .collect()
}

fn best_of(runs: usize, mut f: impl FnMut()) -> Duration {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}

fn human(bytes: usize) -> String {
    let b = bytes as f64;
    if b >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1} GiB", b / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024.0 * 1024.0 {
        format!("{:.1} MiB", b / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KiB", b / 1024.0)
    }
}
