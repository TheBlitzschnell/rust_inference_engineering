//! Measures chapter 14's attention against flash attention on SmolLM2-135M:
//! decode at growing context lengths, prefill, and where the time goes.
//!
//! Run with: cargo run --release -p ch20-flash-attention [profile|decode|ablation|prefill]
//! (needs the model: ./tools/download_model.sh)

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Model, Scratch};
use ch16_real_model::{Placement, load_bf16, model_dir};
use ch17_profiling::{Comparison, TiledBf16, compare, instrument, map_matrices};
use ch20_flash_attention::{FlashOptions, with_flash};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("profile") {
        profile(&dir);
    }
    if run("decode") {
        decode(&dir);
    }
    if run("ablation") {
        ablation(&dir);
    }
    if run("prefill") {
        prefill(&dir);
    }
}

/// SmolLM2 with chapter 17's tiled weights and the built-in attention.
fn builtin(dir: &Path) -> Model<TiledBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        TiledBf16,
    )
}

fn flash(dir: &Path) -> Model<TiledBf16> {
    with_flash(builtin(dir), FlashOptions::default())
}

fn tokens(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 131 + 7) % 49_000) as u32).collect()
}

fn show(what: &str, c: &Comparison, per: u32) {
    println!(
        "   {what}: {:>9.2?} -> {:>9.2?}, speedup {:.2}x (80% of pairs: {:.2}-{:.2}x)",
        c.a.median / per,
        c.b.median / per,
        c.ratio,
        c.ratio_p10,
        c.ratio_p90
    );
}

/// Part 1: at a long context, how much of a decode step is attention?
fn profile(dir: &Path) {
    println!("== 1. a decode step at 4,096 tokens of context, per part");
    for (name, model) in [("built-in", builtin(dir)), ("flash", flash(dir))] {
        // `instrument` rebuilds the model, which drops a custom attention;
        // set it again for the flash model.
        let (timed, timings) = instrument(model);
        let timed = if name == "flash" {
            with_flash(timed, FlashOptions::default())
        } else {
            timed
        };
        let mut pool = SpinPool::with_all_cores();
        let mut cache = KvCache::new(&timed.config, 4200);
        let mut scratch = Scratch::new(&timed.config, 512, 4200);
        timed.forward_last(&mut pool, &tokens(4096), &mut cache, &mut scratch);
        let steps = 32;
        let mut total = Duration::ZERO;
        for _ in 0..2 {
            cache.truncate(4096);
            timings.reset();
            let start = Instant::now();
            for i in 0..steps {
                timed.forward_last(
                    &mut pool,
                    &[(i * 37 + 100) % 49_000],
                    &mut cache,
                    &mut scratch,
                );
            }
            total = start.elapsed();
        }
        let matmul: Duration = timings.totals().iter().map(|t| t.time).sum();
        let rest = total.saturating_sub(matmul);
        let per = |d: Duration| d.as_secs_f64() * 1e3 / f64::from(steps);
        println!(
            "   {name:<9} step {:>6.2} ms: matrix products {:>6.2} ms, the rest (mostly attention) {:>6.2} ms ({:.0}%)",
            per(total),
            per(matmul),
            per(rest),
            100.0 * rest.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!();
}

/// Part 2: decode step time against context length, interleaved.
fn decode(dir: &Path) {
    println!("== 2. one decode step, built-in -> flash attention (4 threads)");
    let (a, b) = (builtin(dir), flash(dir));
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&a.config, 4200);
    let mut scratch = Scratch::new(&a.config, 512, 4200);
    let text = tokens(4096);
    for context in [256, 1024, 2048, 4096] {
        // Extend the cache to `context` tokens (either model fills it with
        // the same keys and values), then decode blocks of 8 tokens from
        // there with each model in turn.
        cache.truncate(context);
        if cache.len() < context {
            let have = cache.len();
            b.forward_last(&mut pool, &text[have..context], &mut cache, &mut scratch);
        }
        let mut block = |m: &Model<TiledBf16>, pool: &mut SpinPool| {
            cache.truncate(context);
            for i in 0..8 {
                m.forward_last(pool, &[(i * 37 + 100) % 49_000], &mut cache, &mut scratch);
            }
        };
        let c = compare_models(&mut pool, 20, &a, &b, &mut block);
        show(&format!("context {context:>4}"), &c, 8);
    }
    println!();
}

/// Part 3: which fix mattered? Decode at 4,096 tokens, built-in against
/// flash with each combination of key splitting and kernels.
fn ablation(dir: &Path) {
    println!("== 3. decode at 4,096 tokens: built-in -> flash variants (4 threads)");
    let a = builtin(dir);
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&a.config, 4200);
    let mut scratch = Scratch::new(&a.config, 512, 4200);
    a.forward_last(&mut pool, &tokens(4096), &mut cache, &mut scratch);
    let variants = [
        ("3 tasks, portable ", 1, false),
        ("9 tasks, portable ", 3, false),
        ("12 tasks, portable", 0, false),
        ("3 tasks, AVX-512  ", 1, true),
        ("9 tasks, AVX-512  ", 3, true),
        ("12 tasks, AVX-512 ", 0, true),
    ];
    for (name, key_splits, simd) in variants {
        let opts = FlashOptions {
            key_splits,
            simd,
            ..FlashOptions::default()
        };
        let b = with_flash(builtin(dir), opts);
        let mut block = |m: &Model<TiledBf16>, pool: &mut SpinPool| {
            cache.truncate(4096);
            for i in 0..8 {
                m.forward_last(pool, &[(i * 37 + 100) % 49_000], &mut cache, &mut scratch);
            }
        };
        let c = compare_models(&mut pool, 16, &a, &b, &mut block);
        show(name, &c, 8);
    }
    println!();
}

/// `compare` with both closures sharing `block` (and so the cache).
fn compare_models<F: FnMut(&Model<TiledBf16>, &mut SpinPool)>(
    pool: &mut SpinPool,
    pairs: usize,
    a: &Model<TiledBf16>,
    b: &Model<TiledBf16>,
    block: &mut F,
) -> Comparison {
    let block = std::cell::RefCell::new(block);
    compare(
        pool,
        pairs,
        |pool| (block.borrow_mut())(a, pool),
        |pool| (block.borrow_mut())(b, pool),
    )
}

/// Part 4: prefill, interleaved.
fn prefill(dir: &Path) {
    println!("== 4. prefill, built-in -> flash attention (4 threads)");
    let (a, b) = (builtin(dir), flash(dir));
    let mut pool = SpinPool::with_all_cores();
    for n in [256, 1024, 2048] {
        let mut cache = KvCache::new(&a.config, n);
        let mut scratch = Scratch::new(&a.config, 512, n);
        let text = tokens(n);
        let mut block = |m: &Model<TiledBf16>, pool: &mut SpinPool| {
            cache.clear();
            m.forward_last(pool, &text, &mut cache, &mut scratch);
        };
        let c = compare_models(&mut pool, if n > 1024 { 5 } else { 10 }, &a, &b, &mut block);
        show(&format!("{n:>4} tokens"), &c, 1);
        println!(
            "      = {:.0} -> {:.0} tokens/s",
            n as f64 / c.a.median.as_secs_f64(),
            n as f64 / c.b.median.as_secs_f64()
        );
    }
}
