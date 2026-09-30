//! Quantizes SmolLM2-135M to int8 and measures what it gains and loses.
//!
//! Run with: cargo run --release -p ch18-int8 [weights|kernels|speed|quality]
//! (needs the model: ./tools/download_model.sh)

use ch02_numbers::Bf16;
use ch06_simd::{dot_bf16, random_vec};
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch16_real_model::{DenseBf16, Placement, Tokenizer, load_bf16, model_dir};
use ch17_profiling::{TiledBf16, compare, map_matrices, measure};
use ch18_int8::eval::{Quality, log_probs};
use ch18_int8::quant::{dequantize, relative_error};
use ch18_int8::{
    Activations, BLOCK, BlockQ8, Granularity, Q8Matrix, dot_q8_f32, dot_q8_q8, quantize,
    quantize_activations,
};
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

fn main() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    // Optional argument: run one part only (weights, kernels, speed, quality).
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("weights") {
        weight_error(&dir);
    }
    if run("kernels") {
        kernel_speed();
    }
    if run("speed") {
        model_speed(&dir);
    }
    if run("quality") {
        quality(&dir);
    }
}

fn plain(dir: &Path) -> Model<DenseBf16> {
    load_bf16(dir, Placement::Mapped).expect("model").0
}

fn to_f32(m: &DenseBf16) -> Vec<f32> {
    m.values().iter().map(|v| v.to_f32()).collect()
}

fn q8(dir: &Path, granularity: Granularity, activations: Activations) -> Model<Q8Matrix> {
    map_matrices(plain(dir), |m| {
        Q8Matrix::new(&to_f32(&m), m.rows(), m.cols(), granularity, activations)
    })
}

/// Part 1: how far the quantized weights are from the originals.
fn weight_error(dir: &Path) {
    let model = plain(dir);
    let mut matrices: Vec<&DenseBf16> = vec![&model.embed];
    for l in &model.layers {
        matrices.extend([&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]);
    }
    println!(
        "== 1. int8 weights: error by granularity ({} matrices)",
        matrices.len()
    );
    let (mut largest, mut sum_sq, mut count) = (0.0f32, 0.0f64, 0usize);
    for m in &matrices {
        for v in m.values() {
            let v = v.to_f32();
            largest = largest.max(v.abs());
            sum_sq += f64::from(v) * f64::from(v);
            count += 1;
        }
    }
    let rms = (sum_sq / count as f64).sqrt();
    println!(
        "   {count} weights, RMS {rms:.4}, largest magnitude {largest:.2} ({:.0} times the RMS)",
        f64::from(largest) / rms
    );
    for (name, g, bits) in [
        ("per tensor", Granularity::PerTensor, 8.0),
        ("per row", Granularity::PerRow, 8.0),
        ("per block of 64", Granularity::PerBlock, 8.0 + 32.0 / 64.0),
    ] {
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        let mut worst: f64 = 0.0;
        for m in &matrices {
            let values = to_f32(m);
            let back = dequantize(&quantize(&values, m.rows(), m.cols(), g));
            let e = relative_error(&values, &back);
            worst = worst.max(e);
            let n: f64 = values.iter().map(|&v| f64::from(v).powi(2)).sum();
            err += e * e * n;
            norm += n;
        }
        println!(
            "   {name:<16} relative error {:>6.3}% overall, {:>6.3}% in the worst matrix   ({bits} bits per weight)",
            100.0 * (err / norm).sqrt(),
            100.0 * worst
        );
    }
    println!();
}

/// Part 2: one thread, a matrix in cache and one streamed from memory.
fn kernel_speed() {
    println!("== 2. kernels, one thread (matrix-vector product)");
    for (label, rows, cols) in [
        ("in cache: 512 x 1536", 512, 1536),
        ("from memory: 256 MiB of bf16", 87_381, 1536),
    ] {
        let values = random_vec(rows * cols, 5);
        let bf16: Vec<Bf16> = values.iter().map(|&v| Bf16::from_f32(v)).collect();
        let q = quantize(&values, rows, cols, Granularity::PerBlock);
        let x = random_vec(cols, 6);
        let mut xq: Vec<BlockQ8> = Vec::new();
        quantize_activations(&x, cols, false, &mut xq);
        let mut y = vec![0.0; rows];
        let runs = if rows > 1000 { 3 } else { 200 };
        let t_bf16 = measure(1, runs, || {
            for (row, o) in bf16.chunks_exact(cols).zip(y.iter_mut()) {
                *o = dot_bf16(row, &x);
            }
            black_box(&y);
        });
        let per_row = cols / BLOCK;
        let t_w8a32 = measure(1, runs, || {
            for (row, o) in q.chunks_exact(per_row).zip(y.iter_mut()) {
                *o = dot_q8_f32(row, &x);
            }
            black_box(&y);
        });
        let t_w8a8 = measure(1, runs, || {
            for (row, o) in q.chunks_exact(per_row).zip(y.iter_mut()) {
                *o = dot_q8_q8(row, &xq);
            }
            black_box(&y);
        });
        let ops = (2 * rows * cols) as f64;
        let show = |name: &str, t: Duration, bytes: usize| {
            println!(
                "   {label:<30} {name:<6} {:>7.1} GB/s of weights, {:>6.1} G multiply-adds/s",
                bytes as f64 / t.as_secs_f64() / 1e9,
                ops / 2.0 / t.as_secs_f64() / 1e9
            );
        };
        show("bf16", t_bf16.median, rows * cols * 2);
        show("W8A32", t_w8a32.median, q.len() * size_of::<BlockQ8>());
        show("W8A8", t_w8a8.median, q.len() * size_of::<BlockQ8>());
    }
    println!();
}

/// Part 3: the whole model: memory, decode and prefill.
fn model_speed(dir: &Path) {
    let bf16 = plain(dir);
    let w8a32 = q8(dir, Granularity::PerBlock, Activations::Float);
    let w8a8 = q8(dir, Granularity::PerBlock, Activations::Int8PerBlock);
    println!("== 3. SmolLM2-135M, 4 threads");
    println!(
        "   weights read per token: bf16 {:.0} MB, int8 blocks {:.0} MB",
        bf16.weight_bytes_per_token() as f64 / 1e6,
        w8a32.weight_bytes_per_token() as f64 / 1e6
    );
    let decode = |name: &str, c: ch17_profiling::Comparison| {
        println!(
            "   decode, bf16 -> {name}: {:.2?} -> {:.2?} per 16 tokens, speedup {:.2}x (80% of pairs: {:.2}-{:.2}x)",
            c.a.median, c.b.median, c.ratio, c.ratio_p10, c.ratio_p90
        );
    };
    decode("W8A32", decode_ab(&bf16, &w8a32));
    decode("W8A8", decode_ab(&bf16, &w8a8));
    let tiled = map_matrices(plain(dir), TiledBf16);
    let tokens: Vec<u32> = (0..256).map(|i| (i * 131 + 7) % 49_000).collect();
    let rate = |t: Duration| 256.0 / t.as_secs_f64();
    println!(
        "   256-token prefill: bf16 {:.0}, bf16 tiled (ch. 17) {:.0}, W8A32 {:.0}, W8A8 {:.0} tokens/s",
        rate(prefill(&bf16, &tokens)),
        rate(prefill(&tiled, &tokens)),
        rate(prefill(&w8a32, &tokens)),
        rate(prefill(&w8a8, &tokens))
    );
    println!();
}

fn prefill<W: Matrix>(model: &Model<W>, tokens: &[u32]) -> Duration {
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&model.config, tokens.len());
    let mut scratch = Scratch::new(&model.config, tokens.len(), tokens.len());
    measure(1, 3, || {
        cache.clear();
        model.forward_last(&mut pool, tokens, &mut cache, &mut scratch);
    })
    .min
}

fn decode_ab<A: Matrix, B: Matrix>(a: &Model<A>, b: &Model<B>) -> ch17_profiling::Comparison {
    let mut pool = SpinPool::with_all_cores();
    let c = &a.config;
    let (mut ca, mut sa) = (KvCache::new(c, 512), Scratch::new(c, 64, 512));
    let (mut cb, mut sb) = (KvCache::new(c, 512), Scratch::new(c, 64, 512));
    let p: Vec<u32> = (0..40).map(|i| i * 131 + 7).collect();
    a.forward_last(&mut pool, &p, &mut ca, &mut sa);
    b.forward_last(&mut pool, &p, &mut cb, &mut sb);
    let run = |m: &dyn Fn(&mut SpinPool, u32, &mut KvCache, &mut Scratch),
               pool: &mut SpinPool,
               cache: &mut KvCache,
               s: &mut Scratch| {
        if cache.len() > 440 {
            cache.truncate(40);
        }
        for i in 0..16 {
            m(pool, (i * 37 + 100) % 49_000, cache, s);
        }
    };
    let step_a = |pool: &mut SpinPool, t: u32, cache: &mut KvCache, s: &mut Scratch| {
        a.forward_last(pool, &[t], cache, s);
    };
    let step_b = |pool: &mut SpinPool, t: u32, cache: &mut KvCache, s: &mut Scratch| {
        b.forward_last(pool, &[t], cache, s);
    };
    compare(
        &mut pool,
        30,
        |pool| run(&step_a, pool, &mut ca, &mut sa),
        |pool| run(&step_b, pool, &mut cb, &mut sb),
    )
}

/// Part 4: quality on 2,048 tokens of real text, against the bf16 model.
fn quality(dir: &Path) {
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let text = include_str!("../../15-sampling/data/pride-and-prejudice-1-6.txt");
    let all = tok.encode(text);
    let (window, windows) = (512, 4);
    let reference = map_matrices(plain(dir), TiledBf16);
    let variants: Vec<(&str, Model<Q8Matrix>)> = vec![
        (
            "W8A32 per tensor",
            q8(dir, Granularity::PerTensor, Activations::Float),
        ),
        (
            "W8A32 per row",
            q8(dir, Granularity::PerRow, Activations::Float),
        ),
        (
            "W8A32 per block",
            q8(dir, Granularity::PerBlock, Activations::Float),
        ),
        (
            "W8A8 per block",
            q8(dir, Granularity::PerBlock, Activations::Int8PerBlock),
        ),
        (
            "W8A8 per token",
            q8(dir, Granularity::PerBlock, Activations::Int8PerToken),
        ),
    ];
    let mut pool = SpinPool::with_all_cores();
    let c = &reference.config;
    let mut cache = KvCache::new(c, window);
    let mut scratch = Scratch::new(c, window, window);
    let mut results = vec![Quality::default(); variants.len()];
    for w in 0..windows {
        let tokens = &all[w * window..(w + 1) * window];
        let r = log_probs(&reference, &mut pool, tokens, &mut cache, &mut scratch);
        for ((_, model), q) in variants.iter().zip(&mut results) {
            let lp = log_probs(model, &mut pool, tokens, &mut cache, &mut scratch);
            q.add(&r, &lp, tokens, c.vocab_size);
        }
    }
    println!(
        "== 4. quality on {} tokens of Pride and Prejudice (bf16 perplexity {:.3})",
        results[0].tokens,
        results[0].reference_perplexity()
    );
    println!(
        "   {:<18} {:>10} {:>12} {:>14}",
        "weights", "perplexity", "KL (nats)", "same top-1"
    );
    for ((name, _), q) in variants.iter().zip(&results) {
        println!(
            "   {name:<18} {:>10.3} {:>12.5} {:>13.1}%",
            q.candidate_perplexity(),
            q.mean_kl(),
            100.0 * q.top1_agreement()
        );
    }
}
