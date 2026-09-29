//! Profiles SmolLM2-135M on the engine, tests three ideas, and keeps the
//! one the measurements support.
//!
//! Run with: cargo run --release -p ch17-profiling
//! (needs the model: ./tools/download_model.sh)

use ch02_numbers::Bf16;
use ch06_simd::{AlignedVec, dot_bf16, random_vec};
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch, matmul_pooled};
use ch16_real_model::{DenseBf16, Placement, load_bf16, model_dir};
use ch17_profiling::{
    Comparison, FusedBf16, Rows4Bf16, TiledBf16, Timings, compare, dot4_bf16, instrument,
    map_matrices, measure, tile_bf16,
};
use std::hint::black_box;
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
    let (timed, timings) = instrument(plain(&dir));
    println!("== 1. where a decode step's time goes");
    decode_breakdown(&timed, &timings);
    drop(timed);

    println!("== 2. decode idea 1: stream four rows at once");
    four_rows();
    let rows4 = map_matrices(plain(&dir), Rows4Bf16);
    report(
        "plain vs four-row kernel, decode",
        &decode_ab(&plain(&dir), &rows4),
    );
    println!();

    println!("== 3. decode idea 2: fewer, larger parallel calls");
    per_call_cost();
    let fused = map_matrices(plain(&dir), FusedBf16);
    report(
        "plain vs fused q|k|v and gate|up, decode",
        &decode_ab(&plain(&dir), &fused),
    );
    println!();

    let (timed, timings) = instrument(plain(&dir));
    println!("== 4. where a 256-token prefill's time goes");
    prefill_breakdown(&timed, &timings);
    drop(timed);

    println!("== 5. prefill idea: a 4 x 4 tile kernel");
    tile_speed();
    let (a, b) = (plain(&dir), map_matrices(plain(&dir), TiledBf16));
    for n in [40, 256] {
        report(
            &format!("plain vs tiled, {n}-token prefill"),
            &prefill_ab(&a, &b, n),
        );
    }
    report(
        "plain vs tiled, decode (same code path)",
        &decode_ab(&a, &b),
    );
    println!();
    drop((a, b));

    let (timed, timings) = instrument(map_matrices(plain(&dir), TiledBf16));
    println!("== 6. where a 256-token prefill's time goes now");
    prefill_breakdown(&timed, &timings);
}

fn plain(dir: &Path) -> Model<DenseBf16> {
    load_bf16(dir, Placement::Mapped).expect("model").0
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 131 + 7) % 49_000) as u32).collect()
}

fn report(what: &str, c: &Comparison) {
    println!(
        "   {what}: {:.2?} -> {:.2?}, speedup {:.2}x (80% of pairs: {:.2}-{:.2}x)",
        c.a.median, c.b.median, c.ratio, c.ratio_p10, c.ratio_p90
    );
}

/// Part 1: decodes 64 tokens and splits the time per kind of matrix.
fn decode_breakdown<W: Matrix>(model: &Model<W>, timings: &Timings) {
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&model.config, 512);
    let mut scratch = Scratch::new(&model.config, 256, 512);
    let steps = 64;
    // Two identical rounds: the first warms up, the second is measured.
    let mut total = Duration::ZERO;
    for _ in 0..2 {
        cache.clear();
        model.forward_last(&mut pool, &prompt(40), &mut cache, &mut scratch);
        timings.reset();
        let start = Instant::now();
        for i in 0..steps {
            let token = (i * 37 + 100) % 49_000;
            model.forward_last(&mut pool, &[token], &mut cache, &mut scratch);
        }
        total = start.elapsed();
    }
    table(model, timings, total, steps as usize, 1);
}

/// Part 4: one 256-token prefill, after two warm-up runs, split per matrix.
fn prefill_breakdown<W: Matrix>(model: &Model<W>, timings: &Timings) {
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&model.config, 512);
    let mut scratch = Scratch::new(&model.config, 256, 512);
    let tokens = prompt(256);
    let mut total = Duration::ZERO;
    for _ in 0..3 {
        cache.clear();
        timings.reset();
        let start = Instant::now();
        model.forward_last(&mut pool, &tokens, &mut cache, &mut scratch);
        total = start.elapsed();
    }
    table(model, timings, total, 1, 256);
}

/// Prints one row per kind of matrix, per step (`steps` decode steps) or
/// per prefill of `tokens` tokens. Decode reports GB/s of weights read;
/// prefill reports GFLOP/s (2 FLOPs per weight per token; the LM head
/// multiplies only the last token).
fn table<W: Matrix>(
    model: &Model<W>,
    timings: &Timings,
    total: Duration,
    steps: usize,
    tokens: usize,
) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3 / steps as f64;
    let step = ms(total);
    let prefill = tokens > 1;
    let unit = if prefill { "GFLOP/s" } else { "GB/s" };
    println!(
        "   {:<9} {:>9} {:>7} {:>8}",
        "matrices", "ms", "share", unit
    );
    let mut matmul = Duration::ZERO;
    for t in timings.totals() {
        matmul += t.time;
        let secs = t.time.as_secs_f64();
        let rate = if prefill {
            let rows = if t.name == "lm_head" { 1 } else { tokens };
            // bf16: bytes / 2 weights, 2 FLOPs each per token.
            t.bytes as f64 * rows as f64 / secs / 1e9
        } else {
            t.bytes as f64 / secs / 1e9
        };
        println!(
            "   {:<9} {:>9.3} {:>6.1}% {:>8.1}",
            t.name,
            ms(t.time),
            100.0 * ms(t.time) / step,
            rate
        );
    }
    let rest = ms(total.saturating_sub(matmul));
    println!(
        "   {:<9} {:>9.3} {:>6.1}%   attention, norms, RoPE, residuals, lookup",
        "the rest",
        rest,
        100.0 * rest / step
    );
    if prefill {
        println!(
            "   {:<9} {:>9.3}  = {:.0} tokens/s",
            "prefill",
            step,
            tokens as f64 / (step / 1e3)
        );
    } else {
        println!(
            "   {:<9} {:>9.3}  = {:.1} tokens/s, {:.1} GB/s of weights overall",
            "step",
            step,
            1e3 / step,
            model.weight_bytes_per_token() as f64 / (step / 1e3) / 1e9
        );
    }
    println!();
}

/// Part 2: the four-row kernel on rows of different lengths, one thread.
fn four_rows() {
    for k in [576, 1536, 8192] {
        let n = (128 << 20) / k; // 256 MiB: far more than the caches hold
        let w: Vec<Bf16> = random_vec(n * k, 3)
            .into_iter()
            .map(Bf16::from_f32)
            .collect();
        let x = random_vec(k, 4);
        let mut y = vec![0.0; n];
        let one = measure(1, 4, || {
            for (row, out) in w.chunks_exact(k).zip(y.iter_mut()) {
                *out = dot_bf16(row, &x);
            }
            black_box(&y);
        });
        let four = measure(1, 4, || {
            for (quad, out) in w.chunks_exact(4 * k).zip(y.chunks_exact_mut(4)) {
                let (a, rest) = quad.split_at(k);
                let (b, rest) = rest.split_at(k);
                let (c, d) = rest.split_at(k);
                out.copy_from_slice(&dot4_bf16([a, b, c, d], &x));
            }
            black_box(&y);
        });
        let gbps = |t: Duration| (n * k * 2) as f64 / t.as_secs_f64() / 1e9;
        println!(
            "   one thread, rows of {k:>4} values ({:>5} bytes): one row at a time {:>5.1} GB/s, four rows {:>5.1} GB/s",
            k * 2,
            gbps(one.median),
            gbps(four.median)
        );
    }
}

/// Part 3: the time of one parallel matrix-vector product against its size,
/// on matrices that are not in any cache.
fn per_call_cost() {
    let k = 576;
    let mut pool = SpinPool::with_all_cores();
    let x = random_vec(k, 1);
    let mut scratch = Vec::new();
    let mut points = Vec::new();
    for rows in [64, 192, 576, 1536, 6144, 24_576] {
        let bytes = rows * k * 2;
        // 512 MiB of distinct matrices, visited in turn, so none is cached.
        let count = (512 << 20) / bytes;
        let mats: Vec<AlignedVec<Bf16>> = (0..count)
            .map(|c| AlignedVec::from_fn(rows * k, |i| Bf16::from_f32(((i + c) % 97) as f32)))
            .collect();
        let mut y = vec![0.0; rows];
        // The fastest of five passes: the work is identical each time, so
        // anything slower is interference from outside.
        let t = measure(1, 5, || {
            for m in &mats {
                matmul_pooled(&mut pool, &x, m, &mut y, 1, k, rows, &mut scratch, dot_bf16);
            }
        });
        let per_call = t.min.as_secs_f64() / count as f64;
        points.push((bytes as f64, per_call));
        println!(
            "   4 threads, {rows:>6} x {k} matrix ({:>6.0} KiB): {:>7.1} µs per call, {:>5.1} GB/s",
            bytes as f64 / 1024.0,
            per_call * 1e6,
            bytes as f64 / per_call / 1e9
        );
    }
    // Least-squares line: time = fixed + bytes / bandwidth.
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
    let (mx, my) = (sx / n, sy / n);
    let slope = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / points.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    println!(
        "   fitted line: {:.1} µs per call + bytes at {:.1} GB/s",
        (my - slope * mx) * 1e6,
        1.0 / slope / 1e9
    );
}

/// Part 5: the tile kernel against one dot product per output, one thread,
/// on a prefill-shaped product (1536 x 576 weights, 64 tokens).
fn tile_speed() {
    let (n, k, m) = (1536, 576, 64);
    let w: Vec<Bf16> = random_vec(n * k, 1)
        .into_iter()
        .map(Bf16::from_f32)
        .collect();
    let x = random_vec(m * k, 2);
    let mut y = vec![0.0; m * n];
    let row = |j: usize| &w[j * k..(j + 1) * k];
    let xr = |i: usize| &x[i * k..(i + 1) * k];
    let dots = measure(2, 10, || {
        for j in 0..n {
            for i in 0..m {
                y[i * n + j] = dot_bf16(row(j), xr(i));
            }
        }
        black_box(&y);
    });
    let tiles = measure(2, 10, || {
        for j in (0..n).step_by(4) {
            for i in (0..m).step_by(4) {
                let out = tile_bf16(
                    [row(j), row(j + 1), row(j + 2), row(j + 3)],
                    [xr(i), xr(i + 1), xr(i + 2), xr(i + 3)],
                );
                for (r, o) in out.iter().enumerate() {
                    for (c, &v) in o.iter().enumerate() {
                        y[(i + c) * n + j + r] = v;
                    }
                }
            }
        }
        black_box(&y);
    });
    let gflops = |t: Duration| (2 * m * n * k) as f64 / t.as_secs_f64() / 1e9;
    println!(
        "   one thread, {n} x {k} weights times {m} tokens: one dot product per output {:.1} GFLOP/s, 4 x 4 tiles {:.1} GFLOP/s",
        gflops(dots.median),
        gflops(tiles.median)
    );
}

/// One decode step of `model` on `token`.
fn step<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    token: u32,
    cache: &mut KvCache,
    s: &mut Scratch,
) {
    // Keep the context between 40 and 440 tokens.
    if cache.len() > 440 {
        cache.truncate(40);
    }
    model.forward_last(pool, &[token], cache, s);
}

/// Interleaved A/B of decode: blocks of 16 steps, 30 pairs.
fn decode_ab<A: Matrix, B: Matrix>(a: &Model<A>, b: &Model<B>) -> Comparison {
    let mut pool = SpinPool::with_all_cores();
    let c = &a.config;
    let (mut ca, mut sa) = (KvCache::new(c, 512), Scratch::new(c, 64, 512));
    let (mut cb, mut sb) = (KvCache::new(c, 512), Scratch::new(c, 64, 512));
    let p = prompt(40);
    a.forward_last(&mut pool, &p, &mut ca, &mut sa);
    b.forward_last(&mut pool, &p, &mut cb, &mut sb);
    compare(
        &mut pool,
        30,
        |pool| {
            for i in 0..16 {
                step(a, pool, (i * 37 + 100) % 49_000, &mut ca, &mut sa);
            }
        },
        |pool| {
            for i in 0..16 {
                step(b, pool, (i * 37 + 100) % 49_000, &mut cb, &mut sb);
            }
        },
    )
}

/// Interleaved A/B of an `n`-token prefill, 20 pairs.
fn prefill_ab<A: Matrix, B: Matrix>(a: &Model<A>, b: &Model<B>, n: usize) -> Comparison {
    let mut pool = SpinPool::with_all_cores();
    let c = &a.config;
    let (mut ca, mut sa) = (KvCache::new(c, n), Scratch::new(c, 256, n));
    let (mut cb, mut sb) = (KvCache::new(c, n), Scratch::new(c, 256, n));
    let p = prompt(n);
    compare(
        &mut pool,
        20,
        |pool| {
            ca.clear();
            a.forward_last(pool, &p, &mut ca, &mut sa);
        },
        |pool| {
            cb.clear();
            b.forward_last(pool, &p, &mut cb, &mut sb);
        },
    )
}
