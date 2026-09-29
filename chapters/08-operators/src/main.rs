//! Operators: stability, accuracy and speed.
//!
//! Run with: cargo run --release -p ch08-operators

use std::hint::black_box;

use ch04_memory::best_of;
use ch08_operators::{
    exp_fast, gelu_erf, gelu_tanh, rms_norm, rms_norm_alloc, softmax, softmax_fast, softmax_naive,
};

fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 * 30.0 - 15.0
        })
        .collect()
}

fn main() {
    println!("== 1. softmax of [1000, 999, 998]");
    let mut naive = [1000.0f32, 999.0, 998.0];
    softmax_naive(&mut naive);
    let mut stable = [1000.0f32, 999.0, 998.0];
    softmax(&mut stable);
    println!("   naive:  {naive:?}");
    println!("   stable: {stable:?}\n");

    println!("== 2. exp over 1,000,000 values");
    let xs = logits(1_000_000, 1);
    let mut out = vec![0.0f32; xs.len()];
    let std_exp = best_of(5, || {
        for (o, &x) in out.iter_mut().zip(black_box(&xs)) {
            *o = x.exp();
        }
        black_box(&out);
    });
    let fast_exp = best_of(5, || {
        for (o, &x) in out.iter_mut().zip(black_box(&xs)) {
            *o = exp_fast(x);
        }
        black_box(&out);
    });
    println!(
        "   f32::exp: {std_exp:>8.2?}  ({:.2} ns each)",
        std_exp.as_secs_f64() * 1e9 / 1e6
    );
    println!(
        "   exp_fast: {fast_exp:>8.2?}  ({:.2} ns each)\n",
        fast_exp.as_secs_f64() * 1e9 / 1e6
    );

    println!("== 3. softmax speed");
    for (label, n) in [
        ("attention row, 1024 keys", 1024),
        ("vocabulary, 49152 tokens", 49_152),
    ] {
        let base = logits(n, 2);
        let mut buf = base.clone();
        let reps = 200;
        let plain = best_of(3, || {
            for _ in 0..reps {
                buf.copy_from_slice(&base);
                softmax(black_box(&mut buf));
            }
        }) / reps;
        let fast = best_of(3, || {
            for _ in 0..reps {
                buf.copy_from_slice(&base);
                softmax_fast(black_box(&mut buf));
            }
        }) / reps;
        println!(
            "   {label:<26} softmax {plain:>9.2?}   softmax_fast {fast:>9.2?}   ({:.1}x)",
            plain.as_secs_f64() / fast.as_secs_f64()
        );
    }
    println!();

    println!("== 4. RMSNorm of a 576-wide vector: allocating vs reusing the output buffer");
    let x = logits(576, 3);
    let w = vec![1.0f32; 576];
    let mut out = vec![0.0f32; 576];
    let reps = 100_000;
    let reuse = best_of(5, || {
        for _ in 0..reps {
            rms_norm(black_box(&x), &w, 1e-5, &mut out);
            black_box(&out);
        }
    }) / reps;
    let alloc = best_of(5, || {
        for _ in 0..reps {
            black_box(rms_norm_alloc(black_box(&x), &w, 1e-5));
        }
    }) / reps;
    println!("   into a reused buffer: {reuse:>8.2?}");
    println!(
        "   into a new Vec:       {alloc:>8.2?}   (+{:.0} ns per call)\n",
        (alloc.as_secs_f64() - reuse.as_secs_f64()) * 1e9
    );

    println!("== 5. GELU: tanh approximation vs exact (erf)");
    let xs: Vec<f32> = (0..=12_000).map(|i| -6.0 + i as f32 * 0.001).collect();
    let (mut a, mut b) = (xs.clone(), xs.clone());
    gelu_tanh(&mut a);
    gelu_erf(&mut b);
    let (worst, at) = a
        .iter()
        .zip(&b)
        .zip(&xs)
        .map(|((p, q), &x)| ((p - q).abs(), x))
        .fold((0.0f32, 0.0f32), |acc, v| if v.0 > acc.0 { v } else { acc });
    println!("   largest difference {worst:.2e} at x = {at:.3}");
}
