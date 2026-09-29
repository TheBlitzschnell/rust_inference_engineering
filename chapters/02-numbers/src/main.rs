//! Experiments with number formats.
//!
//! Run with: cargo run --release -p ch02-numbers

use std::hint::black_box;
use std::time::Instant;

use ch02_numbers::{
    Bf16, F16, Fp8E4M3, XorShift, bf16_to_f32_slice, f16_to_f32_slice, f32_fields,
    sum_bf16_accumulator, sum_f32, sum_f64, sum_kahan, sum_pairwise,
};

fn main() {
    show_bit_layouts();
    show_rounding_error();
    show_overflow();
    show_accumulation();
    show_order_matters();
    show_model_sizes();
    show_conversion_speed();
}

/// Part 1: what the bits of a few numbers look like in each format.
fn show_bit_layouts() {
    println!("== 1. bit layouts (sign | exponent | mantissa)");
    for x in [1.0f32, 0.1, -2.5, std::f32::consts::PI, 65504.0, 1e-6] {
        let f = f32_fields(x);
        let b = Bf16::from_f32(x);
        let h = F16::from_f32(x);
        println!(
            "   {x:>12e}  f32 {}|{:08b}|{:023b}",
            f.sign, f.exponent, f.mantissa
        );
        println!(
            "   {:>12}  bf16 {}|{:08b}|{:07b}         = {:e}",
            "",
            b.to_bits() >> 15,
            (b.to_bits() >> 7) & 0xFF,
            b.to_bits() & 0x7F,
            b.to_f32()
        );
        println!(
            "   {:>12}  f16  {}|{:05b}|{:010b}         = {:e}",
            "",
            h.to_bits() >> 15,
            (h.to_bits() >> 10) & 0x1F,
            h.to_bits() & 0x3FF,
            h.to_f32()
        );
    }
    println!();
}

/// A conversion from f32 to some format and back.
type Convert = fn(f32) -> f32;

/// Part 2: how much error each format adds to typical weight values.
fn show_rounding_error() {
    let mut rng = XorShift::new(42);
    // Weights in trained models are mostly small numbers around zero.
    let weights: Vec<f32> = (0..1_000_000).map(|_| rng.next_f32() * 0.1).collect();

    println!("== 2. rounding 1,000,000 weights in [-0.1, 0.1)");
    println!("   format          | mean rel. error | max rel. error");
    let formats: [(&str, Convert); 4] = [
        ("bf16 (truncate)", |x| Bf16::from_f32_truncate(x).to_f32()),
        ("bf16 (round)", |x| Bf16::from_f32(x).to_f32()),
        ("f16", |x| F16::from_f32(x).to_f32()),
        ("fp8 e4m3", |x| Fp8E4M3::from_f32(x).to_f32()),
    ];
    for (name, convert) in formats {
        let mut sum = 0.0f64;
        let mut max = 0.0f64;
        for &w in &weights {
            let rel = f64::from((convert(w) - w).abs() / w.abs().max(1e-12));
            sum += rel;
            max = max.max(rel);
        }
        println!(
            "   {name:<15} | {:>15.2e} | {:>14.2e}",
            sum / weights.len() as f64,
            max
        );
    }
    println!();
}

/// Part 3: f16 runs out of range where bf16 does not.
fn show_overflow() {
    println!("== 3. range: squaring an activation of 300");
    let a = 300.0f32;
    let in_f16 = F16::from_f32(F16::from_f32(a).to_f32() * F16::from_f32(a).to_f32());
    let in_bf16 = Bf16::from_f32(Bf16::from_f32(a).to_f32() * Bf16::from_f32(a).to_f32());
    println!("   f32:  {}", a * a);
    println!("   f16:  {}   (largest f16 is 65504)", in_f16.to_f32());
    println!("   bf16: {}", in_bf16.to_f32());
    println!();
}

/// Part 4: where you keep the running total matters more than where you
/// keep the inputs.
fn show_accumulation() {
    println!("== 4. accumulation");
    let ones = vec![1.0f32; 1000];
    println!(
        "   1000 x 1.0 : f32 total {}, bf16 total {}",
        sum_f32(&ones),
        sum_bf16_accumulator(&ones)
    );

    let mut rng = XorShift::new(3);
    let xs: Vec<f32> = (0..10_000_000).map(|_| rng.next_f32().abs()).collect();
    let truth = sum_f64(&xs);
    println!("   10,000,000 values in [0, 1): exact sum (f64) = {truth:.1}");
    for (name, total) in [
        ("plain f32 loop", sum_f32(&xs)),
        ("pairwise f32", sum_pairwise(&xs)),
        ("Kahan f32", sum_kahan(&xs)),
        ("bf16 accumulator", sum_bf16_accumulator(&xs)),
    ] {
        let err = (f64::from(total) - truth).abs() / truth;
        println!("   {name:<17} {total:>14.1}  relative error {err:.1e}");
    }
    println!();
}

/// Part 5: floating-point addition is not associative.
fn show_order_matters() {
    println!("== 5. order of addition");
    let (a, b, c) = (1e8f32, 1.0f32, -1e8f32);
    println!("   (1e8 + 1) - 1e8 = {}", (a + b) + c);
    println!("   (1e8 - 1e8) + 1 = {}", (a + c) + b);
    let mut rng = XorShift::new(9);
    let xs: Vec<f32> = (0..1_000_000).map(|_| rng.next_f32()).collect();
    let forward = sum_f32(&xs);
    let backward = xs.iter().rev().fold(0.0f32, |acc, &x| acc + x);
    println!("   same 1,000,000 numbers, forward {forward}, backward {backward}");
    println!();
}

/// Part 6: bytes per weight decides memory and speed.
fn show_model_sizes() {
    println!("== 6. weight memory by format, and the decode speed limit at 50 GB/s");
    println!("   params |     f32 |    bf16 | int8/fp8 |    int4");
    for (name, params) in [("135M", 135e6), ("1B", 1e9), ("8B", 8e9), ("70B", 70e9)] {
        let gb = |bytes_per: f64| params * bytes_per / 1e9;
        println!(
            "   {name:>6} | {:>6.2}G | {:>6.2}G | {:>7.2}G | {:>6.2}G",
            gb(4.0),
            gb(2.0),
            gb(1.0),
            gb(0.5)
        );
    }
    println!("   tokens/s ceiling for one user at 50 GB/s (bandwidth / weight bytes):");
    for (name, params) in [("135M", 135e6), ("8B", 8e9)] {
        let limit = |bytes_per: f64| 50e9 / (params * bytes_per);
        println!(
            "   {name:>6} | f32 {:>6.1} | bf16 {:>6.1} | int8 {:>6.1} | int4 {:>6.1}",
            limit(4.0),
            limit(2.0),
            limit(1.0),
            limit(0.5)
        );
    }
    println!();
}

/// Part 7: how fast can we widen 16-bit weights to f32?
fn show_conversion_speed() {
    let n = 16 * 1024 * 1024;
    let bf: Vec<Bf16> = (0..n)
        .map(|i| Bf16::from_bits((i % 30000) as u16))
        .collect();
    let hf: Vec<F16> = (0..n).map(|i| F16::from_bits((i % 30000) as u16)).collect();
    let mut out = vec![0.0f32; n];

    println!("== 7. converting 16M values to f32");
    // Warm-up: the first pass over `out` pays for the OS handing us 64 MB of
    // fresh pages. Chapter 4 measures that cost on its own.
    bf16_to_f32_slice(&bf, &mut out);
    f16_to_f32_slice(&hf, &mut out);
    let start = Instant::now();
    for _ in 0..5 {
        bf16_to_f32_slice(black_box(&bf), &mut out);
        black_box(&out);
    }
    let t = start.elapsed() / 5;
    println!(
        "   bf16 -> f32: {t:.2?}  ({:.1} G values/s)",
        n as f64 / t.as_secs_f64() / 1e9
    );
    let start = Instant::now();
    for _ in 0..5 {
        f16_to_f32_slice(black_box(&hf), &mut out);
        black_box(&out);
    }
    let t = start.elapsed() / 5;
    println!(
        "   f16  -> f32: {t:.2?}  ({:.1} G values/s)",
        n as f64 / t.as_secs_f64() / 1e9
    );
}
