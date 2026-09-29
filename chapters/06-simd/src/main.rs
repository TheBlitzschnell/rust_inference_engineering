//! Measures what SIMD buys: in cache, and when streaming from memory.
//!
//! Run with: cargo run --release -p ch06-simd

use std::hint::black_box;
use std::time::Duration;

use ch02_numbers::Bf16;
use ch04_memory::best_of;
use ch06_simd::{
    AlignedVec, Isa, best_isa, dot_accumulators, dot_naive, dot_with, matvec, matvec_bf16,
    random_vec,
};

/// A dot-product implementation to time.
type DotFn = fn(&[f32], &[f32]) -> f32;

/// A matrix-vector product writing into its argument. The `'a` matters:
/// without it, a trait object in a type alias defaults to `'static`, and
/// closures that borrow local weights would be rejected.
type Kernel<'a> = dyn Fn(&mut [f32]) + 'a;

fn gflops(flops: f64, t: Duration) -> f64 {
    flops / t.as_secs_f64() / 1e9
}

fn main() {
    println!("== instruction sets on this machine");
    for isa in Isa::ALL {
        println!("   {isa:?}: {}", isa.is_available());
    }
    println!("   selected: {:?}\n", best_isa());

    dot_in_cache();
    alignment();
    accumulator_sweep();
    matvec_sweep();
}

/// Part 1: one dot product of length 4096 (32 KB of data, fits in L1),
/// repeated so the timer has something to measure.
fn dot_in_cache() {
    let n = 4096;
    let a = AlignedVec::from_slice(&random_vec(n, 1));
    let b = AlignedVec::from_slice(&random_vec(n, 2));
    let reps = 20_000;
    let flops = 2.0 * n as f64 * reps as f64;

    println!("== 1. dot product of two 4096-float vectors (data in L1, 64-byte aligned)");
    let portable: [(&str, DotFn); 5] = [
        ("naive, 1 running sum", dot_naive),
        ("portable, 2 sums", dot_accumulators::<2>),
        ("portable, 4 sums", dot_accumulators::<4>),
        ("portable, 8 sums", dot_accumulators::<8>),
        ("portable, 16 sums", dot_accumulators::<16>),
    ];
    for (name, f) in portable {
        let t = best_of(3, || {
            for _ in 0..reps {
                black_box(f(black_box(&a), black_box(&b)));
            }
        });
        println!("   {name:<24} {:>6.1} GFLOP/s", gflops(flops, t));
    }
    for isa in [Isa::Avx2Fma, Isa::Avx512, Isa::Neon] {
        if !isa.is_available() {
            continue;
        }
        let t = best_of(3, || {
            for _ in 0..reps {
                black_box(dot_with(isa, black_box(&a), black_box(&b)));
            }
        });
        println!(
            "   {:<24} {:>6.1} GFLOP/s",
            format!("{isa:?}"),
            gflops(flops, t)
        );
    }
    println!();
}

/// Part 2: the same kernels on data that starts 0, 16 or 32 bytes past a
/// cache-line boundary.
fn alignment() {
    let n = 4096;
    let a = AlignedVec::from_slice(&random_vec(n + 16, 1));
    let b = AlignedVec::from_slice(&random_vec(n + 16, 2));
    let reps = 20_000;
    let flops = 2.0 * n as f64 * reps as f64;
    println!("== 2. alignment: start address modulo 64 bytes");
    for offset_floats in [0, 4, 8] {
        let (sa, sb) = (
            &a[offset_floats..offset_floats + n],
            &b[offset_floats..offset_floats + n],
        );
        print!("   offset {:>2} bytes:", offset_floats * 4);
        for isa in [Isa::Avx2Fma, Isa::Avx512, Isa::Neon] {
            if !isa.is_available() {
                continue;
            }
            let t = best_of(3, || {
                for _ in 0..reps {
                    black_box(dot_with(isa, black_box(sa), black_box(sb)));
                }
            });
            print!("  {isa:?} {:>5.1} GFLOP/s", gflops(flops, t));
        }
        println!();
    }
    println!();
}

/// Part 3: how many independent FMA chains does the hardware need?
#[cfg(target_arch = "x86_64")]
fn accumulator_sweep() {
    use ch06_simd::x86::dot_avx2_accumulators;
    if !Isa::Avx2Fma.is_available() {
        return;
    }
    let n = 4096;
    let a = AlignedVec::from_slice(&random_vec(n, 1));
    let b = AlignedVec::from_slice(&random_vec(n, 2));
    let reps = 20_000;
    let flops = 2.0 * n as f64 * reps as f64;
    println!("== 3. AVX2 dot product with k independent 8-wide accumulators");
    // SAFETY (each entry): AVX2 and FMA were detected just above.
    let kernels: [(usize, DotFn); 5] = [
        (1, |a, b| unsafe { dot_avx2_accumulators::<1>(a, b) }),
        (2, |a, b| unsafe { dot_avx2_accumulators::<2>(a, b) }),
        (4, |a, b| unsafe { dot_avx2_accumulators::<4>(a, b) }),
        (8, |a, b| unsafe { dot_avx2_accumulators::<8>(a, b) }),
        (16, |a, b| unsafe { dot_avx2_accumulators::<16>(a, b) }),
    ];
    for (k, f) in kernels {
        let t = best_of(3, || {
            for _ in 0..reps {
                black_box(f(black_box(&a), black_box(&b)));
            }
        });
        println!("   k = {k:>2}: {:>6.1} GFLOP/s", gflops(flops, t));
    }
    println!();
}

#[cfg(not(target_arch = "x86_64"))]
fn accumulator_sweep() {}

/// Part 4: matrix-vector products at three sizes, f32 and bf16 weights.
fn matvec_sweep() {
    println!("== 4. matrix-vector product y = W x (one core, aligned weights)");
    println!("   weights             | kernel       |     time | GB/s of weights | GFLOP/s");
    for (rows, cols, label) in [
        (512, 512, "in L2"),
        (4096, 4096, "in L3"),
        (8192, 8192, "from DRAM"),
    ] {
        let w = AlignedVec::from_slice(&random_vec(rows * cols, 3));
        let w16 = AlignedVec::from_fn(w.len(), |i| Bf16::from_f32(w[i]));
        let x = AlignedVec::from_slice(&random_vec(cols, 4));
        let mut y = vec![0.0f32; rows];
        let flops = 2.0 * (rows * cols) as f64;
        let reps = (1 << 26) / (rows * cols) + 1;

        let portable = |y: &mut [f32]| {
            for (row, out) in w.chunks_exact(cols).zip(y.iter_mut()) {
                *out = dot_accumulators::<8>(row, &x);
            }
        };
        let runs: [(&str, usize, &Kernel<'_>); 3] = [
            ("f32 portable", 4, &portable),
            ("f32 SIMD", 4, &|y: &mut [f32]| matvec(&w, &x, y)),
            ("bf16 SIMD", 2, &|y: &mut [f32]| matvec_bf16(&w16, &x, y)),
        ];
        for (name, bytes_per_weight, f) in runs {
            f(&mut y); // warm-up
            let t = best_of(3, || {
                for _ in 0..reps {
                    f(black_box(&mut y));
                }
            }) / reps as u32;
            let bytes = (rows * cols * bytes_per_weight) as f64;
            println!(
                "   {:>4}x{:<4} {label:<9} | {name:<12} | {:>8} | {:>15.1} | {:>7.1}",
                rows,
                cols,
                format!("{t:.2?}"),
                bytes / t.as_secs_f64() / 1e9,
                gflops(flops, t)
            );
        }
    }
}
