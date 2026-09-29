//! What threads buy for memory-bound and compute-bound kernels, and what
//! they cost.
//!
//! Run with: cargo run --release -p ch07-threads

use std::hint::black_box;
use std::time::{Duration, Instant};

use ch02_numbers::Bf16;
use ch04_memory::best_of;
use ch06_simd::{AlignedVec, random_vec};
use ch07_threads::{
    SpinPool, count_adjacent, count_padded, matmul_nt_pool, matmul_nt_pool_bf16, matvec_rayon,
    matvec_scoped, matvec_serial,
};
use rayon::prelude::*;

fn per_call(total: Duration, calls: u32) -> Duration {
    total / calls
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    println!("cores: {cores}\n");
    overhead(cores);
    matvec_sizes(cores);
    scaling(cores);
    false_sharing(cores);
}

/// Part 1: the fixed cost of going parallel, with no work to do.
fn overhead(cores: usize) {
    println!("== 1. cost of one parallel call that does nothing");
    let calls = 2000;
    let t = best_of(3, || {
        for _ in 0..calls {
            std::thread::scope(|s| {
                for _ in 0..cores {
                    s.spawn(|| black_box(1));
                }
            });
        }
    });
    println!(
        "   spawn + join {cores} OS threads:  {:>8.2?}",
        per_call(t, calls)
    );
    // Make sure rayon's pool exists before timing it.
    (0..cores).into_par_iter().for_each(|i| {
        black_box(i);
    });
    let t = best_of(3, || {
        for _ in 0..calls {
            (0..cores).into_par_iter().for_each(|i| {
                black_box(i);
            });
        }
    });
    println!(
        "   rayon, {cores} tasks on the pool:  {:>8.2?}",
        per_call(t, calls)
    );
    let mut pool = SpinPool::new(cores);
    let t = best_of(3, || {
        for _ in 0..calls {
            pool.run(&|i| {
                black_box(i);
            });
        }
    });
    println!(
        "   spin pool, {cores} threads:        {:>8.2?}",
        per_call(t, calls)
    );
    println!();
}

/// Part 2: one matvec at sizes from SmolLM2-sized to large.
fn matvec_sizes(cores: usize) {
    let mut pool = SpinPool::new(cores);
    println!("== 2. y = W x (f32 weights), {cores} threads where parallel");
    println!(
        "   weights              |   serial | spawned threads |    rayon | spin pool | best GB/s"
    );
    for (rows, cols) in [(576, 576), (1536, 576), (4096, 4096), (8192, 8192)] {
        let w = AlignedVec::from_slice(&random_vec(rows * cols, 1));
        let x = random_vec(cols, 2);
        let mut y = vec![0.0f32; rows];
        let reps = ((1 << 27) / (rows * cols)).max(1) as u32;
        let mut time = |f: &mut dyn FnMut(&mut [f32])| {
            f(&mut y);
            best_of(3, || {
                for _ in 0..reps {
                    f(black_box(&mut y));
                }
            }) / reps
        };
        let serial = time(&mut |y| matvec_serial(&w, &x, y));
        let spawned = time(&mut |y| matvec_scoped(&w, &x, y, cores));
        let pooled = time(&mut |y| matvec_rayon(&w, &x, y));
        let spin = time(&mut |y| matmul_nt_pool(&mut pool, &x, &w, y, 1, cols, rows));
        let best = serial.min(spawned).min(pooled).min(spin);
        let mb = (rows * cols * 4) as f64 / 1e6;
        println!(
            "   {rows:>4}x{cols:<4} ({mb:>5.1} MB) | {:>8} | {:>15} | {:>8} | {:>9} | {:>9.1}",
            format!("{serial:.1?}"),
            format!("{spawned:.1?}"),
            format!("{pooled:.1?}"),
            format!("{spin:.1?}"),
            (rows * cols * 4) as f64 / best.as_secs_f64() / 1e9
        );
    }
    println!();
}

/// Part 3: speedup with 1..=cores threads, for a memory-bound matvec and a
/// compute-bound matmul, on spin pools of each size.
fn scaling(cores: usize) {
    let (rows, cols) = (8192, 8192);
    let w = AlignedVec::from_slice(&random_vec(rows * cols, 3));
    let w16 = AlignedVec::from_fn(w.len(), |i| Bf16::from_f32(w[i]));
    let x = random_vec(cols, 4);
    let mut y = vec![0.0f32; rows];

    let (m, k, n) = (128, 2048, 2048);
    let px = random_vec(m * k, 5);
    let pw = AlignedVec::from_slice(&random_vec(n * k, 6));
    let pw16 = AlignedVec::from_fn(pw.len(), |i| Bf16::from_f32(pw[i]));
    let mut py = vec![0.0f32; m * n];
    let flops = 2.0 * (m * k * n) as f64;

    println!("== 3. scaling with the number of threads (spin pools of each size)");
    println!(
        "   threads | matvec f32 256 MB  | matvec bf16 128 MB | matmul f32 128x2048x2048 | matmul bf16"
    );
    for threads in 1..=cores {
        let mut pool = SpinPool::new(threads);
        let mut run = |f: &mut dyn FnMut(&mut SpinPool)| {
            f(&mut pool);
            best_of(3, || f(&mut pool))
        };
        let mv = run(&mut |p| matmul_nt_pool(p, black_box(&x), &w, &mut y, 1, cols, rows));
        let mv16 = run(&mut |p| matmul_nt_pool_bf16(p, black_box(&x), &w16, &mut y, 1, cols, rows));
        let mm = run(&mut |p| matmul_nt_pool(p, black_box(&px), &pw, &mut py, m, k, n));
        let mm16 = run(&mut |p| matmul_nt_pool_bf16(p, black_box(&px), &pw16, &mut py, m, k, n));
        println!(
            "   {threads:>7} | {:>7.1?} {:>5.1} GB/s | {:>7.1?} {:>5.1} GB/s | {:>9.1?} {:>6.1} GFLOP/s | {:>6.1} GFLOP/s",
            mv,
            (rows * cols * 4) as f64 / mv.as_secs_f64() / 1e9,
            mv16,
            (rows * cols * 2) as f64 / mv16.as_secs_f64() / 1e9,
            mm,
            flops / mm.as_secs_f64() / 1e9,
            flops / mm16.as_secs_f64() / 1e9,
        );
    }
    println!();
}

/// Part 4: false sharing.
fn false_sharing(cores: usize) {
    let iters = 20_000_000;
    println!("== 4. {cores} threads each incrementing their own counter {iters} times");
    let start = Instant::now();
    black_box(count_adjacent(cores, iters));
    let adjacent = start.elapsed();
    let start = Instant::now();
    black_box(count_padded(cores, iters));
    let padded = start.elapsed();
    println!("   counters side by side (one cache line): {adjacent:>8.1?}");
    println!("   each counter on its own cache line:     {padded:>8.1?}");
    println!(
        "   the shared cache line costs {:.1}x",
        adjacent.as_secs_f64() / padded.as_secs_f64()
    );
}
