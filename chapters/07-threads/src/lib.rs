//! Chapter 7: using every core.
//!
//! A matrix-vector product is memory-bound, and chapter 4 showed that one
//! core can only pull about a quarter of the machine's memory bandwidth.
//! So we split the rows of the weight matrix between threads. Each thread
//! reads its own rows and writes its own part of the output: no two threads
//! ever touch the same output element, so no locks are needed, and Rust's
//! borrow checker can verify that.
//!
//! Three ways to run the same work in parallel:
//! - [`matvec_scoped`]: spawn OS threads for every call (`std::thread::scope`),
//! - [`matvec_rayon`] and [`par_matmul_nt`]: hand tasks to rayon's pool of
//!   sleeping threads, which wakes them for each call,
//! - [`SpinPool`] with [`matmul_nt_pool`]: a pool of threads that spin
//!   between jobs, which is what the engine uses from chapter 14 on.

use std::sync::atomic::{AtomicU64, Ordering};

use ch02_numbers::Bf16;
use ch06_simd::{dot, dot_bf16};
use rayon::prelude::*;

pub mod pool;
pub use pool::SpinPool;

/// How many bytes of weights one parallel task should cover. Big enough
/// that scheduling overhead (a few microseconds per task) is small next to
/// the work, small enough that there are many tasks to balance across cores.
pub const TASK_BYTES: usize = 64 * 1024;

/// Rows per task so that each task reads about [`TASK_BYTES`] of weights.
pub fn rows_per_task(row_bytes: usize) -> usize {
    (TASK_BYTES / row_bytes.max(1)).max(1)
}

/// Single-threaded `y = W x` (weights stored as rows), SIMD dot per row.
pub fn matvec_serial(w: &[f32], x: &[f32], y: &mut [f32]) {
    assert_eq!(w.len(), x.len() * y.len());
    for (row, out) in w.chunks_exact(x.len()).zip(y.iter_mut()) {
        *out = dot(row, x);
    }
}

/// `y = W x` with `threads` OS threads spawned for this call.
///
/// `std::thread::scope` lets the threads borrow `w`, `x` and `y` directly:
/// the scope does not return until every thread has finished, so the
/// borrows cannot outlive the data. `chunks_mut` hands each thread its own
/// disjoint piece of `y`.
pub fn matvec_scoped(w: &[f32], x: &[f32], y: &mut [f32], threads: usize) {
    assert_eq!(w.len(), x.len() * y.len());
    let k = x.len();
    let rows_each = y.len().div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (y_part, w_part) in y.chunks_mut(rows_each).zip(w.chunks(rows_each * k)) {
            s.spawn(move || matvec_serial(w_part, x, y_part));
        }
    });
}

/// `y = W x` on rayon's thread pool, in tasks of about [`TASK_BYTES`].
pub fn matvec_rayon(w: &[f32], x: &[f32], y: &mut [f32]) {
    assert_eq!(w.len(), x.len() * y.len());
    let k = x.len();
    let rows = rows_per_task(size_of_val(x));
    y.par_chunks_mut(rows)
        .zip(w.par_chunks(rows * k))
        .for_each(|(y_part, w_part)| matvec_serial(w_part, x, y_part));
}

/// `Y = X · Wᵀ` for `m` input rows, in parallel over blocks of weight rows.
///
/// Each task owns a block of weight rows and computes, for every input row,
/// the outputs of those weight rows. Those outputs are not contiguous in
/// `Y` (they are a block of *columns*), so each task writes them into its own
/// contiguous piece of a transposed buffer `Yᵀ`, and we transpose once at
/// the end. Choosing the layout of intermediate results so that parallel
/// writes are disjoint slices is a standard trick: it keeps the code free of
/// locks and of `unsafe`.
///
/// `dot` computes one output from a weight row and an input row, so the
/// same function serves `f32`, `bf16` and (in later chapters) quantized
/// weights.
pub fn par_matmul_nt_with<W, D>(
    x: &[f32],
    w: &[W],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    dot: D,
) where
    W: Sync,
    D: Fn(&[W], &[f32]) -> f32 + Sync,
{
    assert_eq!(x.len(), m * k, "x must be m×k");
    assert_eq!(w.len(), n * k, "w must be n×k");
    assert_eq!(y.len(), m * n, "y must be m×n");
    let rows = rows_per_task(k * size_of::<W>());
    if m == 1 {
        // One input row: y itself is the transposed layout, no copy needed.
        y.par_chunks_mut(rows)
            .zip(w.par_chunks(rows * k))
            .for_each(|(y_part, w_part)| {
                for (out, w_row) in y_part.iter_mut().zip(w_part.chunks_exact(k)) {
                    *out = dot(w_row, x);
                }
            });
        return;
    }
    let mut yt = vec![0.0f32; n * m];
    yt.par_chunks_mut(rows * m)
        .zip(w.par_chunks(rows * k))
        .for_each(|(yt_part, w_part)| {
            // Input rows outer, weight rows inner: this task's block of
            // weights (about TASK_BYTES) stays in cache while every input
            // row passes through it.
            for (i, x_row) in x.chunks_exact(k).enumerate() {
                for (r, w_row) in w_part.chunks_exact(k).enumerate() {
                    yt_part[r * m + i] = dot(w_row, x_row);
                }
            }
        });
    for (j, yt_row) in yt.chunks_exact(m).enumerate() {
        for (i, &v) in yt_row.iter().enumerate() {
            y[i * n + j] = v;
        }
    }
}

/// `Y = X · Wᵀ` with `f32` weights, in parallel.
pub fn par_matmul_nt(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    par_matmul_nt_with(x, w, y, m, k, n, dot);
}

/// `Y = X · Wᵀ` with `bf16` weights, in parallel.
pub fn par_matmul_nt_bf16(x: &[f32], w: &[Bf16], y: &mut [f32], m: usize, k: usize, n: usize) {
    par_matmul_nt_with(x, w, y, m, k, n, dot_bf16);
}

/// `Y = X · Wᵀ` on a [`SpinPool`]: the same work split as in
/// [`par_matmul_nt_with`], but one contiguous block of weight rows per
/// thread and no sleeping threads to wake up.
pub fn matmul_nt_pool_with<W, D>(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[W],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    dot: D,
) where
    W: Sync,
    D: Fn(&[W], &[f32]) -> f32 + Sync,
{
    assert_eq!(x.len(), m * k, "x must be m×k");
    assert_eq!(w.len(), n * k, "w must be n×k");
    assert_eq!(y.len(), m * n, "y must be m×n");
    if m == 1 {
        pool.for_each_chunk_mut(y, 1, |start, y_part| {
            let rows = &w[start * k..(start + y_part.len()) * k];
            for (out, w_row) in y_part.iter_mut().zip(rows.chunks_exact(k)) {
                *out = dot(w_row, x);
            }
        });
        return;
    }
    let mut yt = vec![0.0f32; n * m];
    // Chunks of yt are whole rows (multiples of m), one weight row each.
    pool.for_each_chunk_mut(&mut yt, m, |start, yt_part| {
        let first_row = start / m;
        let rows = yt_part.len() / m;
        let w_part = &w[first_row * k..(first_row + rows) * k];
        // Process weight rows in groups of about TASK_BYTES so each group
        // stays in cache while every input row passes through it.
        let group = rows_per_task(k * size_of::<W>());
        for (g, w_group) in w_part.chunks(group * k).enumerate() {
            for (i, x_row) in x.chunks_exact(k).enumerate() {
                for (r, w_row) in w_group.chunks_exact(k).enumerate() {
                    yt_part[(g * group + r) * m + i] = dot(w_row, x_row);
                }
            }
        }
    });
    for (j, yt_row) in yt.chunks_exact(m).enumerate() {
        for (i, &v) in yt_row.iter().enumerate() {
            y[i * n + j] = v;
        }
    }
}

/// `Y = X · Wᵀ` with `f32` weights on a [`SpinPool`].
pub fn matmul_nt_pool(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    matmul_nt_pool_with(pool, x, w, y, m, k, n, dot);
}

/// `Y = X · Wᵀ` with `bf16` weights on a [`SpinPool`].
pub fn matmul_nt_pool_bf16(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[Bf16],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    matmul_nt_pool_with(pool, x, w, y, m, k, n, dot_bf16);
}

/// A counter alone on its own 64-byte cache line.
#[repr(align(64))]
#[derive(Default)]
pub struct PaddedCounter(pub AtomicU64);

/// Each thread increments its own counter `iters` times. The counters sit
/// next to each other in one array, so several of them share a cache line.
pub fn count_adjacent(threads: usize, iters: u64) -> u64 {
    let counters: Vec<AtomicU64> = (0..threads).map(|_| AtomicU64::new(0)).collect();
    std::thread::scope(|s| {
        for c in &counters {
            s.spawn(move || {
                for _ in 0..iters {
                    c.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    counters.iter().map(|c| c.load(Ordering::Relaxed)).sum()
}

/// The same work, but each counter is padded to a full cache line.
pub fn count_padded(threads: usize, iters: u64) -> u64 {
    let counters: Vec<PaddedCounter> = (0..threads).map(|_| PaddedCounter::default()).collect();
    std::thread::scope(|s| {
        for c in &counters {
            s.spawn(move || {
                for _ in 0..iters {
                    c.0.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    counters.iter().map(|c| c.0.load(Ordering::Relaxed)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch06_simd::random_vec;

    fn reference(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut y = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                y[i * n + j] = (0..k)
                    .map(|p| f64::from(x[i * k + p]) * f64::from(w[j * k + p]))
                    .sum();
            }
        }
        y
    }

    fn assert_close(got: &[f32], want: &[f64], k: usize) {
        let tol = 1e-5 * (k as f64).sqrt() * 4.0 + 1e-6;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!((f64::from(g) - w).abs() <= tol, "element {i}: {g} vs {w}");
        }
    }

    #[test]
    fn parallel_matvecs_match_the_reference() {
        for (n, k) in [(1, 1), (7, 13), (100, 300), (1000, 64)] {
            let w = random_vec(n * k, 1);
            let x = random_vec(k, 2);
            let want = reference(&x, &w, 1, k, n);
            let mut y = vec![f32::NAN; n];
            matvec_serial(&w, &x, &mut y);
            assert_close(&y, &want, k);
            for threads in [1, 2, 3, 8] {
                y.fill(f32::NAN);
                matvec_scoped(&w, &x, &mut y, threads);
                assert_close(&y, &want, k);
            }
            y.fill(f32::NAN);
            matvec_rayon(&w, &x, &mut y);
            assert_close(&y, &want, k);
        }
    }

    #[test]
    fn parallel_matmul_matches_the_reference() {
        for (m, k, n) in [(1, 5, 3), (2, 17, 9), (5, 300, 70), (16, 64, 1000)] {
            let x = random_vec(m * k, 3);
            let w = random_vec(n * k, 4);
            let want = reference(&x, &w, m, k, n);
            let mut y = vec![f32::NAN; m * n];
            par_matmul_nt(&x, &w, &mut y, m, k, n);
            assert_close(&y, &want, k);

            let w16: Vec<Bf16> = w.iter().map(|&v| Bf16::from_f32(v)).collect();
            let w16f: Vec<f32> = w16.iter().map(|v| v.to_f32()).collect();
            let want16 = reference(&x, &w16f, m, k, n);
            y.fill(f32::NAN);
            par_matmul_nt_bf16(&x, &w16, &mut y, m, k, n);
            assert_close(&y, &want16, k);
        }
    }

    #[test]
    fn pool_matmul_matches_the_reference() {
        let mut pool = SpinPool::new(3);
        for (m, k, n) in [
            (1, 5, 3),
            (1, 64, 1000),
            (2, 17, 9),
            (5, 300, 70),
            (16, 64, 1000),
        ] {
            let x = random_vec(m * k, 7);
            let w = random_vec(n * k, 8);
            let want = reference(&x, &w, m, k, n);
            let mut y = vec![f32::NAN; m * n];
            matmul_nt_pool(&mut pool, &x, &w, &mut y, m, k, n);
            assert_close(&y, &want, k);
        }
    }

    #[test]
    fn counters_count() {
        assert_eq!(count_adjacent(4, 1000), 4000);
        assert_eq!(count_padded(4, 1000), 4000);
        assert_eq!(std::mem::align_of::<PaddedCounter>(), 64);
    }
}
