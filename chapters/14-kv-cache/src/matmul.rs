//! The engine's matrix multiplication: chapter 7's parallel NT product,
//! with the temporary buffer supplied by the caller so that a forward pass
//! never allocates it again.

use ch07_threads::{SpinPool, rows_per_task};

/// `y[m × n] = x[m × k] · wᵀ` for weights `w` stored as `n` rows of `k`
/// elements of type `W`, split across the pool's threads by weight rows.
///
/// `dot(row, x_row)` computes one output; it decides how `W` is read (plain
/// `f32`, `bf16`, quantized blocks...). For `m > 1`, results go through
/// `scratch` in transposed layout (each thread writes whole rows of it) and
/// are transposed into `y` at the end; `scratch` keeps its capacity between
/// calls, so after the first call at a given size nothing is allocated.
pub fn matmul_pooled<W, D>(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[W],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Vec<f32>,
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
            for (out, row) in y_part.iter_mut().zip(rows.chunks_exact(k)) {
                *out = dot(row, x);
            }
        });
        return;
    }
    scratch.clear();
    scratch.resize(n * m, 0.0);
    pool.for_each_chunk_mut(scratch, m, |start, yt_part| {
        let first = start / m;
        let rows = yt_part.len() / m;
        let w_part = &w[first * k..(first + rows) * k];
        // Groups of weight rows small enough to stay in cache while every
        // input row passes through them.
        let group = rows_per_task(k * size_of::<W>());
        for (g, w_group) in w_part.chunks(group * k).enumerate() {
            for (i, x_row) in x.chunks_exact(k).enumerate() {
                for (r, w_row) in w_group.chunks_exact(k).enumerate() {
                    yt_part[(g * group + r) * m + i] = dot(w_row, x_row);
                }
            }
        }
    });
    for (j, yt_row) in scratch.chunks_exact(m).enumerate() {
        for (i, &v) in yt_row.iter().enumerate() {
            y[i * n + j] = v;
        }
    }
}
