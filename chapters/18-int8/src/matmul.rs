//! Chapter 14's parallel matrix product, for weights stored in blocks.

use ch07_threads::{SpinPool, rows_per_task};

/// `y[m × n] = x · wᵀ` where `w` holds `n` rows of blocks and `x` holds `m`
/// rows of activations of any type `X` (`f32` values, or quantized blocks).
/// `dot(weight_row, x_row)` computes one output. Row lengths are derived
/// from the slice lengths, so the same function serves every block format.
pub fn matmul_blocks<B, X, D>(
    pool: &mut SpinPool,
    x: &[X],
    w: &[B],
    y: &mut [f32],
    m: usize,
    n: usize,
    scratch: &mut Vec<f32>,
    dot: D,
) where
    B: Sync,
    X: Sync,
    D: Fn(&[B], &[X]) -> f32 + Sync,
{
    assert!(m > 0 && n > 0, "empty product");
    assert!(x.len().is_multiple_of(m), "x must have m equal rows");
    assert!(w.len().is_multiple_of(n), "w must have n equal rows");
    assert_eq!(y.len(), m * n, "y must be m×n");
    let (x_len, w_len) = (x.len() / m, w.len() / n);
    if m == 1 {
        pool.for_each_chunk_mut(y, 1, |start, y_part| {
            let rows = &w[start * w_len..(start + y_part.len()) * w_len];
            for (out, row) in y_part.iter_mut().zip(rows.chunks_exact(w_len)) {
                *out = dot(row, x);
            }
        });
        return;
    }
    scratch.clear();
    scratch.resize(n * m, 0.0);
    let group = rows_per_task(w_len * size_of::<B>());
    pool.for_each_chunk_mut(scratch, m, |start, yt_part| {
        let first = start / m;
        let rows = yt_part.len() / m;
        let w_part = &w[first * w_len..(first + rows) * w_len];
        for (g, w_group) in w_part.chunks(group * w_len).enumerate() {
            for (i, x_row) in x.chunks_exact(x_len).enumerate() {
                for (r, w_row) in w_group.chunks_exact(w_len).enumerate() {
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
