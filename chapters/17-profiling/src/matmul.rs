//! Chapter 14's parallel matmul, rebuilt around a four-row kernel.

use ch07_threads::{SpinPool, rows_per_task};

/// `y[m × n] = x[m × k] · wᵀ`, where `w` holds `n` rows of `w.len() / n`
/// elements of type `W` (for `bf16`, `k` elements; for the quantized
/// formats of chapters 18-19, `k / block` blocks).
///
/// Threads split the weight rows in multiples of 4. Within its share, a
/// thread walks groups of rows small enough to stay in cache and, for each
/// input row, calls `dot4` on four weight rows at a time (`dot` for the last
/// one to three rows). With `m > 1` the results go through `scratch` in
/// transposed layout, as in chapter 14.
pub fn matmul_rows4<W, D, D4>(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[W],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Vec<f32>,
    dot: D,
    dot4: D4,
) where
    W: Sync,
    D: Fn(&[W], &[f32]) -> f32 + Sync,
    D4: Fn([&[W]; 4], &[f32]) -> [f32; 4] + Sync,
{
    assert_eq!(x.len(), m * k, "x must be m×k");
    assert_eq!(y.len(), m * n, "y must be m×n");
    assert!(
        n > 0 && w.len().is_multiple_of(n),
        "w must have n equal rows"
    );
    let row_len = w.len() / n;
    // Computes outputs for weight rows `first..first + count` and input row
    // `x_row`, calling `put(row index within the share, value)`.
    let rows_times =
        |first: usize, count: usize, x_row: &[f32], put: &mut dyn FnMut(usize, f32)| {
            let w_part = &w[first * row_len..(first + count) * row_len];
            let mut quads = w_part.chunks_exact(4 * row_len);
            for (q, quad) in quads.by_ref().enumerate() {
                let (a, rest) = quad.split_at(row_len);
                let (b, rest) = rest.split_at(row_len);
                let (c, d) = rest.split_at(row_len);
                let out = dot4([a, b, c, d], x_row);
                for (j, v) in out.into_iter().enumerate() {
                    put(4 * q + j, v);
                }
            }
            let done = count - count % 4;
            for (j, row) in quads.remainder().chunks_exact(row_len).enumerate() {
                put(done + j, dot(row, x_row));
            }
        };

    if m == 1 {
        pool.for_each_chunk_mut(y, 4, |start, y_part| {
            rows_times(start, y_part.len(), x, &mut |j, v| y_part[j] = v);
        });
        return;
    }
    scratch.clear();
    scratch.resize(n * m, 0.0);
    // Groups of rows that stay in cache while every input row passes by,
    // rounded down to a multiple of 4 (at least 4).
    let group = (rows_per_task(row_len * size_of::<W>()) / 4).max(1) * 4;
    pool.for_each_chunk_mut(scratch, 4 * m, |start, yt_part| {
        let first = start / m;
        let rows = yt_part.len() / m;
        let mut g = 0;
        while g < rows {
            let count = group.min(rows - g);
            for (i, x_row) in x.chunks_exact(k).enumerate() {
                rows_times(first + g, count, x_row, &mut |j, v| {
                    yt_part[(g + j) * m + i] = v;
                });
            }
            g += count;
        }
    });
    for (j, yt_row) in scratch.chunks_exact(m).enumerate() {
        for (i, &v) in yt_row.iter().enumerate() {
            y[i * n + j] = v;
        }
    }
}

/// `y[m × n] = x[m × k] · wᵀ` for prefill (`m > 1`), with a 4 × 4 tile
/// kernel: four weight rows against four input rows per call.
///
/// Threads split the weight rows in multiples of 4. Each walks its rows in
/// cache-sized groups; for each group, every block of four input rows goes
/// through `tile` once per four weight rows. Leftover rows (when `m` or a
/// share is not a multiple of 4) use `dot`. For `m == 1` there is nothing
/// to tile and this is chapter 14's matrix-vector product.
pub fn matmul_tiled<W, D, T>(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[W],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Vec<f32>,
    dot: D,
    tile: T,
) where
    W: Sync,
    D: Fn(&[W], &[f32]) -> f32 + Sync,
    T: Fn([&[W]; 4], [&[f32]; 4]) -> [[f32; 4]; 4] + Sync,
{
    if m == 1 {
        ch14_kv_cache::matmul_pooled(pool, x, w, y, m, k, n, scratch, dot);
        return;
    }
    assert_eq!(x.len(), m * k, "x must be m×k");
    assert_eq!(w.len(), n * k, "w must be n×k");
    assert_eq!(y.len(), m * n, "y must be m×n");
    scratch.clear();
    scratch.resize(n * m, 0.0);
    let x_row = |i: usize| &x[i * k..(i + 1) * k];
    let w_row = |j: usize| &w[j * k..(j + 1) * k];
    let group = (rows_per_task(k * size_of::<W>()) / 4).max(1) * 4;
    let full = m - m % 4;
    pool.for_each_chunk_mut(scratch, 4 * m, |start, yt_part| {
        let first = start / m;
        let rows = yt_part.len() / m;
        // yt_part holds rows first..first + rows of the transposed output.
        let mut put = |j: usize, i: usize, v: f32| yt_part[(j - first) * m + i] = v;
        for g in (first..first + rows).step_by(group) {
            let end = (g + group).min(first + rows);
            let quads = g + (end - g) / 4 * 4;
            for i0 in (0..full).step_by(4) {
                let xs = [x_row(i0), x_row(i0 + 1), x_row(i0 + 2), x_row(i0 + 3)];
                for j0 in (g..quads).step_by(4) {
                    let ws = [w_row(j0), w_row(j0 + 1), w_row(j0 + 2), w_row(j0 + 3)];
                    let out = tile(ws, xs);
                    for (r, row) in out.iter().enumerate() {
                        for (c, &v) in row.iter().enumerate() {
                            put(j0 + r, i0 + c, v);
                        }
                    }
                }
                for j in quads..end {
                    for (c, &xi) in xs.iter().enumerate() {
                        put(j, i0 + c, dot(w_row(j), xi));
                    }
                }
            }
            for i in full..m {
                for j in g..end {
                    put(j, i, dot(w_row(j), x_row(i)));
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

/// Decode-time fusion: `outputs[i] = x · parts[i]ᵀ` for several matrices
/// that share `x`, as **one** parallel pass over all their rows.
///
/// Each `parts[i]` holds whole rows of `row_len` elements of `W`; `x` is
/// the activation row in whatever form `dot` takes (`f32` values, or
/// quantized blocks in chapter 18). Threads split the combined rows evenly, writing into
/// `scratch`, and the results are then copied to their outputs. One pass
/// means one synchronisation and one set of per-thread streams instead of
/// one per matrix.
pub fn matvec_many<W, X, D>(
    pool: &mut SpinPool,
    parts: &[&[W]],
    row_len: usize,
    x: &[X],
    outputs: &mut [&mut [f32]],
    scratch: &mut Vec<f32>,
    dot: D,
) where
    W: Sync,
    X: Sync,
    D: Fn(&[W], &[X]) -> f32 + Sync,
{
    assert_eq!(parts.len(), outputs.len(), "one output per matrix");
    let rows: Vec<usize> = parts.iter().map(|p| p.len() / row_len).collect();
    for (r, out) in rows.iter().zip(outputs.iter()) {
        assert_eq!(*r, out.len(), "output length must equal the matrix's rows");
    }
    scratch.clear();
    scratch.resize(rows.iter().sum(), 0.0);
    pool.for_each_chunk_mut(scratch, 1, |start, out| {
        // Find the matrix and row where this thread's share begins, then walk.
        let (mut part, mut row) = (0, start);
        while row >= rows[part] {
            row -= rows[part];
            part += 1;
        }
        for o in out {
            *o = dot(&parts[part][row * row_len..(row + 1) * row_len], x);
            row += 1;
            if row == rows[part] && part + 1 < rows.len() {
                (part, row) = (part + 1, 0);
            }
        }
    });
    let mut at = 0;
    for out in outputs.iter_mut() {
        out.copy_from_slice(&scratch[at..at + out.len()]);
        at += out.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch02_numbers::Bf16;
    use ch06_simd::{dot_bf16, random_vec};

    #[test]
    fn matches_one_dot_product_per_output() {
        let mut pool = SpinPool::new(3);
        let mut scratch = Vec::new();
        for (m, k, n) in [
            (1, 64, 4),
            (1, 48, 7),
            (1, 576, 193),
            (3, 40, 9),
            (5, 64, 130),
        ] {
            let w: Vec<Bf16> = random_vec(n * k, 1)
                .into_iter()
                .map(Bf16::from_f32)
                .collect();
            let x = random_vec(m * k, 2);
            let mut y = vec![0.0; m * n];
            matmul_rows4(
                &mut pool,
                &x,
                &w,
                &mut y,
                m,
                k,
                n,
                &mut scratch,
                dot_bf16,
                crate::dot4_bf16,
            );
            for i in 0..m {
                for j in 0..n {
                    let want = dot_bf16(&w[j * k..(j + 1) * k], &x[i * k..(i + 1) * k]);
                    let got = y[i * n + j];
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "m={m} n={n} ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn the_tiled_matmul_matches_one_dot_product_per_output() {
        let mut pool = SpinPool::new(3);
        let mut scratch = Vec::new();
        for (m, k, n) in [
            (1, 64, 5),
            (4, 64, 8),
            (5, 40, 9),
            (7, 33, 130),
            (40, 576, 196),
        ] {
            let w: Vec<Bf16> = random_vec(n * k, 1)
                .into_iter()
                .map(Bf16::from_f32)
                .collect();
            let x = random_vec(m * k, 2);
            let mut y = vec![0.0; m * n];
            matmul_tiled(
                &mut pool,
                &x,
                &w,
                &mut y,
                m,
                k,
                n,
                &mut scratch,
                dot_bf16,
                crate::tile_bf16,
            );
            for i in 0..m {
                for j in 0..n {
                    let want = dot_bf16(&w[j * k..(j + 1) * k], &x[i * k..(i + 1) * k]);
                    let got = y[i * n + j];
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "m={m} n={n} ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn a_fused_pass_equals_separate_products_bit_for_bit() {
        let mut pool = SpinPool::new(3);
        let mut scratch = Vec::new();
        let k = 40;
        let sizes = [7, 1, 13];
        let mats: Vec<Vec<Bf16>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                random_vec(n * k, i as u64)
                    .into_iter()
                    .map(Bf16::from_f32)
                    .collect()
            })
            .collect();
        let x = random_vec(k, 9);
        let mut outs: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0.0; n]).collect();
        {
            let parts: Vec<&[Bf16]> = mats.iter().map(Vec::as_slice).collect();
            let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(Vec::as_mut_slice).collect();
            matvec_many(&mut pool, &parts, k, &x, &mut refs, &mut scratch, dot_bf16);
        }
        for (mat, out) in mats.iter().zip(&outs) {
            let want: Vec<f32> = mat.chunks_exact(k).map(|row| dot_bf16(row, &x)).collect();
            assert_eq!(*out, want);
        }
    }
}
