//! The kernels of an LLM's forward pass, written for the simulator in the
//! shape they have on a GPU: the same indices, loops, barriers and warp
//! operations as the cuda-oxide versions in `oxide/main.rs`.

use crate::sim::{Global, Stats, WARP, launch};

/// Threads per block for the one-dimensional kernels.
pub const BLOCK: usize = 256;
/// Tile edge for the matrix product.
pub const TILE: usize = 16;

/// `y = W·x` for `W` of `rows × cols`, one thread per output row: thread
/// `r` walks row `r`. Correct, and slow on a GPU: at each step the 32
/// lanes of a warp read 32 different rows, `cols` floats apart.
pub fn gemv_thread_per_row(
    w: &Global<'_>,
    x: &Global<'_>,
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Stats {
    launch(rows.div_ceil(BLOCK), BLOCK, 0, |b| {
        b.each(|t| {
            let row = t.global();
            if row < rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += t.load(w, row * cols + c) * t.load(x, c);
                }
                y[row] = sum;
            }
        });
    })
}

/// `y = W·x`, one warp per output row: lane `l` adds up columns `l`,
/// `l + 32`, `l + 64`..., so at each step the warp reads 32 consecutive
/// floats of the row; then the 32 partial sums are added with warp
/// shuffles, and lane 0 writes the result.
pub fn gemv_warp_per_row(
    w: &Global<'_>,
    x: &Global<'_>,
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Stats {
    let warps = BLOCK / WARP;
    launch(rows.div_ceil(warps), BLOCK, 0, |b| {
        let mut partial = vec![0.0; BLOCK];
        b.each(|t| {
            let row = t.block * warps + t.warp();
            if row < rows {
                let mut sum = 0.0;
                let mut c = t.lane();
                while c < cols {
                    sum += t.load(w, row * cols + c) * t.load(x, c);
                    c += WARP;
                }
                partial[t.index] = sum;
            }
        });
        b.warp_sum(&mut partial);
        b.each(|t| {
            let row = t.block * warps + t.warp();
            if row < rows && t.lane() == 0 {
                y[row] = partial[t.index];
            }
        });
    })
}

/// RMSNorm of each row of `x` (`rows × n`): one block per row. Each thread
/// adds the squares of a strided slice of the row; the block adds the 256
/// partial sums in shared memory, halving the active threads at each step,
/// with a barrier between steps; then every thread scales its slice.
pub fn rms_norm(
    x: &Global<'_>,
    weight: &Global<'_>,
    y: &mut [f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Stats {
    launch(rows, BLOCK, BLOCK, |b| {
        b.each(|t| {
            let row = t.block;
            let mut sum = 0.0;
            let mut c = t.index;
            while c < n {
                let v = t.load(x, row * n + c);
                sum += v * v;
                c += BLOCK;
            }
            t.shared_write(t.index, sum);
        });
        b.sync_threads();
        let mut stride = BLOCK / 2;
        while stride > 0 {
            b.each(|t| {
                if t.index < stride {
                    let v = t.shared_read(t.index) + t.shared_read(t.index + stride);
                    t.shared_write(t.index, v);
                }
            });
            b.sync_threads();
            stride /= 2;
        }
        b.each(|t| {
            let row = t.block;
            let scale = 1.0 / (t.shared_read(0) / n as f32 + eps).sqrt();
            let mut c = t.index;
            while c < n {
                y[row * n + c] = t.load(x, row * n + c) * scale * t.load(weight, c);
                c += BLOCK;
            }
        });
    })
}

/// `C = A·Bᵀ` (`A` of `m × k`, `B` of `n × k`, the course's `y = x·Wᵀ`),
/// one thread per output, each reading a whole row of `A` and of `B` from
/// global memory: `2k` loads per output.
pub fn matmul_naive(
    a: &Global<'_>,
    bt: &Global<'_>,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Stats {
    let (gx, gy) = (n.div_ceil(TILE), m.div_ceil(TILE));
    launch(gx * gy, TILE * TILE, 0, |b| {
        let (bx, by) = (b.index % gx, b.index / gx);
        b.each(|t| {
            let (tx, ty) = (t.index % TILE, t.index / TILE);
            let (row, col) = (by * TILE + ty, bx * TILE + tx);
            if row < m && col < n {
                let mut sum = 0.0;
                for i in 0..k {
                    sum += t.load(a, row * k + i) * t.load(bt, col * k + i);
                }
                c[row * n + col] = sum;
            }
        });
    })
}

/// `C = A·Bᵀ` with 16 × 16 tiles in shared memory: a block computes a
/// 16 × 16 tile of `C`; for each 16-wide slice of `k`, its 256 threads
/// first copy one tile of `A` and one of `B` into shared memory (one value
/// each), wait at a barrier, and then each multiplies a row of one tile by
/// a column of the other. Every value loaded from global memory is used 16
/// times. `transposed` stores `B`'s tile as `[k][column]`, so that in the
/// inner loop the 16 lanes reading `B` hit 16 different banks.
pub fn matmul_tiled(
    a: &Global<'_>,
    bt: &Global<'_>,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    transposed: bool,
) -> Stats {
    let (gx, gy) = (n.div_ceil(TILE), m.div_ceil(TILE));
    let tile_b = TILE * TILE; // B's tile starts after A's
    launch(gx * gy, TILE * TILE, 2 * TILE * TILE, |b| {
        let (bx, by) = (b.index % gx, b.index / gx);
        let mut acc = vec![0.0f32; TILE * TILE];
        for k0 in (0..k).step_by(TILE) {
            b.each(|t| {
                let (tx, ty) = (t.index % TILE, t.index / TILE);
                // A's tile: rows of C, k along x.
                let (row, i) = (by * TILE + ty, k0 + tx);
                let va = if row < m && i < k {
                    t.load(a, row * k + i)
                } else {
                    0.0
                };
                t.shared_write(ty * TILE + tx, va);
                // B's tile: columns of C (rows of B), k along x.
                let (col, i) = (bx * TILE + ty, k0 + tx);
                let vb = if col < n && i < k {
                    t.load(bt, col * k + i)
                } else {
                    0.0
                };
                let at = if transposed {
                    tx * TILE + ty
                } else {
                    ty * TILE + tx
                };
                t.shared_write(tile_b + at, vb);
            });
            b.sync_threads();
            b.each(|t| {
                let (tx, ty) = (t.index % TILE, t.index / TILE);
                let mut sum = 0.0;
                for i in 0..TILE {
                    let bv = if transposed {
                        t.shared_read(tile_b + i * TILE + tx)
                    } else {
                        t.shared_read(tile_b + tx * TILE + i)
                    };
                    sum += t.shared_read(ty * TILE + i) * bv;
                }
                acc[t.index] += sum;
            });
            b.sync_threads();
        }
        b.each(|t| {
            let (tx, ty) = (t.index % TILE, t.index / TILE);
            let (row, col) = (by * TILE + ty, bx * TILE + tx);
            if row < m && col < n {
                c[row * n + col] = acc[t.index];
            }
        });
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32 / 500.0) - 1.0)
            .collect()
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
    }

    fn reference_gemv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
            .collect()
    }

    #[test]
    fn both_gemv_kernels_compute_w_times_x_and_differ_in_traffic() {
        // Sizes that are not multiples of 32 or 256, to test the edges.
        let (rows, cols) = (300, 200);
        let (w, x) = (values(rows * cols, 1), values(cols, 2));
        let (gw, gx) = (Global::new(&w, 0), Global::new(&x, 1 << 30));
        let want = reference_gemv(&w, &x, rows, cols);
        let mut y = vec![0.0; rows];
        let slow = gemv_thread_per_row(&gw, &gx, &mut y, rows, cols);
        assert!(close(&y, &want));
        let mut y = vec![0.0; rows];
        let fast = gemv_warp_per_row(&gw, &gx, &mut y, rows, cols);
        assert!(close(&y, &want));
        assert_eq!(slow.loads, fast.loads, "the same work");
        assert!(fast.load_efficiency() > 0.9, "{}", fast.load_efficiency());
        assert!(slow.load_efficiency() < 0.6, "{}", slow.load_efficiency());
        assert_eq!(slow.races + fast.races, 0);
    }

    #[test]
    fn rms_norm_matches_the_cpu_and_has_no_races() {
        let (rows, n) = (3, 576);
        let (x, w) = (values(rows * n, 3), values(n, 4));
        let mut y = vec![0.0; rows * n];
        let s = rms_norm(
            &Global::new(&x, 0),
            &Global::new(&w, 1 << 30),
            &mut y,
            rows,
            n,
            1e-5,
        );
        for r in 0..rows {
            let want = ch08_operators::rms_norm_alloc(&x[r * n..(r + 1) * n], &w, 1e-5);
            assert!(close(&y[r * n..(r + 1) * n], &want));
        }
        assert_eq!(s.races, 0);
        assert_eq!(s.barriers, rows * 9, "one, then one per halving");
    }

    #[test]
    fn tiled_matmul_matches_naive_with_16_times_fewer_loads() {
        let (m, n, k) = (40, 50, 72);
        let (a, bt) = (values(m * k, 5), values(n * k, 6));
        let (ga, gb) = (Global::new(&a, 0), Global::new(&bt, 1 << 30));
        let want: Vec<f32> = (0..m * n)
            .map(|i| (0..k).map(|j| a[i / n * k + j] * bt[i % n * k + j]).sum())
            .collect();
        let mut c = vec![0.0; m * n];
        let naive = matmul_naive(&ga, &gb, &mut c, m, n, k);
        assert!(close(&c, &want));
        for transposed in [false, true] {
            let mut c = vec![0.0; m * n];
            let tiled = matmul_tiled(&ga, &gb, &mut c, m, n, k, transposed);
            assert!(close(&c, &want), "transposed {transposed}");
            assert_eq!(tiled.races, 0);
            assert!(
                tiled.loads * 10 < naive.loads,
                "{} vs {}",
                tiled.loads,
                naive.loads
            );
        }
    }
}
