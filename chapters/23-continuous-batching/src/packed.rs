//! A `bf16` matrix laid out for small batches.
//!
//! Chapter 17's kernels compute each output as a dot product: a weight row
//! times an activation row, summed across a vector register at the end (a
//! "horizontal" sum). With a batch of `m` rows that is `m` dot products per
//! weight row, each with its own horizontal sum.
//!
//! Here the weights are repacked once, at load time: for every block of 16
//! output rows, column by column, the 16 weights side by side. One vector
//! load then holds column `c` of 16 different outputs, and
//!
//! ```text
//! acc[r] += broadcast(x[r][c]) × weights[c][16 outputs]
//! ```
//!
//! updates 16 outputs of batch row `r` in one instruction. Each weight
//! vector is loaded and converted once and used by every batch row (up to
//! 8 at a time, each with its own register of 16 sums), and no horizontal
//! sums are needed at all. This is how optimized libraries lay out weights
//! for matrix products ("packing").

#![expect(
    clippy::inline_always,
    reason = "multiversioning: the generic row kernel must be inlined into the #[target_feature] function"
)]

use ch02_numbers::Bf16;
use ch07_threads::SpinPool;
use ch14_kv_cache::Matrix;

/// Outputs per block: one AVX-512 register of `f32`.
pub const LANES: usize = 16;
/// Batch rows processed together (one accumulator register each).
pub const MAX_ROWS: usize = 8;

/// A `rows × cols` matrix of `bf16` weights stored as `[rows / 16][cols][16]`.
pub struct PackedBf16 {
    packed: Vec<u16>,
    rows: usize,
    cols: usize,
}

impl PackedBf16 {
    /// Repacks a row-major `rows × cols` matrix. `rows` must be a multiple
    /// of 16 (true of every matrix in Llama-style models).
    pub fn new(values: &[Bf16], rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols, "matrix has the wrong size");
        assert_eq!(rows % LANES, 0, "rows must be a multiple of {LANES}");
        let mut packed = vec![0u16; rows * cols];
        for block in 0..rows / LANES {
            for c in 0..cols {
                for t in 0..LANES {
                    packed[(block * cols + c) * LANES + t] =
                        values[(block * LANES + t) * cols + c].to_bits();
                }
            }
        }
        Self { packed, rows, cols }
    }
}

impl Matrix for PackedBf16 {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn bytes(&self) -> usize {
        self.packed.len() * 2
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        let (block, t) = (r / LANES, r % LANES);
        for (c, o) in out.iter_mut().enumerate() {
            let bits = self.packed[(block * self.cols + c) * LANES + t];
            *o = Bf16::from_bits(bits).to_f32();
        }
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        let (n, k) = (self.rows, self.cols);
        assert_eq!(x.len(), m * k, "x must be m×k");
        assert_eq!(y.len(), m * n, "y must be m×n");
        // Threads split the blocks of 16 outputs. Each block's results go
        // to `scratch` as `[m][16]`, contiguous per block, so every thread
        // writes its own slice.
        scratch.clear();
        scratch.resize(n * m, 0.0);
        let per_block = m * LANES;
        pool.for_each_chunk_mut(scratch, per_block, |first, part| {
            for (i, out) in part.chunks_exact_mut(per_block).enumerate() {
                let block = first / per_block + i;
                let w = &self.packed[block * k * LANES..(block + 1) * k * LANES];
                block_product(w, x, m, k, out);
            }
        });
        for (block, results) in scratch.chunks_exact(per_block).enumerate() {
            for (r, sums) in results.chunks_exact(LANES).enumerate() {
                y[r * n + block * LANES..][..LANES].copy_from_slice(sums);
            }
        }
    }
}

/// One block of 16 outputs for all `m` rows of `x`: `out[r][t]` is the
/// dot product of row `r` of `x` with weight row `16·block + t`.
fn block_product(w: &[u16], x: &[f32], m: usize, k: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512F was just detected; the sizes are those
        // `matmul` checked.
        unsafe { x86::block_product(w, x, m, k, out) };
        return;
    }
    for r in 0..m {
        let mut acc = [0.0f32; LANES];
        for (c, &xv) in x[r * k..(r + 1) * k].iter().enumerate() {
            for (a, &bits) in acc.iter_mut().zip(&w[c * LANES..(c + 1) * LANES]) {
                *a += xv * Bf16::from_bits(bits).to_f32();
            }
        }
        out[r * LANES..(r + 1) * LANES].copy_from_slice(&acc);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{LANES, MAX_ROWS};
    use std::arch::x86_64::{
        __m256i, __m512, _mm256_loadu_si256, _mm512_castsi512_ps, _mm512_cvtepu16_epi32,
        _mm512_fmadd_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_slli_epi32, _mm512_storeu_ps,
    };

    /// # Safety
    ///
    /// AVX-512F must be available; `w` holds `k × 16` values, `x` `m × k`,
    /// `out` `m × 16`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn block_product(w: &[u16], x: &[f32], m: usize, k: usize, out: &mut [f32]) {
        assert!(w.len() == k * LANES && x.len() == m * k && out.len() == m * LANES);
        let mut r0 = 0;
        while r0 < m {
            // SAFETY (all arms): the sizes were checked above, and
            // `r0 + rows <= m` for each call.
            unsafe {
                match (m - r0).min(MAX_ROWS) {
                    1 => rows::<1>(w, x, r0, k, out),
                    2 => rows::<2>(w, x, r0, k, out),
                    3 => rows::<3>(w, x, r0, k, out),
                    4 => rows::<4>(w, x, r0, k, out),
                    5 => rows::<5>(w, x, r0, k, out),
                    6 => rows::<6>(w, x, r0, k, out),
                    7 => rows::<7>(w, x, r0, k, out),
                    _ => rows::<8>(w, x, r0, k, out),
                }
            }
            r0 += MAX_ROWS.min(m - r0);
        }
    }

    /// Rows `r0..r0 + R` of `x` against one block: `R` registers of 16
    /// sums, one weight load and conversion per column for all of them.
    ///
    /// # Safety
    ///
    /// Called only from `block_product`, with its checks, so that it is
    /// compiled with AVX-512F.
    #[inline(always)]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    unsafe fn rows<const R: usize>(w: &[u16], x: &[f32], r0: usize, k: usize, out: &mut [f32]) {
        // SAFETY: see the function's contract.
        unsafe {
            let mut acc: [__m512; R] = [_mm512_setzero_ps(); R];
            let xs: [*const f32; R] = std::array::from_fn(|r| x.as_ptr().add((r0 + r) * k));
            let wp = w.as_ptr();
            for c in 0..k {
                let bits = _mm256_loadu_si256(wp.add(c * LANES).cast::<__m256i>());
                let wv = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(bits)));
                for (a, xr) in acc.iter_mut().zip(&xs) {
                    *a = _mm512_fmadd_ps(_mm512_set1_ps(*xr.add(c)), wv, *a);
                }
            }
            for (r, a) in acc.iter().enumerate() {
                _mm512_storeu_ps(out.as_mut_ptr().add((r0 + r) * LANES), *a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_products_match_plain_dot_products() {
        let (n, k) = (48, 40);
        let w: Vec<Bf16> = (0..n * k)
            .map(|i| Bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 64.0))
            .collect();
        let packed = PackedBf16::new(&w, n, k);
        let mut pool = SpinPool::new(3);
        let mut scratch = Vec::new();
        for m in [1, 3, 8, 11] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0)
                .collect();
            let mut y = vec![0.0; m * n];
            packed.matmul(&mut pool, &x, &mut y, m, &mut scratch);
            for r in 0..m {
                for j in 0..n {
                    let want: f32 = (0..k).map(|c| x[r * k + c] * w[j * k + c].to_f32()).sum();
                    let got = y[r * n + j];
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "m {m}, row {r}, output {j}"
                    );
                }
            }
        }
        let mut row = vec![0.0; k];
        packed.row_to_f32(21, &mut row);
        assert!(
            row.iter()
                .zip(&w[21 * k..22 * k])
                .all(|(a, b)| a.to_bits() == b.to_f32().to_bits())
        );
    }
}
