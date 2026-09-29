//! Two kernels that do more than one dot product per call.
//!
//! - [`dot4_bf16`]: four weight rows against one activation vector. Each
//!   activation is loaded once and used four times, and four rows stream
//!   from memory at once. Aimed at decode; the lesson measures why it does
//!   not help SmolLM2.
//! - [`tile_bf16`]: four weight rows against four activation vectors (four
//!   tokens), sixteen results. Each loaded value is used four times, so the
//!   kernel does four multiply-adds per load instead of one. Aimed at
//!   prefill, where the arithmetic is the bottleneck.

use ch02_numbers::Bf16;
use ch06_simd::{Isa, best_isa, dot_bf16_portable};

/// Four `bf16 · f32` dot products: `[w0·x, w1·x, w2·x, w3·x]`.
pub fn dot4_bf16(rows: [&[Bf16]; 4], x: &[f32]) -> [f32; 4] {
    dot4_bf16_with(best_isa(), rows, x).expect("best_isa only returns available instruction sets")
}

/// [`dot4_bf16`] with a specific instruction set, or `None` if unavailable.
pub fn dot4_bf16_with(isa: Isa, rows: [&[Bf16]; 4], x: &[f32]) -> Option<[f32; 4]> {
    for row in rows {
        assert_eq!(row.len(), x.len(), "rows and x must have the same length");
    }
    if !isa.is_available() {
        return None;
    }
    Some(match isa {
        // SAFETY (both arms): the instruction set was just detected.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::dot4_bf16_avx512(rows, x) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { x86::dot4_bf16_avx2(rows, x) },
        _ => dot4_bf16_portable(rows, x),
    })
}

/// Portable version: four separate dot products.
pub fn dot4_bf16_portable(rows: [&[Bf16]; 4], x: &[f32]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (o, row) in out.iter_mut().zip(rows) {
        *o = dot_bf16_portable(row, x);
    }
    out
}

/// A 4 × 4 tile of dot products: `out[r][i] = rows[r] · xs[i]`.
pub fn tile_bf16(rows: [&[Bf16]; 4], xs: [&[f32]; 4]) -> [[f32; 4]; 4] {
    tile_bf16_with(best_isa(), rows, xs).expect("best_isa only returns available instruction sets")
}

/// [`tile_bf16`] with a specific instruction set, or `None` if unavailable.
pub fn tile_bf16_with(isa: Isa, rows: [&[Bf16]; 4], xs: [&[f32]; 4]) -> Option<[[f32; 4]; 4]> {
    let k = xs[0].len();
    for row in rows {
        assert_eq!(row.len(), k, "every row must have k values");
    }
    for x in xs {
        assert_eq!(x.len(), k, "every activation vector must have k values");
    }
    if !isa.is_available() {
        return None;
    }
    Some(match isa {
        // SAFETY (both arms): the instruction set was just detected.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::tile_bf16_avx512(rows, xs) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { x86::tile_bf16_avx2(rows, xs) },
        _ => tile_bf16_portable(rows, xs),
    })
}

/// Portable version: sixteen separate dot products.
pub fn tile_bf16_portable(rows: [&[Bf16]; 4], xs: [&[f32]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (o, row) in out.iter_mut().zip(rows) {
        for (v, x) in o.iter_mut().zip(xs) {
            *v = dot_bf16_portable(row, x);
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    //! AVX2 and AVX-512 versions.

    use ch02_numbers::Bf16;
    use std::arch::x86_64::{
        __m128i, __m256, __m256i, __m512, _mm_loadu_si128, _mm256_add_ps, _mm256_castsi256_ps,
        _mm256_cvtepu16_epi32, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
        _mm256_setzero_ps, _mm256_slli_epi32, _mm256_storeu_ps, _mm512_add_ps, _mm512_castsi512_ps,
        _mm512_cvtepu16_epi32, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps,
        _mm512_setzero_ps, _mm512_slli_epi32,
    };

    /// 16 `bf16` values at `row[col..]`, widened to `f32` (chapter 6).
    ///
    /// # Safety
    ///
    /// AVX-512F must be available and `col + 16 <= row.len()`.
    #[target_feature(enable = "avx512f")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    unsafe fn load16(row: &[Bf16], col: usize) -> __m512 {
        // SAFETY: the caller guarantees 16 values from `col` are in bounds.
        let bits = unsafe { _mm256_loadu_si256(row.as_ptr().add(col).cast::<__m256i>()) };
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(bits)))
    }

    /// 8 `bf16` values at `row[col..]`, widened to `f32`.
    ///
    /// # Safety
    ///
    /// AVX2 must be available and `col + 8 <= row.len()`.
    #[target_feature(enable = "avx2,fma")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    unsafe fn load8(row: &[Bf16], col: usize) -> __m256 {
        // SAFETY: the caller guarantees 8 values from `col` are in bounds.
        let bits = unsafe { _mm_loadu_si128(row.as_ptr().add(col).cast::<__m128i>()) };
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(bits)))
    }

    #[target_feature(enable = "avx2,fma")]
    fn sum8(v: __m256) -> f32 {
        let mut lanes = [0.0f32; 8];
        // SAFETY: `lanes` has room for 8 floats.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
        lanes.iter().sum()
    }

    /// AVX-512 four-row kernel: 32 columns per step, two accumulators per
    /// row (eight independent chains of multiply-adds).
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F. (Lengths are checked.)
    #[target_feature(enable = "avx512f")]
    #[expect(
        clippy::needless_range_loop,
        reason = "`half` picks both an accumulator and a column offset"
    )]
    pub fn dot4_bf16_avx512(rows: [&[Bf16]; 4], x: &[f32]) -> [f32; 4] {
        let k = x.len();
        assert!(
            rows.iter().all(|r| r.len() == k),
            "rows and x differ in length"
        );
        let steps = k / 32;
        let mut acc = [[_mm512_setzero_ps(); 2]; 4];
        for s in 0..steps {
            for half in 0..2 {
                let col = s * 32 + half * 16;
                // SAFETY: col + 16 <= steps * 32 <= k, and every row and `x`
                // have k elements.
                let vx = unsafe { _mm512_loadu_ps(x.as_ptr().add(col)) };
                for (r, row) in rows.iter().enumerate() {
                    // SAFETY: the same bounds, for this row.
                    let w = unsafe { load16(row, col) };
                    acc[r][half] = _mm512_fmadd_ps(w, vx, acc[r][half]);
                }
            }
        }
        let mut out = [0.0f32; 4];
        for (r, o) in out.iter_mut().enumerate() {
            *o = _mm512_reduce_add_ps(_mm512_add_ps(acc[r][0], acc[r][1]));
            for col in steps * 32..k {
                *o += rows[r][col].to_f32() * x[col];
            }
        }
        out
    }

    /// AVX2 four-row kernel: 16 columns per step, two accumulators per row.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA. (Lengths are checked.)
    #[target_feature(enable = "avx2,fma")]
    #[expect(
        clippy::needless_range_loop,
        reason = "`half` picks both an accumulator and a column offset"
    )]
    pub fn dot4_bf16_avx2(rows: [&[Bf16]; 4], x: &[f32]) -> [f32; 4] {
        let k = x.len();
        assert!(
            rows.iter().all(|r| r.len() == k),
            "rows and x differ in length"
        );
        let steps = k / 16;
        let mut acc = [[_mm256_setzero_ps(); 2]; 4];
        for s in 0..steps {
            for half in 0..2 {
                let col = s * 16 + half * 8;
                // SAFETY: col + 8 <= k for every row and for `x`.
                let vx = unsafe { _mm256_loadu_ps(x.as_ptr().add(col)) };
                for (r, row) in rows.iter().enumerate() {
                    // SAFETY: the same bounds, for this row.
                    let w = unsafe { load8(row, col) };
                    acc[r][half] = _mm256_fmadd_ps(w, vx, acc[r][half]);
                }
            }
        }
        let mut out = [0.0f32; 4];
        for (r, o) in out.iter_mut().enumerate() {
            *o = sum8(_mm256_add_ps(acc[r][0], acc[r][1]));
            for col in steps * 16..k {
                *o += rows[r][col].to_f32() * x[col];
            }
        }
        out
    }

    /// AVX-512 4 × 4 tile: per 16 columns, 4 activation loads and 4 weight
    /// loads feed 16 multiply-adds into 16 accumulators (16 of the 32
    /// vector registers).
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F. (Lengths are checked.)
    #[target_feature(enable = "avx512f")]
    pub fn tile_bf16_avx512(rows: [&[Bf16]; 4], xs: [&[f32]; 4]) -> [[f32; 4]; 4] {
        let k = xs[0].len();
        assert!(
            rows.iter().all(|r| r.len() == k) && xs.iter().all(|x| x.len() == k),
            "lengths differ"
        );
        let steps = k / 16;
        let mut acc = [[_mm512_setzero_ps(); 4]; 4];
        for s in 0..steps {
            let col = s * 16;
            let mut vx = [_mm512_setzero_ps(); 4];
            for i in 0..4 {
                // SAFETY: col + 16 <= k, the length of every vector.
                vx[i] = unsafe { _mm512_loadu_ps(xs[i].as_ptr().add(col)) };
            }
            for r in 0..4 {
                // SAFETY: col + 16 <= k, the length of every row.
                let w = unsafe { load16(rows[r], col) };
                for i in 0..4 {
                    acc[r][i] = _mm512_fmadd_ps(w, vx[i], acc[r][i]);
                }
            }
        }
        let mut out = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for i in 0..4 {
                out[r][i] = _mm512_reduce_add_ps(acc[r][i]);
                for col in steps * 16..k {
                    out[r][i] += rows[r][col].to_f32() * xs[i][col];
                }
            }
        }
        out
    }

    /// AVX2 4 × 4 tile, computed as two 4 × 2 halves: sixteen 8-wide
    /// accumulators plus the loaded values would need more than AVX2's 16
    /// vector registers, and spilling them to memory would cost more than
    /// widening each weight twice.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA. (Lengths are checked.)
    #[target_feature(enable = "avx2,fma")]
    pub fn tile_bf16_avx2(rows: [&[Bf16]; 4], xs: [&[f32]; 4]) -> [[f32; 4]; 4] {
        let k = xs[0].len();
        assert!(
            rows.iter().all(|r| r.len() == k) && xs.iter().all(|x| x.len() == k),
            "lengths differ"
        );
        let steps = k / 8;
        let mut out = [[0.0f32; 4]; 4];
        for pair in 0..2 {
            let (i0, i1) = (2 * pair, 2 * pair + 1);
            let mut acc = [[_mm256_setzero_ps(); 2]; 4];
            for s in 0..steps {
                let col = s * 8;
                // SAFETY: col + 8 <= k, the length of every vector and row.
                let (x0, x1) = unsafe {
                    (
                        _mm256_loadu_ps(xs[i0].as_ptr().add(col)),
                        _mm256_loadu_ps(xs[i1].as_ptr().add(col)),
                    )
                };
                for r in 0..4 {
                    // SAFETY: as above.
                    let w = unsafe { load8(rows[r], col) };
                    acc[r][0] = _mm256_fmadd_ps(w, x0, acc[r][0]);
                    acc[r][1] = _mm256_fmadd_ps(w, x1, acc[r][1]);
                }
            }
            for r in 0..4 {
                for (j, i) in [i0, i1].into_iter().enumerate() {
                    out[r][i] = sum8(acc[r][j]);
                    for col in steps * 8..k {
                        out[r][i] += rows[r][col].to_f32() * xs[i][col];
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch06_simd::{dot_bf16, random_vec};

    fn bf16_vec(len: usize, seed: u64) -> Vec<Bf16> {
        random_vec(len, seed)
            .into_iter()
            .map(Bf16::from_f32)
            .collect()
    }

    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-4 * (1.0 + want.abs())
    }

    #[test]
    fn four_row_kernels_match_single_dot_products() {
        for k in [1, 7, 16, 31, 32, 33, 576, 1000, 1536] {
            let rows: Vec<Vec<Bf16>> = (0..4).map(|r| bf16_vec(k, 10 + r)).collect();
            let x = random_vec(k, 99);
            let refs = [&rows[0][..], &rows[1][..], &rows[2][..], &rows[3][..]];
            for isa in Isa::ALL {
                let Some(got) = dot4_bf16_with(isa, refs, &x) else {
                    continue;
                };
                for (g, row) in got.iter().zip(&rows) {
                    assert!(close(*g, dot_bf16(row, &x)), "{isa:?}, k = {k}");
                }
            }
        }
    }

    #[test]
    fn tile_kernels_match_single_dot_products() {
        for k in [1, 7, 8, 15, 16, 17, 576, 1000, 1536] {
            let rows: Vec<Vec<Bf16>> = (0..4).map(|r| bf16_vec(k, 20 + r)).collect();
            let xs: Vec<Vec<f32>> = (0..4).map(|i| random_vec(k, 30 + i)).collect();
            let row_refs = [&rows[0][..], &rows[1][..], &rows[2][..], &rows[3][..]];
            let x_refs = [&xs[0][..], &xs[1][..], &xs[2][..], &xs[3][..]];
            for isa in Isa::ALL {
                let Some(got) = tile_bf16_with(isa, row_refs, x_refs) else {
                    continue;
                };
                for (r, row) in rows.iter().enumerate() {
                    for (i, x) in xs.iter().enumerate() {
                        assert!(
                            close(got[r][i], dot_bf16(row, x)),
                            "{isa:?}, k = {k}, ({r},{i})"
                        );
                    }
                }
            }
        }
    }
}
