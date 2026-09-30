//! Dot products over rows of int8 blocks.
//!
//! - [`dot_q8_f32`]: int8 weights, `f32` activations ("W8A32"). Each weight
//!   is widened to `f32` in registers; only memory traffic shrinks.
//! - [`dot_q8_q8`]: int8 weights and int8 activations ("W8A8"). The
//!   products are computed on integers, 64 per instruction with AVX-512
//!   VNNI, and the scales are applied once per block.

use crate::quant::{BLOCK, BlockQ8};
use std::sync::OnceLock;

/// The best kernel family this CPU supports, detected once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernels {
    Portable,
    /// AVX2 + FMA.
    Avx2,
    /// AVX-512F for W8A32; AVX-512 VNNI for W8A8.
    Avx512Vnni,
}

impl Kernels {
    pub const ALL: [Kernels; 3] = [Kernels::Avx512Vnni, Kernels::Avx2, Kernels::Portable];

    pub fn is_available(self) -> bool {
        match self {
            Kernels::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Kernels::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Kernels::Avx512Vnni => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }

    pub fn best() -> Self {
        static BEST: OnceLock<Kernels> = OnceLock::new();
        *BEST.get_or_init(|| {
            Kernels::ALL
                .into_iter()
                .find(|k| k.is_available())
                .unwrap_or(Kernels::Portable)
        })
    }
}

/// `Σ w · x` for one row of weight blocks and `x.len() = 64 · w.len()`.
pub fn dot_q8_f32(w: &[BlockQ8], x: &[f32]) -> f32 {
    dot_q8_f32_with(Kernels::best(), w, x).expect("best() only returns available kernels")
}

/// [`dot_q8_f32`] with a specific kernel family, or `None` if unavailable.
pub fn dot_q8_f32_with(kernels: Kernels, w: &[BlockQ8], x: &[f32]) -> Option<f32> {
    assert_eq!(x.len(), w.len() * BLOCK, "x must have 64 values per block");
    if !kernels.is_available() {
        return None;
    }
    Some(match kernels {
        // SAFETY (both arms): the features were just detected.
        #[cfg(target_arch = "x86_64")]
        Kernels::Avx512Vnni => unsafe { x86::dot_q8_f32_avx512(w, x) },
        #[cfg(target_arch = "x86_64")]
        Kernels::Avx2 => unsafe { x86::dot_q8_f32_avx2(w, x) },
        _ => dot_q8_f32_portable(w, x),
    })
}

pub fn dot_q8_f32_portable(w: &[BlockQ8], x: &[f32]) -> f32 {
    let mut total = 0.0;
    for (b, xs) in w.iter().zip(x.chunks_exact(BLOCK)) {
        // Eight running sums, so the compiler can vectorize.
        let mut sums = [0.0f32; 8];
        for (q8, x8) in b.q.chunks_exact(8).zip(xs.chunks_exact(8)) {
            for lane in 0..8 {
                sums[lane] += f32::from(q8[lane]) * x8[lane];
            }
        }
        total += b.scale * sums.iter().sum::<f32>();
    }
    total
}

/// `Σ w · x` where both rows are int8 blocks with their own scales.
pub fn dot_q8_q8(w: &[BlockQ8], x: &[BlockQ8]) -> f32 {
    dot_q8_q8_with(Kernels::best(), w, x).expect("best() only returns available kernels")
}

/// [`dot_q8_q8`] with a specific kernel family, or `None` if unavailable.
pub fn dot_q8_q8_with(kernels: Kernels, w: &[BlockQ8], x: &[BlockQ8]) -> Option<f32> {
    assert_eq!(w.len(), x.len(), "rows must have the same number of blocks");
    if !kernels.is_available() {
        return None;
    }
    Some(match kernels {
        // SAFETY (both arms): the features were just detected.
        #[cfg(target_arch = "x86_64")]
        Kernels::Avx512Vnni => unsafe { x86::dot_q8_q8_vnni(w, x) },
        #[cfg(target_arch = "x86_64")]
        Kernels::Avx2 => unsafe { x86::dot_q8_q8_avx2(w, x) },
        _ => dot_q8_q8_portable(w, x),
    })
}

pub fn dot_q8_q8_portable(w: &[BlockQ8], x: &[BlockQ8]) -> f32 {
    let mut total = 0.0;
    for (bw, bx) in w.iter().zip(x) {
        let int: i32 =
            bw.q.iter()
                .zip(&bx.q)
                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                .sum();
        total += bw.scale * bx.scale * int as f32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    //! AVX2 and AVX-512 versions.

    use crate::quant::{BLOCK, BlockQ8};
    use std::arch::x86_64::{
        __m128i, __m256i, __m512i, _mm_loadl_epi64, _mm_loadu_si128, _mm256_abs_epi8,
        _mm256_add_epi32, _mm256_add_ps, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi16, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
        _mm256_sign_epi8, _mm256_storeu_ps, _mm512_cvtepi8_epi32, _mm512_cvtepi32_ps,
        _mm512_dpbusd_epi32, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_loadu_si512,
        _mm512_reduce_add_ps, _mm512_set1_epi8, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_setzero_si512, _mm512_xor_si512,
    };

    #[target_feature(enable = "avx2,fma")]
    fn sum8(v: std::arch::x86_64::__m256) -> f32 {
        let mut lanes = [0.0f32; 8];
        // SAFETY: `lanes` has room for 8 floats.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
        lanes.iter().sum()
    }

    /// AVX-512 W8A32: per block, four groups of 16 weights are sign-extended
    /// to 32-bit integers, converted to `f32` and multiplied with the
    /// activations; the block's scale is applied once, at the end of the
    /// block.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F. (Lengths are checked.)
    #[target_feature(enable = "avx512f")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    pub fn dot_q8_f32_avx512(w: &[BlockQ8], x: &[f32]) -> f32 {
        assert_eq!(x.len(), w.len() * BLOCK, "x must have 64 values per block");
        let mut total = _mm512_setzero_ps();
        for (b, xs) in w.iter().zip(x.chunks_exact(BLOCK)) {
            let mut acc = _mm512_setzero_ps();
            for part in 0..4 {
                // SAFETY: 16 bytes at 16 * part are inside the 64-byte `q`,
                // and 16 floats at 16 * part inside the 64-float `xs`.
                let (bytes, xv) = unsafe {
                    (
                        _mm_loadu_si128(b.q.as_ptr().add(16 * part).cast::<__m128i>()),
                        _mm512_loadu_ps(xs.as_ptr().add(16 * part)),
                    )
                };
                let wf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes));
                acc = _mm512_fmadd_ps(wf, xv, acc);
            }
            total = _mm512_fmadd_ps(acc, _mm512_set1_ps(b.scale), total);
        }
        _mm512_reduce_add_ps(total)
    }

    /// AVX2 W8A32: eight groups of 8 weights per block.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA. (Lengths are checked.)
    #[target_feature(enable = "avx2,fma")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    pub fn dot_q8_f32_avx2(w: &[BlockQ8], x: &[f32]) -> f32 {
        assert_eq!(x.len(), w.len() * BLOCK, "x must have 64 values per block");
        let mut total = _mm256_setzero_ps();
        for (b, xs) in w.iter().zip(x.chunks_exact(BLOCK)) {
            let mut acc = [_mm256_setzero_ps(); 2];
            for part in 0..8 {
                // SAFETY: 8 bytes at 8 * part are inside `q`, 8 floats at
                // 8 * part inside `xs`.
                let (bytes, xv) = unsafe {
                    (
                        _mm_loadl_epi64(b.q.as_ptr().add(8 * part).cast::<__m128i>()),
                        _mm256_loadu_ps(xs.as_ptr().add(8 * part)),
                    )
                };
                let wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
                acc[part % 2] = _mm256_fmadd_ps(wf, xv, acc[part % 2]);
            }
            total = _mm256_fmadd_ps(
                _mm256_add_ps(acc[0], acc[1]),
                _mm256_set1_ps(b.scale),
                total,
            );
        }
        sum8(total)
    }

    /// AVX-512 VNNI W8A8. `vpdpbusd` multiplies 64 *unsigned* bytes by 64
    /// signed bytes and adds each group of four products into one of 16
    /// 32-bit lanes. Activations are signed, so each is shifted by 128
    /// (flipping its top bit maps -127..127 to 1..255):
    ///
    /// `Σ (x + 128) · w = Σ x · w + 128 · Σ w`
    ///
    /// and the block's precomputed `Σ w` removes the extra term.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F and AVX-512 VNNI. (Lengths are
    /// checked.)
    #[target_feature(enable = "avx512f,avx512vnni")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    pub fn dot_q8_q8_vnni(w: &[BlockQ8], x: &[BlockQ8]) -> f32 {
        assert_eq!(w.len(), x.len(), "rows must have the same number of blocks");
        let flip = _mm512_set1_epi8(-128);
        let mut total = _mm512_setzero_ps();
        let mut correction = 0.0f32;
        for (bw, bx) in w.iter().zip(x) {
            // SAFETY: each `q` is exactly 64 bytes.
            let (vw, vx) = unsafe {
                (
                    _mm512_loadu_si512(bw.q.as_ptr().cast::<__m512i>()),
                    _mm512_loadu_si512(bx.q.as_ptr().cast::<__m512i>()),
                )
            };
            let unsigned_x = _mm512_xor_si512(vx, flip);
            let ints = _mm512_dpbusd_epi32(_mm512_setzero_si512(), unsigned_x, vw);
            let scale = bw.scale * bx.scale;
            total = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ints), _mm512_set1_ps(scale), total);
            correction += 128.0 * bw.sum as f32 * scale;
        }
        _mm512_reduce_add_ps(total) - correction
    }

    /// AVX2 W8A8, the sign trick: `vpmaddubsw` also wants one unsigned
    /// operand, so multiply `|x|` by `w` with the sign of `x` moved onto it.
    /// Products are at most 127 · 127, and pairs of them fit in 16 bits
    /// without saturating (activations are never -128).
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2. (Lengths are checked.)
    #[target_feature(enable = "avx2,fma")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    pub fn dot_q8_q8_avx2(w: &[BlockQ8], x: &[BlockQ8]) -> f32 {
        assert_eq!(w.len(), x.len(), "rows must have the same number of blocks");
        let ones = _mm256_set1_epi16(1);
        let mut total = _mm256_setzero_ps();
        for (bw, bx) in w.iter().zip(x) {
            let mut ints = _mm256_setzero_si256();
            for half in 0..2 {
                // SAFETY: 32 bytes at 32 * half are inside each 64-byte `q`.
                let (vw, vx) = unsafe {
                    (
                        _mm256_loadu_si256(bw.q.as_ptr().add(32 * half).cast::<__m256i>()),
                        _mm256_loadu_si256(bx.q.as_ptr().add(32 * half).cast::<__m256i>()),
                    )
                };
                let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(vx), _mm256_sign_epi8(vw, vx));
                ints = _mm256_add_epi32(ints, _mm256_madd_epi16(pairs, ones));
            }
            let scale = _mm256_set1_ps(bw.scale * bx.scale);
            total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ints), scale, total);
        }
        sum8(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{Granularity, quantize, quantize_activations};
    use ch06_simd::random_vec;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * (1.0 + b.abs())
    }

    #[test]
    fn w8a32_kernels_agree_with_dequantized_arithmetic() {
        for blocks in [1, 3, 9, 24] {
            let k = blocks * BLOCK;
            let w = quantize(&random_vec(k, 1), 1, k, Granularity::PerBlock);
            let x = random_vec(k, 2);
            let want: f32 = crate::quant::dequantize(&w)
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum();
            for kernels in Kernels::ALL {
                if let Some(got) = dot_q8_f32_with(kernels, &w, &x) {
                    assert!(close(got, want), "{kernels:?}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn w8a8_kernels_agree_with_the_portable_one() {
        for blocks in [1, 3, 9, 24] {
            let k = blocks * BLOCK;
            let w = quantize(&random_vec(k, 3), 1, k, Granularity::PerBlock);
            let mut x = Vec::new();
            quantize_activations(&random_vec(k, 4), k, false, &mut x);
            let want = dot_q8_q8_portable(&w, &x);
            for kernels in Kernels::ALL {
                if let Some(got) = dot_q8_q8_with(kernels, &w, &x) {
                    assert!(close(got, want), "{kernels:?}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn extreme_values_do_not_saturate() {
        // All ±127: the AVX2 pairs reach 2 · 127 · 127 = 32,258 < 32,767.
        let big = vec![127.0f32; BLOCK];
        let neg: Vec<f32> = big.iter().map(|v| -v).collect();
        let w = quantize(&big, 1, BLOCK, Granularity::PerBlock);
        let mut x = Vec::new();
        quantize_activations(&neg, BLOCK, false, &mut x);
        let want = dot_q8_q8_portable(&w, &x);
        assert!(close(want, -127.0 * 127.0 * 64.0));
        for kernels in Kernels::ALL {
            if let Some(got) = dot_q8_q8_with(kernels, &w, &x) {
                assert!(close(got, want), "{kernels:?}");
            }
        }
    }
}
