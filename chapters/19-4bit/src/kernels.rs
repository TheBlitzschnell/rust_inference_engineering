//! Dot products of 4-bit weight blocks with activations.
//!
//! For one block, with weights `w_i = scale · c_i + min`:
//!
//! ```text
//! Σ w_i x_i = scale · Σ c_i x_i + min · Σ x_i
//! ```
//!
//! With activations quantized to int8 blocks (chapter 18,
//! `x_i ≈ xs · xq_i`, `Σ xq_i` stored in the block), both sums are integer
//! sums: `Σ c_i xq_i` (4-bit codes times int8) and `Σ xq_i` (precomputed).
//! The codes are unsigned (0..=15), which is exactly the operand
//! `vpdpbusd` wants unsigned, so no correction is needed.

use crate::quant::BlockQ4;
use ch18_int8::{BLOCK, BlockQ8, Kernels};

/// 4-bit weights times int8 activation blocks ("W4A8").
pub fn dot_q4_q8(w: &[BlockQ4], x: &[BlockQ8]) -> f32 {
    dot_q4_q8_with(Kernels::best(), w, x).expect("best() only returns available kernels")
}

pub fn dot_q4_q8_with(kernels: Kernels, w: &[BlockQ4], x: &[BlockQ8]) -> Option<f32> {
    assert_eq!(w.len(), x.len(), "rows must have the same number of blocks");
    if !kernels.is_available() {
        return None;
    }
    Some(match kernels {
        // SAFETY (both arms): the features were just detected.
        #[cfg(target_arch = "x86_64")]
        Kernels::Avx512Vnni => unsafe { x86::dot_q4_q8_vnni(w, x) },
        #[cfg(target_arch = "x86_64")]
        Kernels::Avx2 => unsafe { x86::dot_q4_q8_avx2(w, x) },
        _ => dot_q4_q8_portable(w, x),
    })
}

pub fn dot_q4_q8_portable(w: &[BlockQ4], x: &[BlockQ8]) -> f32 {
    let mut total = 0.0;
    for (bw, bx) in w.iter().zip(x) {
        let int: i32 = (0..BLOCK)
            .map(|i| i32::from(bw.code(i)) * i32::from(bx.q[i]))
            .sum();
        total += bx.scale * (bw.scale * int as f32 + bw.min * bx.sum as f32);
    }
    total
}

/// 4-bit weights times `f32` activations ("W4A32").
pub fn dot_q4_f32(w: &[BlockQ4], x: &[f32]) -> f32 {
    assert_eq!(x.len(), w.len() * BLOCK, "x must have 64 values per block");
    let mut total = 0.0;
    for (b, xs) in w.iter().zip(x.chunks_exact(BLOCK)) {
        let (mut cx, mut sx) = (0.0f32, 0.0f32);
        for (i, &v) in xs.iter().enumerate() {
            cx += f32::from(b.code(i)) * v;
            sx += v;
        }
        total += b.scale * cx + b.min * sx;
    }
    total
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    //! AVX2 and AVX-512 VNNI versions.

    use crate::quant::BlockQ4;
    use ch18_int8::BlockQ8;
    use std::arch::x86_64::{
        __m256i, __m512i, _mm256_add_epi32, _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8,
        _mm256_set1_epi16, _mm256_set1_ps, _mm256_setzero_ps, _mm256_srli_epi16, _mm256_storeu_ps,
        _mm512_castsi256_si512, _mm512_cvtepi32_ps, _mm512_dpbusd_epi32, _mm512_fmadd_ps,
        _mm512_inserti64x4, _mm512_loadu_si512, _mm512_reduce_add_ps, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_setzero_si512,
    };

    /// The 32 packed bytes of a block as two vectors of 32 codes: the low
    /// nibbles (values 0..32) and the high nibbles (values 32..64).
    ///
    /// # Safety
    ///
    /// AVX2 must be available.
    #[target_feature(enable = "avx2")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    unsafe fn unpack(b: &BlockQ4) -> (__m256i, __m256i) {
        // SAFETY: `packed` is exactly 32 bytes.
        let bytes = unsafe { _mm256_loadu_si256(b.packed.as_ptr().cast::<__m256i>()) };
        let mask = _mm256_set1_epi8(0x0F);
        let lo = _mm256_and_si256(bytes, mask);
        // Shift 16-bit lanes right by 4, then mask: each byte's high nibble.
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), mask);
        (lo, hi)
    }

    /// AVX-512 VNNI: the 64 codes (unsigned) against 64 int8 activations in
    /// one `vpdpbusd`.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F, AVX-512 VNNI and AVX2. (Lengths are
    /// checked.)
    #[target_feature(enable = "avx512f,avx512vnni,avx2")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    pub fn dot_q4_q8_vnni(w: &[BlockQ4], x: &[BlockQ8]) -> f32 {
        assert_eq!(w.len(), x.len(), "rows must have the same number of blocks");
        let mut total = _mm512_setzero_ps();
        let mut offsets = 0.0f32;
        for (bw, bx) in w.iter().zip(x) {
            // SAFETY: AVX2 is enabled here; `q` is exactly 64 bytes.
            let ((lo, hi), xq) = unsafe {
                (
                    unpack(bw),
                    _mm512_loadu_si512(bx.q.as_ptr().cast::<__m512i>()),
                )
            };
            let codes = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            let ints = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, xq);
            let scale = bw.scale * bx.scale;
            total = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ints), _mm512_set1_ps(scale), total);
            offsets += bw.min * bx.scale * bx.sum as f32;
        }
        _mm512_reduce_add_ps(total) + offsets
    }

    /// AVX2: `vpmaddubsw` multiplies unsigned bytes (the codes, at most 15)
    /// by signed bytes (the activations); pairs reach at most 2 · 15 · 127,
    /// far from 16-bit saturation.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA. (Lengths are checked.)
    #[target_feature(enable = "avx2,fma")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that take a vector-typed pointer"
    )]
    pub fn dot_q4_q8_avx2(w: &[BlockQ4], x: &[BlockQ8]) -> f32 {
        assert_eq!(w.len(), x.len(), "rows must have the same number of blocks");
        let ones = _mm256_set1_epi16(1);
        let mut total = _mm256_setzero_ps();
        let mut offsets = 0.0f32;
        for (bw, bx) in w.iter().zip(x) {
            // SAFETY: AVX2 is enabled; `q` holds 64 bytes, 32 at 0 and 32 at 32.
            let ((lo, hi), x0, x1) = unsafe {
                (
                    unpack(bw),
                    _mm256_loadu_si256(bx.q.as_ptr().cast::<__m256i>()),
                    _mm256_loadu_si256(bx.q.as_ptr().add(32).cast::<__m256i>()),
                )
            };
            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), ones);
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), ones);
            let ints = _mm256_add_epi32(p0, p1);
            let scale = _mm256_set1_ps(bw.scale * bx.scale);
            total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ints), scale, total);
            offsets += bw.min * bx.scale * bx.sum as f32;
        }
        let mut lanes = [0.0f32; 8];
        // SAFETY: `lanes` has room for 8 floats.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), total) };
        lanes.iter().sum::<f32>() + offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{Scheme, dequantize, quantize};
    use ch06_simd::random_vec;
    use ch18_int8::quantize_activations;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * (1.0 + b.abs())
    }

    #[test]
    fn w4a32_matches_dequantized_arithmetic() {
        for scheme in [Scheme::Symmetric, Scheme::MinMax] {
            let k = 5 * BLOCK;
            let w = quantize(&random_vec(k, 1), 1, k, BLOCK, scheme);
            let x = random_vec(k, 2);
            let want: f32 = dequantize(&w).iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!(close(dot_q4_f32(&w, &x), want));
        }
    }

    #[test]
    fn w4a8_kernels_agree_with_the_portable_one() {
        for scheme in [Scheme::Symmetric, Scheme::SymmetricSearch, Scheme::MinMax] {
            for blocks in [1, 3, 9, 24] {
                let k = blocks * BLOCK;
                let w = quantize(&random_vec(k, 3), 1, k, BLOCK, scheme);
                let mut x = Vec::new();
                quantize_activations(&random_vec(k, 4), k, false, &mut x);
                let want = dot_q4_q8_portable(&w, &x);
                for kernels in Kernels::ALL {
                    if let Some(got) = dot_q4_q8_with(kernels, &w, &x) {
                        assert!(close(got, want), "{scheme:?} {kernels:?}: {got} vs {want}");
                    }
                }
            }
        }
    }
}
