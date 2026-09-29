//! Chapter 6: SIMD, one instruction working on many numbers at once.
//!
//! The module layout mirrors how production kernels are organized:
//!
//! - portable code that any CPU can run ([`dot_naive`], [`dot_unrolled`]),
//! - one module per instruction set, compiled only for its architecture
//!   (`x86` for AVX2 and AVX-512, `arm` for NEON),
//! - a dispatcher ([`dot`], [`dot_bf16`]) that checks once, at runtime,
//!   which instructions this CPU has and calls the best kernel.
//!
//! All `unsafe` code lives in the architecture modules, and every `unsafe`
//! block carries a `SAFETY` comment saying why it is sound.

use std::sync::OnceLock;

use ch02_numbers::Bf16;

pub mod aligned;
pub use aligned::AlignedVec;

/// Instruction sets this crate has kernels for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// Plain Rust; the compiler vectorizes what it can for the build target.
    Portable,
    /// x86-64 with 256-bit AVX2 vectors and fused multiply-add.
    Avx2Fma,
    /// x86-64 with 512-bit AVX-512 vectors.
    Avx512,
    /// 64-bit ARM (Apple Silicon, Graviton, ...) with 128-bit NEON vectors.
    Neon,
}

impl Isa {
    /// Every variant, in order of preference (best first).
    pub const ALL: [Isa; 4] = [Isa::Avx512, Isa::Avx2Fma, Isa::Neon, Isa::Portable];

    /// Can this CPU run kernels for this instruction set?
    pub fn is_available(self) -> bool {
        match self {
            Isa::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2Fma => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            // NEON is part of the baseline of every 64-bit ARM CPU.
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => true,
            // Kernels for another architecture are never available.
            _ => false,
        }
    }
}

/// The best instruction set available, detected once and then cached.
///
/// `is_x86_feature_detected!` runs the CPUID instruction (itself cached by
/// the standard library); `OnceLock` makes sure we pick the kernel exactly
/// once, even if many threads call `dot` at the same time.
pub fn best_isa() -> Isa {
    static BEST: OnceLock<Isa> = OnceLock::new();
    *BEST.get_or_init(|| {
        Isa::ALL
            .into_iter()
            .find(|isa| isa.is_available())
            .unwrap_or(Isa::Portable)
    })
}

/// Dot product with one running sum: every addition waits for the last.
pub fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dot product with `N` independent running sums, in portable Rust.
///
/// With `N` sums the processor can have `N` additions in flight at once.
/// When `N` is a multiple of the vector width, the compiler also turns the
/// inner loop into vector instructions.
pub fn dot_accumulators<const N: usize>(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (a_chunks, a_rest) = a.as_chunks::<N>();
    let (b_chunks, b_rest) = b.as_chunks::<N>();
    let mut sums = [0.0f32; N];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..N {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        total += x * y;
    }
    total
}

/// The portable kernel used when no SIMD kernel applies: eight sums.
pub fn dot_unrolled(a: &[f32], b: &[f32]) -> f32 {
    dot_accumulators::<8>(a, b)
}

/// Dot product using the best kernel for this CPU.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_with(best_isa(), a, b).expect("best_isa only returns available instruction sets")
}

/// Dot product with a specific instruction set, or `None` if this CPU (or
/// this build's target architecture) does not have it.
pub fn dot_with(isa: Isa, a: &[f32], b: &[f32]) -> Option<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "dot product of slices with different lengths"
    );
    if !isa.is_available() {
        return None;
    }
    Some(match isa {
        // SAFETY (all three arms): `is_available` just confirmed that this
        // CPU supports the instructions the kernel was compiled with.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::dot_avx512(a, b) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { x86::dot_avx2(a, b) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::dot_neon(a, b) },
        _ => dot_unrolled(a, b),
    })
}

/// Dot product of `bf16` weights with `f32` activations, widening each
/// weight to `f32` inside the registers. This is how a model stored in
/// `bf16` runs without ever keeping an `f32` copy of its weights.
pub fn dot_bf16(w: &[Bf16], x: &[f32]) -> f32 {
    dot_bf16_with(best_isa(), w, x).expect("best_isa only returns available instruction sets")
}

/// `dot_bf16` with a specific instruction set, or `None` if unavailable.
pub fn dot_bf16_with(isa: Isa, w: &[Bf16], x: &[f32]) -> Option<f32> {
    assert_eq!(
        w.len(),
        x.len(),
        "dot product of slices with different lengths"
    );
    if !isa.is_available() {
        return None;
    }
    Some(match isa {
        // SAFETY (all three arms): the instruction set was just detected.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::dot_bf16_avx512(w, x) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { x86::dot_bf16_avx2(w, x) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::dot_bf16_neon(w, x) },
        _ => dot_bf16_portable(w, x),
    })
}

/// Portable `bf16` dot product with eight running sums.
pub fn dot_bf16_portable(w: &[Bf16], x: &[f32]) -> f32 {
    let (w8, w_rest) = w.as_chunks::<8>();
    let (x8, x_rest) = x.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for (wc, xc) in w8.iter().zip(x8) {
        for lane in 0..8 {
            sums[lane] += wc[lane].to_f32() * xc[lane];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for (wv, xv) in w_rest.iter().zip(x_rest) {
        total += wv.to_f32() * xv;
    }
    total
}

/// `y = W x` for `f32` weights stored as rows, one dot product per row.
pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32]) {
    assert_eq!(w.len(), x.len() * y.len());
    for (row, out) in w.chunks_exact(x.len()).zip(y.iter_mut()) {
        *out = dot(row, x);
    }
}

/// `y = W x` for `bf16` weights stored as rows.
pub fn matvec_bf16(w: &[Bf16], x: &[f32], y: &mut [f32]) {
    assert_eq!(w.len(), x.len() * y.len());
    for (row, out) in w.chunks_exact(x.len()).zip(y.iter_mut()) {
        *out = dot_bf16(row, x);
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    //! AVX2 and AVX-512 kernels.
    //!
    //! Each kernel is a safe function marked `#[target_feature(enable =
    //! ...)]`. Inside it, the compiler may use those instructions freely,
    //! and arithmetic intrinsics are safe to call. Calling the function
    //! itself from code *without* those features requires `unsafe`: the
    //! caller promises the CPU really has them. Loads through raw pointers
    //! are still `unsafe` and each one says why it stays in bounds.

    use std::arch::x86_64::{
        __m128i, __m256, __m256i, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_loadu_si128,
        _mm_movehl_ps, _mm_shuffle_ps, _mm256_add_ps, _mm256_castps256_ps128, _mm256_castsi256_ps,
        _mm256_cvtepu16_epi32, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_loadu_si256, _mm256_setzero_ps, _mm256_slli_epi32, _mm512_add_ps,
        _mm512_castsi512_ps, _mm512_cvtepu16_epi32, _mm512_fmadd_ps, _mm512_loadu_ps,
        _mm512_reduce_add_ps, _mm512_setzero_ps, _mm512_slli_epi32,
    };

    use ch02_numbers::Bf16;

    /// Adds the eight lanes of a 256-bit vector.
    #[target_feature(enable = "avx2,fma")]
    fn horizontal_sum(v: __m256) -> f32 {
        let low = _mm256_castps256_ps128(v); // lanes 0-3
        let high = _mm256_extractf128_ps::<1>(v); // lanes 4-7
        let four = _mm_add_ps(low, high); // 4 partial sums
        let two = _mm_add_ps(four, _mm_movehl_ps(four, four)); // lanes 0+2, 1+3
        let one = _mm_add_ss(two, _mm_shuffle_ps::<0b01>(two, two)); // lane 0 + lane 1
        _mm_cvtss_f32(one)
    }

    /// AVX2 + FMA dot product: four 8-wide accumulators, 32 floats per step.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA (see [`crate::Isa::is_available`]).
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let (a32, a_rest) = a.as_chunks::<32>();
        let (b32, b_rest) = b.as_chunks::<32>();
        let mut acc = [_mm256_setzero_ps(); 4];
        for (ca, cb) in a32.iter().zip(b32) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: `ca` and `cb` each hold exactly 32 floats, so the 8
                // floats starting at 8*j (j < 4) are inside them. `loadu`
                // does not require any particular alignment.
                let (va, vb) = unsafe {
                    (
                        _mm256_loadu_ps(ca.as_ptr().add(8 * j)),
                        _mm256_loadu_ps(cb.as_ptr().add(8 * j)),
                    )
                };
                *acc_j = _mm256_fmadd_ps(va, vb, *acc_j);
            }
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let mut total = horizontal_sum(sum);
        for (x, y) in a_rest.iter().zip(b_rest) {
            total += x * y;
        }
        total
    }

    /// AVX2 + FMA dot product with `ACC` accumulators of 8 lanes each.
    /// Used in the demo to show how many independent chains the hardware
    /// needs before it stops waiting on the previous FMA.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA (see [`crate::Isa::is_available`]).
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_avx2_accumulators<const ACC: usize>(a: &[f32], b: &[f32]) -> f32 {
        let (a8, _) = a.as_chunks::<8>();
        let (b8, _) = b.as_chunks::<8>();
        let (a_groups, a_rest) = a8.as_chunks::<ACC>();
        let (b_groups, _) = b8.as_chunks::<ACC>();
        assert!(
            a_rest.is_empty() && a.len().is_multiple_of(8),
            "demo kernel: length must be a multiple of 8*ACC"
        );
        let mut acc = [_mm256_setzero_ps(); ACC];
        for (ga, gb) in a_groups.iter().zip(b_groups) {
            for j in 0..ACC {
                // SAFETY: `ga[j]` and `gb[j]` are `[f32; 8]` arrays.
                let (va, vb) = unsafe {
                    (
                        _mm256_loadu_ps(ga[j].as_ptr()),
                        _mm256_loadu_ps(gb[j].as_ptr()),
                    )
                };
                acc[j] = _mm256_fmadd_ps(va, vb, acc[j]);
            }
        }
        let mut sum = _mm256_setzero_ps();
        for v in acc {
            sum = _mm256_add_ps(sum, v);
        }
        horizontal_sum(sum)
    }

    /// AVX-512 dot product: four 16-wide accumulators, 64 floats per step.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F (see [`crate::Isa::is_available`]).
    #[target_feature(enable = "avx512f")]
    pub fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        let (a64, a_rest) = a.as_chunks::<64>();
        let (b64, b_rest) = b.as_chunks::<64>();
        let mut acc = [_mm512_setzero_ps(); 4];
        for (ca, cb) in a64.iter().zip(b64) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: each chunk holds 64 floats; 16 at offset 16*j fit.
                let (va, vb) = unsafe {
                    (
                        _mm512_loadu_ps(ca.as_ptr().add(16 * j)),
                        _mm512_loadu_ps(cb.as_ptr().add(16 * j)),
                    )
                };
                *acc_j = _mm512_fmadd_ps(va, vb, *acc_j);
            }
        }
        let sum = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        let mut total = _mm512_reduce_add_ps(sum);
        for (x, y) in a_rest.iter().zip(b_rest) {
            total += x * y;
        }
        total
    }

    /// Widens 8 `bf16` values to 8 `f32` values: zero-extend each 16-bit
    /// pattern to 32 bits, then shift it into the top half (chapter 2).
    #[target_feature(enable = "avx2,fma")]
    fn bf16x8_to_f32(bits: __m128i) -> __m256 {
        let widened = _mm256_cvtepu16_epi32(bits);
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(widened))
    }

    /// AVX2 + FMA `bf16 · f32` dot product.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA (see [`crate::Isa::is_available`]).
    #[target_feature(enable = "avx2,fma")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that happen to take a vector-typed pointer"
    )]
    pub fn dot_bf16_avx2(w: &[Bf16], x: &[f32]) -> f32 {
        let (w32, w_rest) = w.as_chunks::<32>();
        let (x32, x_rest) = x.as_chunks::<32>();
        let mut acc = [_mm256_setzero_ps(); 4];
        for (cw, cx) in w32.iter().zip(x32) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: `cw` holds 32 `Bf16` (64 bytes) and `Bf16` is
                // `repr(transparent)` over `u16`, so the 16 bytes at element
                // 8*j are in bounds; `cx` holds 32 floats. Unaligned loads.
                let (vw, vx) = unsafe {
                    (
                        _mm_loadu_si128(cw.as_ptr().add(8 * j).cast::<__m128i>()),
                        _mm256_loadu_ps(cx.as_ptr().add(8 * j)),
                    )
                };
                *acc_j = _mm256_fmadd_ps(bf16x8_to_f32(vw), vx, *acc_j);
            }
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let mut total = horizontal_sum(sum);
        for (wv, xv) in w_rest.iter().zip(x_rest) {
            total += wv.to_f32() * xv;
        }
        total
    }

    /// AVX-512 `bf16 · f32` dot product.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F (see [`crate::Isa::is_available`]).
    #[target_feature(enable = "avx512f")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the integer load intrinsics are unaligned loads that happen to take a vector-typed pointer"
    )]
    pub fn dot_bf16_avx512(w: &[Bf16], x: &[f32]) -> f32 {
        let (w64, w_rest) = w.as_chunks::<64>();
        let (x64, x_rest) = x.as_chunks::<64>();
        let mut acc = [_mm512_setzero_ps(); 4];
        for (cw, cx) in w64.iter().zip(x64) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: 16 `Bf16` (32 bytes) and 16 floats at element 16*j
                // are inside the 64-element chunks.
                let (vw, vx) = unsafe {
                    (
                        _mm256_loadu_si256(cw.as_ptr().add(16 * j).cast::<__m256i>()),
                        _mm512_loadu_ps(cx.as_ptr().add(16 * j)),
                    )
                };
                let wf = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(vw)));
                *acc_j = _mm512_fmadd_ps(wf, vx, *acc_j);
            }
        }
        let sum = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
        let mut total = _mm512_reduce_add_ps(sum);
        for (wv, xv) in w_rest.iter().zip(x_rest) {
            total += wv.to_f32() * xv;
        }
        total
    }
}

#[cfg(target_arch = "aarch64")]
pub mod arm {
    //! NEON kernels for 64-bit ARM. NEON is always present on aarch64, but
    //! the intrinsics are still marked with the feature so they inline
    //! correctly, and loads through raw pointers are still `unsafe`.

    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vget_low_u16, vld1q_f32, vld1q_u16,
        vmovl_high_u16, vmovl_u16, vreinterpretq_f32_u32, vshlq_n_u32,
    };

    use ch02_numbers::Bf16;

    /// NEON dot product: four 4-wide accumulators, 16 floats per step.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON, which every aarch64 CPU does.
    #[target_feature(enable = "neon")]
    pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        let (a16, a_rest) = a.as_chunks::<16>();
        let (b16, b_rest) = b.as_chunks::<16>();
        let mut acc = [vdupq_n_f32(0.0); 4];
        for (ca, cb) in a16.iter().zip(b16) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: each chunk holds 16 floats; 4 at offset 4*j fit.
                let (va, vb) = unsafe {
                    (
                        vld1q_f32(ca.as_ptr().add(4 * j)),
                        vld1q_f32(cb.as_ptr().add(4 * j)),
                    )
                };
                *acc_j = vfmaq_f32(*acc_j, va, vb);
            }
        }
        let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        let mut total = vaddvq_f32(sum);
        for (x, y) in a_rest.iter().zip(b_rest) {
            total += x * y;
        }
        total
    }

    /// NEON `bf16 · f32` dot product.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON, which every aarch64 CPU does.
    #[target_feature(enable = "neon")]
    pub fn dot_bf16_neon(w: &[Bf16], x: &[f32]) -> f32 {
        let (w16, w_rest) = w.as_chunks::<16>();
        let (x16, x_rest) = x.as_chunks::<16>();
        let mut acc = [vdupq_n_f32(0.0); 4];
        for (cw, cx) in w16.iter().zip(x16) {
            for half in 0..2 {
                // SAFETY: 8 `Bf16` (16 bytes) at element 8*half are inside
                // the 16-element chunk; `Bf16` is `repr(transparent)` over u16.
                let bits = unsafe { vld1q_u16(cw.as_ptr().add(8 * half).cast::<u16>()) };
                let lo = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(bits))));
                let hi = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_high_u16(bits)));
                // SAFETY: 8 floats at element 8*half are inside `cx`.
                let (x_lo, x_hi) = unsafe {
                    (
                        vld1q_f32(cx.as_ptr().add(8 * half)),
                        vld1q_f32(cx.as_ptr().add(8 * half + 4)),
                    )
                };
                acc[2 * half] = vfmaq_f32(acc[2 * half], lo, x_lo);
                acc[2 * half + 1] = vfmaq_f32(acc[2 * half + 1], hi, x_hi);
            }
        }
        let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        let mut total = vaddvq_f32(sum);
        for (wv, xv) in w_rest.iter().zip(x_rest) {
            total += wv.to_f32() * xv;
        }
        total
    }
}

/// Deterministic pseudo-random numbers in [-1, 1), for tests and demos.
pub fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn close(got: f32, want: f64, len: usize) -> bool {
        (f64::from(got) - want).abs() <= 1e-5 * (len as f64).sqrt() * 4.0 + 1e-6
    }

    #[test]
    fn every_available_kernel_matches_the_reference() {
        // Every length from 0 to 200 hits every tail size of every kernel;
        // starting at offset 1 makes the slices unaligned.
        let a_all = random_vec(5000, 1);
        let b_all = random_vec(5000, 2);
        let lengths = (0..=200).chain([1000, 4096, 4999]);
        for len in lengths {
            let (a, b) = (&a_all[1..=len], &b_all[1..=len]);
            let want = reference(a, b);
            assert!(close(dot_naive(a, b), want, len));
            for isa in Isa::ALL {
                if let Some(got) = dot_with(isa, a, b) {
                    assert!(close(got, want, len), "{isa:?} len {len}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn bf16_kernels_match_the_reference() {
        let w_all: Vec<Bf16> = random_vec(5000, 3)
            .into_iter()
            .map(Bf16::from_f32)
            .collect();
        let x_all = random_vec(5000, 4);
        for len in (0..=200).chain([4096, 4999]) {
            let (w, x) = (&w_all[1..=len], &x_all[1..=len]);
            let wf: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
            let want = reference(&wf, x);
            for isa in Isa::ALL {
                if let Some(got) = dot_bf16_with(isa, w, x) {
                    assert!(close(got, want, len), "{isa:?} len {len}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn aarch64_selects_neon() {
        assert_eq!(best_isa(), Isa::Neon);
    }

    #[test]
    fn best_isa_is_available() {
        assert!(best_isa().is_available());
        assert!(Isa::Portable.is_available());
    }

    #[test]
    fn matvec_uses_every_row() {
        let (rows, cols) = (5, 37);
        let w = random_vec(rows * cols, 5);
        let x = random_vec(cols, 6);
        let mut y = vec![0.0; rows];
        matvec(&w, &x, &mut y);
        for (r, &got) in y.iter().enumerate() {
            let want = reference(&w[r * cols..(r + 1) * cols], &x);
            assert!(close(got, want, cols));
        }
    }
}
