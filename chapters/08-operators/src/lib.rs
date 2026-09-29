//! Chapter 8: the operators between the matrix multiplications.
//!
//! A transformer is mostly matmuls, but between them sit a handful of
//! element-wise and row-wise operators: normalization, activation functions,
//! softmax, residual additions and embedding lookups. They are cheap in
//! FLOPs and easy to get subtly wrong, so each one here comes with a test
//! against a high-precision (`f64`) reference.
//!
//! Every function writes into a buffer the caller provides (or works in
//! place). Nothing in this crate allocates, so the operators can run inside
//! the per-token loop of an inference engine without touching the allocator.

use std::f32::consts::LOG2_E;

// ---------------------------------------------------------------------------
// Element-wise helpers
// ---------------------------------------------------------------------------

/// `x += y`, element by element: the residual connection.
pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// The row of an embedding table for one token, borrowed, not copied.
///
/// The table is `vocab × dim`, row-major. A "lookup" is just a slice.
pub fn embedding(table: &[f32], dim: usize, token: usize) -> &[f32] {
    &table[token * dim..(token + 1) * dim]
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

pub fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

/// The logistic sigmoid `1 / (1 + e^-x)`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (also called swish): `x · sigmoid(x)`. Used by Llama-style models.
pub fn silu(x: &mut [f32]) {
    for v in x {
        *v *= sigmoid(*v);
    }
}

/// SwiGLU, the gated activation of Llama's feed-forward block:
/// `out[i] = silu(gate[i]) · up[i]`. Done in one pass over the data.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = g * sigmoid(g) * u;
    }
}

/// GELU with the `tanh` approximation used by GPT-2 and many others:
/// `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
pub fn gelu_tanh(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in x {
        let u = SQRT_2_OVER_PI * (*v + 0.044_715 * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + u.tanh());
    }
}

/// GELU as defined: `0.5 x (1 + erf(x / √2))`.
pub fn gelu_erf(x: &mut [f32]) {
    for v in x {
        *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
}

/// The error function. Abramowitz and Stegun's formula 7.1.26 is accurate
/// to 1.5e-7 in exact arithmetic; evaluated in `f32` it stays within about
/// 5e-7. Rust's standard library has no stable `erf`.
pub fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let y = 1.0 - poly * (-x * x).exp();
    y.copysign(x)
}

// ---------------------------------------------------------------------------
// exp and softmax
// ---------------------------------------------------------------------------

/// Rounds to the nearest integer without calling `round` (a library call
/// on baseline x86-64): adding 1.5·2^23 pushes the fraction bits out of the
/// mantissa, and subtracting it back leaves the rounded value. Valid for
/// |x| < 2^22.
const ROUNDER: f32 = 12_582_912.0;

/// ln 2 split into a part with trailing zero bits (so `n · LN2_HI` is exact
/// for the small integers `n` we use) and a small correction.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;

/// A fast `e^x` that the compiler can vectorize, with a worst-case relative
/// error of about 8e-8 (under one unit in the last place) for x in [-87, 88].
///
/// The standard `f32::exp` calls a library function once per element,
/// which blocks vectorization. This version uses only arithmetic and bit
/// manipulation:
///
/// 1. Write `x = n·ln 2 + r` with `n` an integer and `|r| ≤ ln 2 / 2`.
/// 2. Then `e^x = 2^n · e^r`.
/// 3. `2^n` is built directly as the bits of an `f32` (exponent field).
/// 4. `e^r` is a degree-7 polynomial, accurate because `r` is small. The
///    coefficients are the minimax fit from the Cephes library, which is
///    about 6x more accurate than the same-length Taylor series.
#[inline]
pub fn exp_fast(x: f32) -> f32 {
    // Outside this range the result would underflow or overflow; clamping
    // keeps n in the range where 2^n is a normal f32.
    let x = x.clamp(-87.0, 88.0);
    // After adding ROUNDER, the integer n = round(x·log2 e) sits in the low
    // mantissa bits of `shifted`: its bit pattern is ROUNDER's plus n.
    let shifted = x * LOG2_E + ROUNDER;
    let n = shifted - ROUNDER;
    let r = x - n * LN2_HI - n * LN2_LO;
    // e^r ≈ 1 + r + r²·P(r), P evaluated in Horner form.
    let mut p = 1.987_569_1e-4_f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    let e_r = p * r * r + r + 1.0;
    // 2^n is an f32 whose exponent field is n + 127. We get n + 127 with
    // integer arithmetic on the bits of `shifted`, not with `n as i32`:
    // Rust's float-to-int `as` saturates out-of-range values, which costs
    // extra instructions in every vector lane.
    let two_to_n = f32::from_bits(shifted.to_bits().wrapping_sub(ROUNDER.to_bits() - 127) << 23);
    e_r * two_to_n
}

/// Softmax computed the textbook way: `e^x_i / Σ e^x_j`. Overflows to
/// `inf / inf = NaN` as soon as any input is above about 88.
pub fn softmax_naive(x: &mut [f32]) {
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = v.exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Numerically stable softmax, in place.
///
/// Subtracting the maximum first changes nothing mathematically (the factor
/// `e^-max` cancels between numerator and denominator) but makes every
/// exponent ≤ 0, so every `e^x` is in (0, 1] and nothing can overflow.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Stable softmax using [`exp_fast`], written so the compiler can vectorize
/// every loop (eight running sums, no library calls).
pub fn softmax_fast(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (chunks, rest) = x.as_chunks_mut::<8>();
    let mut sums = [0.0f32; 8];
    for chunk in chunks {
        for lane in 0..8 {
            chunk[lane] = exp_fast(chunk[lane] - max);
            sums[lane] += chunk[lane];
        }
    }
    let mut sum: f32 = sums.iter().sum();
    for v in rest {
        *v = exp_fast(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `log(softmax(x))`, computed without ever forming the probabilities:
/// `x_i - max - log(Σ e^(x_j - max))`. Used for log-probabilities and
/// perplexity (chapter 18), where `log(softmax)` of a tiny probability
/// would round to `log(0) = -inf`.
pub fn log_softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    let log_sum = sum.ln();
    for v in x.iter_mut() {
        *v = *v - max - log_sum;
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// RMSNorm (Llama, Mistral, Qwen, SmolLM...):
/// `out_i = x_i / sqrt(mean(x²) + eps) · weight_i`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

/// LayerNorm (GPT-2, BERT...): subtract the mean, divide by the standard
/// deviation, then scale and shift:
/// `out_i = (x_i - mean) / sqrt(var + eps) · weight_i + bias_i`.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    assert!(x.len() == weight.len() && x.len() == bias.len() && x.len() == out.len());
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    // Two passes (mean first, then variance around it) is more accurate
    // than the one-pass formula E[x²] - E[x]², which can cancel badly.
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let scale = 1.0 / (var + eps).sqrt();
    for (((o, &v), &w), &b) in out.iter_mut().zip(x).zip(weight).zip(bias) {
        *o = (v - mean) * scale * w + b;
    }
}

/// A convenience that allocates its result. Used only to show what an
/// allocation per call costs; engine code uses [`rms_norm`].
pub fn rms_norm_alloc(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    rms_norm(x, weight, eps, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sweep(from: f32, to: f32, steps: usize) -> impl Iterator<Item = f32> {
        (0..=steps).map(move |i| from + (to - from) * i as f32 / steps as f32)
    }

    #[test]
    fn exp_fast_is_accurate() {
        let mut worst = 0.0f64;
        for x in sweep(-87.0, 88.0, 2_000_000) {
            let want = f64::from(x).exp();
            let rel = ((f64::from(exp_fast(x)) - want) / want).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 1.5e-7, "worst relative error {worst:e}");
        assert!((exp_fast(0.0) - 1.0).abs() < f32::EPSILON);
        assert!(exp_fast(-1000.0) >= 0.0 && exp_fast(-1000.0) < 1e-37);
    }

    #[test]
    fn softmax_is_stable_and_correct() {
        let logits = [1000.0f32, 999.0, 998.0, -5.0];
        let mut naive = logits;
        softmax_naive(&mut naive);
        assert!(naive[0].is_nan(), "the naive version overflows");

        let max = 1000.0f64;
        let denom: f64 = logits.iter().map(|&v| (f64::from(v) - max).exp()).sum();
        for f in [softmax as fn(&mut [f32]), softmax_fast] {
            let mut p = logits;
            f(&mut p);
            for (&got, &l) in p.iter().zip(&logits) {
                let want = (f64::from(l) - max).exp() / denom;
                assert!((f64::from(got) - want).abs() < 1e-6);
            }
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_on_a_vocabulary_matches_f64() {
        let mut state = 7u64;
        let logits: Vec<f32> = (0..49_152)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32 * 30.0 - 15.0
            })
            .collect();
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let denom: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
        let worst = |p: &[f32]| {
            p.iter()
                .zip(&logits)
                .map(|(&got, &l)| {
                    let want = (f64::from(l) - max).exp() / denom;
                    ((f64::from(got) - want) / want).abs()
                })
                .fold(0.0f64, f64::max)
        };
        let (mut plain, mut fast) = (logits.clone(), logits.clone());
        softmax(&mut plain);
        softmax_fast(&mut fast);
        // The plain version adds 49,152 terms into one running sum, and its
        // error is dominated by that sum (chapter 2). Eight running sums
        // make the fast version *more* accurate despite its approximate exp.
        // Measured on the reference machine: plain 2.8e-5, fast 4.2e-6.
        assert!(worst(&plain) < 5e-5, "plain: {:e}", worst(&plain));
        assert!(worst(&fast) < 1e-5, "fast: {:e}", worst(&fast));
        assert!(worst(&fast) < worst(&plain));
    }

    #[test]
    fn log_softmax_matches_log_of_softmax() {
        let logits = [2.0f32, -1.0, 0.5, 3.0, -40.0];
        let (mut lp, mut p) = (logits, logits);
        log_softmax(&mut lp);
        softmax(&mut p);
        for (l, q) in lp.iter().zip(&p) {
            assert!((l - q.ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn rms_norm_matches_the_formula() {
        let x: Vec<f32> = (0..576).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let w: Vec<f32> = (0..576).map(|i| 1.0 + i as f32 / 576.0).collect();
        let mut out = vec![0.0; 576];
        rms_norm(&x, &w, 1e-5, &mut out);
        let ms: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 576.0;
        for i in 0..576 {
            let want = f64::from(x[i]) / (ms + 1e-5).sqrt() * f64::from(w[i]);
            assert!((f64::from(out[i]) - want).abs() < 1e-5);
        }
    }

    #[test]
    fn layer_norm_output_has_zero_mean_and_unit_variance() {
        let x: Vec<f32> = (0..768).map(|i| 100.0 + (i as f32).cos()).collect();
        let (w, b) = (vec![1.0; 768], vec![0.0; 768]);
        let mut out = vec![0.0; 768];
        layer_norm(&x, &w, &b, 1e-5, &mut out);
        let mean: f64 = out.iter().map(|&v| f64::from(v)).sum::<f64>() / 768.0;
        let var: f64 = out
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / 768.0;
        assert!(mean.abs() < 1e-4);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn activations_match_their_formulas() {
        let xs: Vec<f32> = sweep(-6.0, 6.0, 1000).collect();
        let (mut s, mut gt, mut ge) = (xs.clone(), xs.clone(), xs.clone());
        silu(&mut s);
        gelu_tanh(&mut gt);
        gelu_erf(&mut ge);
        for (i, &x) in xs.iter().enumerate() {
            let x64 = f64::from(x);
            let silu_want = x64 / (1.0 + (-x64).exp());
            assert!((f64::from(s[i]) - silu_want).abs() < 1e-6);
            // The two GELU variants differ by at most a few 1e-4.
            assert!((gt[i] - ge[i]).abs() < 1e-3);
        }
        // Reference values from a high-precision table.
        for (x, want) in [
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (2.0, 0.995_322_265),
        ] {
            assert!((f64::from(erf(x)) - want).abs() < 5e-7);
            assert!((f64::from(erf(-x)) + want).abs() < 5e-7);
        }
    }

    #[test]
    fn swiglu_is_silu_times_up() {
        let gate = [-2.0f32, 0.0, 1.5];
        let up = [3.0f32, 5.0, -2.0];
        let mut out = [0.0; 3];
        swiglu(&gate, &up, &mut out);
        let mut g = gate;
        silu(&mut g);
        for i in 0..3 {
            assert!((out[i] - g[i] * up[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn embedding_is_a_borrowed_row() {
        let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let row = embedding(&table, 4, 2);
        assert_eq!(row, &[8.0, 9.0, 10.0, 11.0]);
        assert!(std::ptr::eq(row.as_ptr(), table[8..].as_ptr()));
    }
}
