//! Chapter 2: the number formats models are stored and computed in.
//!
//! Rust has `f32` and `f64` built in. Inference also needs smaller formats:
//! `bf16` and `f16` (16 bits) for weights and activations, and `fp8` (8 bits)
//! on the newest accelerators. This crate implements all three from scratch,
//! on top of plain integers, so every rounding decision is visible.

/// The sign, exponent and mantissa fields of an `f32`, pulled apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32Fields {
    /// 0 for positive, 1 for negative.
    pub sign: u32,
    /// The stored exponent, 0..=255. The real exponent is `exponent - 127`.
    pub exponent: u32,
    /// The 23 stored fraction bits. Normal numbers have an implicit leading 1.
    pub mantissa: u32,
}

/// Splits an `f32` into its three bit fields.
///
/// `to_bits` reinterprets the four bytes as a `u32` without changing them.
/// This is different from `x as u32`, which converts the *value* (and would
/// turn 1.5 into 1).
pub fn f32_fields(x: f32) -> F32Fields {
    let bits = x.to_bits();
    F32Fields {
        sign: bits >> 31,
        exponent: (bits >> 23) & 0xFF,
        mantissa: bits & 0x007F_FFFF,
    }
}

/// A 16-bit "brain float": 1 sign bit, 8 exponent bits, 7 mantissa bits.
///
/// It is exactly the top half of an `f32`. Same range as `f32`, far less
/// precision.
///
/// `#[repr(transparent)]` guarantees this struct has the same memory layout
/// as a bare `u16`, so a slice of `Bf16` can be viewed as raw file bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bf16(u16);

impl Bf16 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(0x3F80);
    pub const INFINITY: Self = Self(0x7F80);

    /// Wraps raw bits, as read from a weight file.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts to `f32`. This is exact: every `bf16` value is also an `f32`.
    /// Put the 16 bits in the top half of a 32-bit word, and zero the rest.
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }

    /// Converts from `f32`, rounding to the nearest `bf16`. When `x` is
    /// exactly halfway between two `bf16` values, it goes to the one whose
    /// last bit is 0 ("round half to even"). This is the IEEE 754 default and
    /// what PyTorch does.
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        if x.is_nan() {
            // Truncating a NaN could clear every mantissa bit that survives,
            // which would turn it into infinity. Force a mantissa bit on.
            return Self((bits >> 16) as u16 | 0x0040);
        }
        // We are about to throw away the low 16 bits. Adding 0x7FFF rounds
        // up exactly when those bits are more than half (0x8000). Adding the
        // lowest kept bit as well makes an exact half round up only when
        // that bit is 1, which is what "ties to even" means.
        let lowest_kept_bit = (bits >> 16) & 1;
        let rounded = bits + 0x7FFF + lowest_kept_bit;
        Self((rounded >> 16) as u16)
    }

    /// Converts by chopping off the low 16 bits, with no rounding. Cheaper,
    /// and about twice the error of `from_f32` on average. Shown for comparison.
    pub fn from_f32_truncate(x: f32) -> Self {
        Self((x.to_bits() >> 16) as u16)
    }
}

/// IEEE 754 half precision: 1 sign bit, 5 exponent bits, 10 mantissa bits.
///
/// More precise than `bf16`, but the largest finite value is 65504. Anything
/// bigger becomes infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(0x3C00);
    pub const MAX: Self = Self(0x7BFF);
    pub const INFINITY: Self = Self(0x7C00);

    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts to `f32`. Exact: `f32` has more exponent and mantissa bits.
    pub fn to_f32(self) -> f32 {
        let h = u32::from(self.0);
        let sign = (h & 0x8000) << 16;
        let exponent = (h >> 10) & 0x1F;
        let mantissa = h & 0x03FF;
        match exponent {
            // Zero or subnormal: value = mantissa × 2^-24. Multiplying a
            // small integer by a power of two is exact in f32.
            0 => {
                let magnitude = mantissa as f32 * f32::from_bits(0x3380_0000); // 2^-24
                if sign == 0 { magnitude } else { -magnitude }
            }
            // Infinity (mantissa 0) or NaN (mantissa non-zero).
            0x1F => f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13)),
            // Normal: rebias the exponent from 15 to 127, widen the mantissa.
            _ => f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)),
        }
    }

    /// Converts from `f32` with round-half-to-even. Values too large become
    /// infinity; values too small become subnormal numbers or zero.
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        if exponent == 0xFF {
            // Infinity stays infinity; any NaN becomes a quiet NaN.
            let nan_bit = if mantissa == 0 { 0 } else { 0x0200 };
            return Self(sign | 0x7C00 | nan_bit);
        }

        let e = exponent - 127; // the real exponent
        if e > 15 {
            return Self(sign | 0x7C00); // too big for f16
        }
        if e >= -14 {
            // A normal f16. Keep the top 10 of 23 mantissa bits and round
            // on the 13 we drop. If rounding carries out of the mantissa it
            // correctly bumps the exponent (and 65520 and up become infinity).
            let mut half = (((e + 15) as u32) << 10) | (mantissa >> 13);
            let dropped = mantissa & 0x1FFF;
            if round_up(dropped, 0x1000, half) {
                half += 1;
            }
            return Self(sign | half as u16);
        }
        if e < -25 {
            return Self(sign); // below half the smallest subnormal: zero
        }
        // Subnormal f16: count units of 2^-24. Put the implicit leading 1
        // back, then shift right until one unit is worth 2^-24.
        let significand = mantissa | 0x0080_0000;
        let shift = (-e - 1) as u32; // between 14 and 24
        let mut half = significand >> shift;
        let dropped = significand & ((1 << shift) - 1);
        if round_up(dropped, 1 << (shift - 1), half) {
            half += 1;
        }
        Self(sign | half as u16)
    }
}

/// The round-half-to-even decision. `dropped` are the bits being thrown
/// away, `halfway` is the value of exactly one half unit, `kept` is what
/// remains.
fn round_up(dropped: u32, halfway: u32, kept: u32) -> bool {
    dropped > halfway || (dropped == halfway && kept & 1 == 1)
}

/// FP8 in the E4M3 layout: 1 sign bit, 4 exponent bits, 3 mantissa bits.
///
/// Used for weights and activations on recent GPUs (NVIDIA H100 and later).
/// It has no infinity: the largest value is 448, and bit patterns
/// `S.1111.111` are NaN. With only 256 possible values, conversion from `f32`
/// can simply search a sorted table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Fp8E4M3(u8);

impl Fp8E4M3 {
    pub const MAX: f32 = 448.0;

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u8 {
        self.0
    }

    pub fn to_f32(self) -> f32 {
        let sign = if self.0 & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = i32::from((self.0 >> 3) & 0x0F);
        let mantissa_bits = self.0 & 0x07;
        if exponent == 0x0F && mantissa_bits == 0x07 {
            return f32::NAN;
        }
        let mantissa = f32::from(mantissa_bits);
        if exponent == 0 {
            // Subnormal: mantissa/8 × 2^(1 - 7)
            return sign * mantissa / 8.0 * 2f32.powi(-6);
        }
        sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }

    /// Rounds to the nearest representable value, ties to even. Values beyond
    /// ±448 saturate to ±448, which is what inference libraries do (there is
    /// no infinity to overflow into).
    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self(0x7F);
        }
        let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
        let magnitude = x.abs().min(Self::MAX);
        // Positive patterns 0x00..=0x7E are sorted by value, so find the
        // first one that is >= magnitude and compare it with the one below.
        let upper = (0u8..=0x7E)
            .find(|&b| Self(b).to_f32() >= magnitude)
            .expect("magnitude is clamped to MAX, which is 0x7E");
        if upper == 0 {
            return Self(sign);
        }
        let lower = upper - 1;
        let (lo, hi) = (Self(lower).to_f32(), Self(upper).to_f32());
        let pick = if magnitude - lo < hi - magnitude {
            lower
        } else if hi - magnitude < magnitude - lo {
            upper
        } else if lower % 2 == 0 {
            lower
        } else {
            upper
        };
        Self(sign | pick)
    }
}

/// Converts a slice of `bf16` weights to `f32`. Used when loading a model
/// stored in bf16 into a kernel that computes in f32.
pub fn bf16_to_f32_slice(src: &[Bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

pub fn f16_to_f32_slice(src: &[F16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

/// Adds numbers with an `f32` running total.
pub fn sum_f32(xs: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for &x in xs {
        total += x;
    }
    total
}

/// Adds numbers but stores the running total in `bf16` after every step,
/// the way a kernel would if it accumulated in the storage format.
pub fn sum_bf16_accumulator(xs: &[f32]) -> f32 {
    let mut total = Bf16::ZERO;
    for &x in xs {
        total = Bf16::from_f32(total.to_f32() + x);
    }
    total.to_f32()
}

/// Adds numbers in `f64`. Used as the "true" answer.
pub fn sum_f64(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| f64::from(x)).sum()
}

/// Kahan (compensated) summation: keeps a second `f32` holding the rounding
/// error of the main total, and feeds it back in on the next step.
pub fn sum_kahan(xs: &[f32]) -> f32 {
    let mut total = 0.0f32;
    let mut compensation = 0.0f32;
    for &x in xs {
        let y = x - compensation;
        let t = total + y;
        compensation = (t - total) - y;
        total = t;
    }
    total
}

/// Pairwise summation: split in half, sum each half, add the two results.
/// Errors grow with log(n) instead of n. This is roughly what a parallel or
/// SIMD reduction does, and why it gives slightly different answers from a
/// plain loop.
pub fn sum_pairwise(xs: &[f32]) -> f32 {
    if xs.len() <= 8 {
        return sum_f32(xs);
    }
    let (left, right) = xs.split_at(xs.len() / 2);
    sum_pairwise(left) + sum_pairwise(right)
}

/// A small deterministic pseudo-random generator (xorshift64*), so the demo
/// and tests do not need an external crate. Chapter 15 covers RNGs properly.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-1, 1).
    pub fn next_f32(&mut self) -> f32 {
        // The top 24 bits give every representable step of an f32 in [0, 1).
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "format conversions are exact by design, so tests compare exactly"
)]
mod tests {
    use super::*;

    /// Sorted positive values of a 16-bit format, including infinity at the
    /// end (treated as the next step up so that overflow rounding works).
    fn positive_values(decode: impl Fn(u16) -> f32, infinity: u16) -> Vec<f64> {
        (0..=infinity)
            .map(|b| {
                if b == infinity {
                    // One step above the largest finite value, same spacing.
                    let max = f64::from(decode(b - 1));
                    let below = f64::from(decode(b - 2));
                    max + (max - below)
                } else {
                    f64::from(decode(b))
                }
            })
            .collect()
    }

    /// Reference rounding by brute force: find the neighbours of |x| in the
    /// sorted table and pick the nearer, ties to the even bit pattern.
    fn reference_round(x: f32, table: &[f64]) -> u16 {
        let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
        let m = f64::from(x.abs());
        let upper = table.partition_point(|&v| v < m);
        if upper == table.len() {
            return sign | (table.len() - 1) as u16; // beyond infinity
        }
        if upper == 0 || table[upper] == m {
            return sign | upper as u16;
        }
        let lower = upper - 1;
        let (dl, du) = (m - table[lower], table[upper] - m);
        let pick = if dl < du {
            lower
        } else if du < dl || lower % 2 == 1 {
            upper
        } else {
            lower
        };
        sign | pick as u16
    }

    /// Walks a sample of all 2^32 f32 bit patterns.
    fn sample_f32() -> impl Iterator<Item = f32> {
        (0..=u32::MAX)
            .step_by(4099)
            .map(f32::from_bits)
            .filter(|x| x.is_finite())
    }

    #[test]
    fn bf16_round_trips_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let b = Bf16::from_bits(bits);
            let back = Bf16::from_f32(b.to_f32());
            if b.to_f32().is_nan() {
                assert!(back.to_f32().is_nan());
            } else {
                assert_eq!(back, b, "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn f16_round_trips_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let h = F16::from_bits(bits);
            let back = F16::from_f32(h.to_f32());
            if h.to_f32().is_nan() {
                assert!(back.to_f32().is_nan());
            } else {
                assert_eq!(back, h, "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn f16_decodes_to_the_textbook_formula() {
        for bits in 0..0x7C00u16 {
            let exponent = i32::from(bits >> 10);
            let mantissa = f64::from(bits & 0x3FF);
            let expected = if exponent == 0 {
                mantissa / 1024.0 * 2f64.powi(-14)
            } else {
                (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15)
            };
            assert_eq!(f64::from(F16::from_bits(bits).to_f32()), expected);
        }
    }

    #[test]
    fn bf16_rounding_matches_brute_force() {
        let table = positive_values(|b| Bf16::from_bits(b).to_f32(), 0x7F80);
        for x in sample_f32() {
            let expected = reference_round(x, &table);
            assert_eq!(Bf16::from_f32(x).to_bits(), expected, "x = {x:e}");
        }
    }

    #[test]
    fn f16_rounding_matches_brute_force() {
        let table = positive_values(|b| F16::from_bits(b).to_f32(), 0x7C00);
        for x in sample_f32() {
            let expected = reference_round(x, &table);
            assert_eq!(F16::from_f32(x).to_bits(), expected, "x = {x:e}");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(Bf16::from_f32(1.0), Bf16::ONE);
        assert_eq!(F16::from_f32(1.0), F16::ONE);
        assert_eq!(F16::from_f32(65504.0), F16::MAX);
        assert_eq!(F16::from_f32(65519.0), F16::MAX); // rounds down
        assert_eq!(F16::from_f32(65520.0), F16::INFINITY); // tie rounds to even = inf
        assert_eq!(F16::from_f32(1e-8).to_f32(), 0.0); // underflow
        let big = Bf16::from_f32(1e30).to_f32(); // bf16 keeps f32's range
        assert!((big - 1e30).abs() / 1e30 < 1.0 / 256.0);
        assert!(Bf16::from_f32(f32::NAN).to_f32().is_nan());
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
        assert_eq!(Bf16::from_f32(-0.0).to_bits(), 0x8000); // sign of zero kept
    }

    #[test]
    fn fp8_round_trips_and_saturates() {
        for bits in 0..=u8::MAX {
            let v = Fp8E4M3::from_bits(bits);
            if v.to_f32().is_nan() {
                continue;
            }
            let back = Fp8E4M3::from_f32(v.to_f32());
            // +0 and -0 both exist; everything else must match exactly.
            assert_eq!(back.to_f32(), v.to_f32(), "bits {bits:#04x}");
        }
        assert_eq!(Fp8E4M3::from_f32(1000.0).to_f32(), 448.0);
        assert_eq!(Fp8E4M3::from_f32(-1000.0).to_f32(), -448.0);
        assert_eq!(Fp8E4M3::from_bits(0x7E).to_f32(), 448.0);
        assert_eq!(Fp8E4M3::from_bits(0x01).to_f32(), 2f32.powi(-9)); // smallest subnormal
    }

    #[test]
    fn fp8_rounding_goes_to_nearest() {
        let mut rng = XorShift::new(7);
        let values: Vec<f32> = (0..=0x7E).map(|b| Fp8E4M3::from_bits(b).to_f32()).collect();
        for _ in 0..10_000 {
            let x = rng.next_f32() * 500.0;
            let got = Fp8E4M3::from_f32(x).to_f32();
            let clamped = x.clamp(-448.0, 448.0);
            let best = values
                .iter()
                .map(|v| (v - clamped.abs()).abs())
                .fold(f32::INFINITY, f32::min);
            assert!((got.abs() - clamped.abs()).abs() <= best + f32::EPSILON);
        }
    }

    #[test]
    fn bf16_accumulator_stalls() {
        // Adding 1.0 a thousand times: bf16 can hold 256 and 258 but not
        // 257, so 256 + 1 rounds back to 256 forever.
        let ones = vec![1.0f32; 1000];
        assert_eq!(sum_f32(&ones), 1000.0);
        assert_eq!(sum_bf16_accumulator(&ones), 256.0);
    }

    #[test]
    fn kahan_and_pairwise_beat_the_plain_loop() {
        let mut rng = XorShift::new(1);
        let xs: Vec<f32> = (0..1_000_000).map(|_| rng.next_f32().abs()).collect();
        let truth = sum_f64(&xs);
        let err = |s: f32| (f64::from(s) - truth).abs();
        assert!(err(sum_kahan(&xs)) < err(sum_f32(&xs)));
        assert!(err(sum_pairwise(&xs)) < err(sum_f32(&xs)));
    }
}
