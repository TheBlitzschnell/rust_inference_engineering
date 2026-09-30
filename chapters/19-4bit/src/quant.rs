//! 4-bit quantization: 64 values per block, two values per byte.

use ch18_int8::BLOCK;

/// 64 weights as 4-bit codes. Value `i` is `scale * code(i) + min`, with
/// codes 0..=15. Byte `j` holds value `j` in its low nibble and value
/// `j + 32` in its high nibble, so unpacking gives values 0..32 from the
/// low nibbles and 32..64 from the high ones, in order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BlockQ4 {
    pub scale: f32,
    pub min: f32,
    pub packed: [u8; BLOCK / 2],
}

impl BlockQ4 {
    /// The code of value `i` (0..=15).
    pub fn code(&self, i: usize) -> u8 {
        let byte = self.packed[i % 32];
        if i < 32 { byte & 0x0F } else { byte >> 4 }
    }

    pub fn dequantize(&self, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate().take(BLOCK) {
            *o = self.scale * f32::from(self.code(i)) + self.min;
        }
    }

    /// Packs codes (each 0..=15) with the given scale and offset.
    pub fn pack(codes: &[u8; BLOCK], scale: f32, min: f32) -> Self {
        let mut packed = [0u8; BLOCK / 2];
        for (j, p) in packed.iter_mut().enumerate() {
            *p = (codes[j] & 0x0F) | (codes[j + 32] << 4);
        }
        Self { scale, min, packed }
    }
}

/// How codes are assigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// Symmetric around zero: `scale = max|w| / 7`, codes 1..=15 stand for
    /// -7..=7 (`min = -8 · scale`).
    Symmetric,
    /// Symmetric, but the scale is chosen by trying slightly smaller ones
    /// and keeping the one with the smallest (weighted) squared error:
    /// clipping the largest values a little makes every other step finer.
    SymmetricSearch,
    /// Asymmetric: 16 levels from the group's minimum to its maximum
    /// (`scale = (max − min) / 15`).
    MinMax,
    /// Asymmetric, trying ranges slightly inside the minimum and maximum
    /// and keeping the one with the smallest (weighted) squared error.
    MinMaxSearch,
}

/// Quantizes the values that share one scale (a multiple of 64 values)
/// into blocks. `importance[i]` weighs the squared error of value `i` in
/// the searches (all 1 when `None`).
fn quantize_group(
    values: &[f32],
    importance: Option<&[f32]>,
    scheme: Scheme,
    out: &mut Vec<BlockQ4>,
) {
    let error = |c: &(f32, f32)| weighted_error(values, importance, *c);
    let best = |candidates: Vec<(f32, f32)>| {
        candidates
            .into_iter()
            .min_by(|a, b| error(a).total_cmp(&error(b)))
            .expect("at least one candidate")
    };
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (scale, min) = match scheme {
        Scheme::Symmetric => symmetric(values, 1.0),
        // Candidate scales from the absmax scale down to 70% of it.
        Scheme::SymmetricSearch => best(
            (0..=15u8)
                .map(|i| symmetric(values, 1.0 - 0.02 * f32::from(i)))
                .collect(),
        ),
        Scheme::MinMax => ((hi - lo) / 15.0, lo),
        // Move each end inwards by 0-10% of the range, independently.
        Scheme::MinMaxSearch => {
            let range = hi - lo;
            let mut candidates = Vec::with_capacity(36);
            for a in 0..=5u8 {
                for b in 0..=5u8 {
                    let (l, h) = (
                        lo + 0.02 * f32::from(a) * range,
                        hi - 0.02 * f32::from(b) * range,
                    );
                    candidates.push(((h - l) / 15.0, l));
                }
            }
            best(candidates)
        }
    };
    for chunk in values.chunks_exact(BLOCK) {
        let mut codes = [0u8; BLOCK];
        for (c, &v) in codes.iter_mut().zip(chunk) {
            *c = code(v, scale, min);
        }
        out.push(BlockQ4::pack(&codes, scale, min));
    }
}

/// `(scale, min)` for the symmetric scheme with the scale shrunk by
/// `factor` (1.0 = absmax).
fn symmetric(values: &[f32], factor: f32) -> (f32, f32) {
    let absmax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = absmax * factor / 7.0;
    (scale, -8.0 * scale)
}

fn code(v: f32, scale: f32, min: f32) -> u8 {
    if scale > 0.0 {
        ((v - min) / scale).round().clamp(0.0, 15.0) as u8
    } else {
        0
    }
}

fn weighted_error(values: &[f32], importance: Option<&[f32]>, (scale, min): (f32, f32)) -> f32 {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let back = scale * f32::from(code(v, scale, min)) + min;
            importance.map_or(1.0, |imp| imp[i]) * (v - back) * (v - back)
        })
        .sum()
}

/// Quantizes a `rows × cols` matrix. `group` values share a scale: 64 (one
/// block), a multiple of 64 that divides `cols`, or `cols` (one per row).
pub fn quantize(
    values: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
    scheme: Scheme,
) -> Vec<BlockQ4> {
    quantize_impl(values, rows, cols, group, scheme, None)
}

/// Like [`quantize`], but the searches minimize the error weighted by
/// `importance[c]` for column `c`: typically the mean square of the
/// activations that multiply that column, measured on sample text.
pub fn quantize_weighted(
    values: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
    scheme: Scheme,
    importance: &[f32],
) -> Vec<BlockQ4> {
    assert_eq!(importance.len(), cols, "one importance per column");
    quantize_impl(values, rows, cols, group, scheme, Some(importance))
}

fn quantize_impl(
    values: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
    scheme: Scheme,
    importance: Option<&[f32]>,
) -> Vec<BlockQ4> {
    assert_eq!(values.len(), rows * cols, "matrix has the wrong size");
    assert!(
        group.is_multiple_of(BLOCK) && cols.is_multiple_of(group),
        "groups must be whole blocks and divide the rows"
    );
    let mut blocks = Vec::with_capacity(rows * cols / BLOCK);
    for row in values.chunks_exact(cols) {
        for (g, chunk) in row.chunks_exact(group).enumerate() {
            let imp = importance.map(|imp| &imp[g * group..(g + 1) * group]);
            quantize_group(chunk, imp, scheme, &mut blocks);
        }
    }
    blocks
}

pub fn dequantize(blocks: &[BlockQ4]) -> Vec<f32> {
    let mut out = vec![0.0; blocks.len() * BLOCK];
    for (b, o) in blocks.iter().zip(out.chunks_exact_mut(BLOCK)) {
        b.dequantize(o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch18_int8::quant::relative_error;

    fn sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * (1.0 + (i % 13) as f32))
            .collect()
    }

    #[test]
    fn packing_round_trips_every_code() {
        let codes: [u8; 64] = std::array::from_fn(|i| (i * 5 % 16) as u8);
        let b = BlockQ4::pack(&codes, 1.0, 0.0);
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(b.code(i), c);
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "code 8 must reconstruct exactly zero: 8 · scale − 8 · scale"
    )]
    fn symmetric_zero_is_exact_and_error_is_at_most_half_a_step() {
        let mut v = sample(64);
        v[3] = 0.0;
        let b = &quantize(&v, 1, 64, 64, Scheme::Symmetric)[0];
        let back = dequantize(std::slice::from_ref(b));
        assert_eq!(back[3], 0.0);
        for (x, y) in v.iter().zip(&back) {
            assert!((x - y).abs() <= b.scale / 2.0 + 1e-6);
        }
    }

    #[test]
    fn searches_never_do_worse_than_their_starting_point() {
        let v = sample(64 * 20);
        let err = |s| relative_error(&v, &dequantize(&quantize(&v, 20, 64, 64, s)));
        assert!(err(Scheme::SymmetricSearch) <= err(Scheme::Symmetric));
        assert!(err(Scheme::MinMaxSearch) <= err(Scheme::MinMax));
    }

    #[test]
    fn importance_moves_the_error_to_unimportant_columns() {
        let v = sample(64 * 20);
        let mut imp = vec![1.0f32; 64];
        imp[..8].fill(100.0);
        let plain = dequantize(&quantize(&v, 20, 64, 64, Scheme::MinMaxSearch));
        let weighted = dequantize(&quantize_weighted(
            &v,
            20,
            64,
            64,
            Scheme::MinMaxSearch,
            &imp,
        ));
        // Squared error of the first 8 columns (the important ones).
        let err = |back: &[f32]| -> f32 {
            v.chunks_exact(64)
                .zip(back.chunks_exact(64))
                .map(|(a, b)| {
                    a[..8]
                        .iter()
                        .zip(&b[..8])
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f32>()
                })
                .sum()
        };
        assert!(err(&weighted) <= err(&plain));
    }
}
