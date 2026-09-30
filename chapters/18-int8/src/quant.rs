//! Symmetric int8 quantization in blocks of 64 values.

/// Values per block: 64 int8 values are 64 bytes, one cache line.
pub const BLOCK: usize = 64;

/// 64 quantized values and what is needed to use them.
///
/// The real value `i` is approximately `scale * q[i]`. `sum` (the sum of
/// `q`) lets the integer kernel correct for an offset it applies to the
/// activations (see [`crate::kernels`]).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BlockQ8 {
    pub scale: f32,
    pub sum: i32,
    pub q: [i8; BLOCK],
}

impl BlockQ8 {
    pub const ZERO: Self = Self {
        scale: 0.0,
        sum: 0,
        q: [0; BLOCK],
    };

    /// Quantizes 64 values with the given scale: `q = round(v / scale)`,
    /// clamped to [-127, 127]. A scale of 0 (all values 0) gives zeros.
    pub fn with_scale(values: &[f32], scale: f32) -> Self {
        assert_eq!(values.len(), BLOCK, "a block has 64 values");
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut q = [0i8; BLOCK];
        for (qi, &v) in q.iter_mut().zip(values) {
            *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        Self {
            scale,
            sum: q.iter().map(|&x| i32::from(x)).sum(),
            q,
        }
    }

    /// The block's values, dequantized.
    pub fn dequantize(&self, out: &mut [f32]) {
        for (o, &qi) in out.iter_mut().zip(&self.q) {
            *o = self.scale * f32::from(qi);
        }
    }
}

/// How many values share one scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    /// One scale for the whole matrix.
    PerTensor,
    /// One scale per row (per output channel).
    PerRow,
    /// One scale per block of 64 values.
    PerBlock,
}

/// The scale that maps the largest magnitude in `values` to 127.
pub fn absmax_scale(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0
}

/// Quantizes a `rows × cols` matrix (`cols` a multiple of 64) into blocks,
/// row by row, with scales chosen at the given granularity.
pub fn quantize(
    values: &[f32],
    rows: usize,
    cols: usize,
    granularity: Granularity,
) -> Vec<BlockQ8> {
    assert_eq!(values.len(), rows * cols, "matrix has the wrong size");
    assert!(
        cols.is_multiple_of(BLOCK),
        "columns must be a multiple of 64"
    );
    let tensor_scale = absmax_scale(values);
    let mut blocks = Vec::with_capacity(rows * cols / BLOCK);
    for row in values.chunks_exact(cols) {
        let row_scale = absmax_scale(row);
        for chunk in row.chunks_exact(BLOCK) {
            let scale = match granularity {
                Granularity::PerTensor => tensor_scale,
                Granularity::PerRow => row_scale,
                Granularity::PerBlock => absmax_scale(chunk),
            };
            blocks.push(BlockQ8::with_scale(chunk, scale));
        }
    }
    blocks
}

/// Quantizes activation rows (`x` is `m × k`) into `out`, one scale per
/// block or one per row (token).
pub fn quantize_activations(x: &[f32], k: usize, per_token: bool, out: &mut Vec<BlockQ8>) {
    out.clear();
    for row in x.chunks_exact(k) {
        let row_scale = absmax_scale(row);
        for chunk in row.chunks_exact(BLOCK) {
            let scale = if per_token {
                row_scale
            } else {
                absmax_scale(chunk)
            };
            out.push(BlockQ8::with_scale(chunk, scale));
        }
    }
}

/// `‖a − b‖ / ‖a‖`: the size of the error relative to the size of `a`.
pub fn relative_error(a: &[f32], b: &[f32]) -> f64 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        err += f64::from(x - y).powi(2);
        norm += f64::from(x).powi(2);
    }
    (err / norm).sqrt()
}

/// Dequantizes a whole matrix of blocks.
pub fn dequantize(blocks: &[BlockQ8]) -> Vec<f32> {
    let mut out = vec![0.0; blocks.len() * BLOCK];
    for (b, o) in blocks.iter().zip(out.chunks_exact_mut(BLOCK)) {
        b.dequantize(o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_error_is_at_most_half_a_step() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let b = BlockQ8::with_scale(&values, absmax_scale(&values));
        let mut back = [0.0; 64];
        b.dequantize(&mut back);
        for (v, w) in values.iter().zip(back) {
            assert!((v - w).abs() <= b.scale / 2.0 + 1e-6);
        }
        assert_eq!(b.q.iter().map(|&x| x.unsigned_abs()).max(), Some(127));
        assert_eq!(b.sum, b.q.iter().map(|&x| i32::from(x)).sum::<i32>());
    }

    #[test]
    fn finer_granularity_never_increases_the_error_here() {
        // Rows with very different magnitudes, and one outlier.
        let mut values: Vec<f32> = (0..4 * 128)
            .map(|i| ((i * 7919) % 101) as f32 / 100.0 - 0.5)
            .collect();
        for v in &mut values[128..256] {
            *v *= 0.01;
        }
        values[300] = 40.0;
        let err = |g| relative_error(&values, &dequantize(&quantize(&values, 4, 128, g)));
        let (t, r, b) = (
            err(Granularity::PerTensor),
            err(Granularity::PerRow),
            err(Granularity::PerBlock),
        );
        assert!(t > r && r > b, "{t} {r} {b}");
    }

    #[test]
    fn a_zero_block_stays_zero() {
        let b = BlockQ8::with_scale(&[0.0; 64], 0.0);
        assert_eq!(b, BlockQ8::ZERO);
    }
}
