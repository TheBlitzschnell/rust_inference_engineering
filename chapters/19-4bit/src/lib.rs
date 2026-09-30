//! Chapter 19: 4-bit weights.
//!
//! - [`quant`]: blocks of 64 weights as packed 4-bit codes with a scale and
//!   an offset; symmetric, symmetric with a scale search, and min-max
//!   schemes; scales shared by one block, several blocks or a whole row.
//! - [`kernels`]: 4-bit weights against int8 activations (AVX-512 VNNI,
//!   AVX2, portable) and against `f32` activations.
//! - [`Q4Matrix`]: a `Matrix` for chapter 14's engine.
//! - [`Recorder`]: measures how large each input channel's activations are
//!   while a model runs sample text, for importance-weighted quantization.
//!
//! Quality is measured with chapter 18's `eval`.

use ch07_threads::SpinPool;
use ch14_kv_cache::Matrix;
use ch18_int8::{BLOCK, BlockQ8, matmul_blocks, quantize_activations};
use std::sync::Mutex;

pub mod kernels;
pub mod quant;

pub use kernels::{dot_q4_f32, dot_q4_q8};
pub use quant::{BlockQ4, Scheme, quantize, quantize_weighted};

/// A 4-bit weight matrix: rows of 64-value blocks.
pub struct Q4Matrix {
    blocks: Vec<BlockQ4>,
    rows: usize,
    cols: usize,
    /// Quantize activations to int8 (W4A8) or keep them in `f32` (W4A32).
    int8_activations: bool,
}

impl Q4Matrix {
    /// Quantizes `rows × cols` values; `group` values share a scale.
    pub fn new(
        values: &[f32],
        rows: usize,
        cols: usize,
        group: usize,
        scheme: Scheme,
        int8_activations: bool,
    ) -> Self {
        Self {
            blocks: quantize(values, rows, cols, group, scheme),
            rows,
            cols,
            int8_activations,
        }
    }

    /// Like [`Q4Matrix::new`], with the error weighted per column by
    /// `importance` (see [`Recorder`]).
    pub fn new_weighted(
        values: &[f32],
        rows: usize,
        cols: usize,
        group: usize,
        scheme: Scheme,
        importance: &[f32],
        int8_activations: bool,
    ) -> Self {
        Self {
            blocks: quantize_weighted(values, rows, cols, group, scheme, importance),
            rows,
            cols,
            int8_activations,
        }
    }

    pub fn blocks(&self) -> &[BlockQ4] {
        &self.blocks
    }
}

/// Wraps a matrix and records, for each of its input columns, the sum of
/// the squared activations it was multiplied by. Run a model built from
/// recorders on sample text, then read [`Recorder::importance`].
pub struct Recorder<W> {
    pub inner: W,
    /// (sum of x² per column, number of rows seen).
    stats: Mutex<(Vec<f64>, u64)>,
}

impl<W: Matrix> Recorder<W> {
    pub fn new(inner: W) -> Self {
        let cols = inner.cols();
        Self {
            inner,
            stats: Mutex::new((vec![0.0; cols], 0)),
        }
    }

    /// Mean of x² per input column over everything recorded. A column fed
    /// by large activations matters more: its weights' errors are
    /// multiplied by those activations.
    pub fn importance(&self) -> Vec<f32> {
        let (sums, rows) = &*self.stats.lock().expect("stats lock");
        let n = (*rows).max(1) as f64;
        sums.iter().map(|&s| (s / n) as f32).collect()
    }
}

impl<W: Matrix> Matrix for Recorder<W> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn bytes(&self) -> usize {
        self.inner.bytes()
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        self.inner.row_to_f32(r, out);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        {
            let (sums, rows) = &mut *self.stats.lock().expect("stats lock");
            for row in x.chunks_exact(self.inner.cols()) {
                for (s, &v) in sums.iter_mut().zip(row) {
                    *s += f64::from(v) * f64::from(v);
                }
            }
            *rows += m as u64;
        }
        self.inner.matmul(pool, x, y, m, scratch);
    }
}

impl Matrix for Q4Matrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn bytes(&self) -> usize {
        self.blocks.len() * size_of::<BlockQ4>()
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        let per_row = self.cols / BLOCK;
        for (b, o) in self.blocks[r * per_row..(r + 1) * per_row]
            .iter()
            .zip(out.chunks_exact_mut(BLOCK))
        {
            b.dequantize(o);
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
        if self.int8_activations {
            let mut xq: Vec<BlockQ8> = Vec::with_capacity(x.len() / BLOCK);
            quantize_activations(x, self.cols, false, &mut xq);
            matmul_blocks(pool, &xq, &self.blocks, y, m, self.rows, scratch, dot_q4_q8);
        } else {
            matmul_blocks(pool, x, &self.blocks, y, m, self.rows, scratch, dot_q4_f32);
        }
    }
}
