//! Chapter 18: int8 quantization.
//!
//! - [`quant`]: blocks of 64 int8 values with a scale; quantizing weights
//!   (per tensor, per row or per block) and activations (per block or per
//!   token).
//! - [`kernels`]: dot products with int8 weights and `f32` activations, and
//!   with both in int8 (AVX-512 VNNI, AVX2, portable).
//! - [`Q8Matrix`]: a `Matrix` for chapter 14's engine.
//! - [`eval`]: perplexity, KL divergence and top-1 agreement against a
//!   reference model, to measure what quantization costs.

use ch07_threads::SpinPool;
use ch14_kv_cache::Matrix;

pub mod eval;
pub mod kernels;
pub mod matmul;
pub mod quant;

pub use kernels::{Kernels, dot_q8_f32, dot_q8_q8};
pub use matmul::matmul_blocks;
pub use quant::{BLOCK, BlockQ8, Granularity, quantize, quantize_activations};

/// What the activations are when they meet the int8 weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activations {
    /// Left in `f32` ("W8A32", weight-only quantization).
    Float,
    /// Quantized to int8 on the fly, one scale per block of 64 ("W8A8").
    Int8PerBlock,
    /// Quantized to int8 on the fly, one scale per token (per row).
    Int8PerToken,
}

/// An int8 weight matrix: rows of 64-value blocks.
pub struct Q8Matrix {
    blocks: Vec<BlockQ8>,
    rows: usize,
    cols: usize,
    activations: Activations,
}

impl Q8Matrix {
    /// Quantizes `rows × cols` values (`cols` a multiple of 64).
    pub fn new(
        values: &[f32],
        rows: usize,
        cols: usize,
        granularity: Granularity,
        activations: Activations,
    ) -> Self {
        Self {
            blocks: quantize(values, rows, cols, granularity),
            rows,
            cols,
            activations,
        }
    }

    pub fn blocks(&self) -> &[BlockQ8] {
        &self.blocks
    }
}

impl Matrix for Q8Matrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn bytes(&self) -> usize {
        self.blocks.len() * size_of::<BlockQ8>()
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
        let n = self.rows;
        match self.activations {
            Activations::Float => {
                matmul_blocks(pool, x, &self.blocks, y, m, n, scratch, dot_q8_f32);
            }
            Activations::Int8PerBlock | Activations::Int8PerToken => {
                let xq = self.quantize_input(x);
                matmul_blocks(pool, &xq, &self.blocks, y, m, n, scratch, dot_q8_q8);
            }
        }
    }
}

impl Q8Matrix {
    /// Quantizes activations for this matrix's mode. A few kilobytes per
    /// call, small next to the matrix itself.
    fn quantize_input(&self, x: &[f32]) -> Vec<BlockQ8> {
        let mut xq = Vec::with_capacity(x.len() / BLOCK);
        let per_token = self.activations == Activations::Int8PerToken;
        quantize_activations(x, self.cols, per_token, &mut xq);
        xq
    }
}
