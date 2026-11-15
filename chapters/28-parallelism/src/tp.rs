//! Tensor parallelism: every layer split across devices (Megatron-LM's
//! scheme).
//!
//! With `n` devices, device `r` holds:
//!
//! - **column-parallel** slices (`wq`, `wk`, `wv`, `w_gate`, `w_up`): the
//!   rows of the output features it computes (its attention heads, its
//!   share of the MLP's hidden units). No communication is needed: each
//!   device computes its own part of the output from the full input.
//! - **row-parallel** slices (`wo`, `w_down`): the columns matching those
//!   features. Each device's product is a partial sum of the full output;
//!   an **all-reduce** adds them up. Two all-reduces per layer.
//! - the embedding, norms and RoPE in full (they are small);
//! - a slice of the LM head's rows (vocabulary-parallel), whose logits an
//!   **all-gather** puts together.
//!
//! Each device's share is an ordinary `Model` with fewer heads and a
//! narrower MLP in its config, and chapter 14's forward pass runs on it
//! unchanged: the communication hides inside the [`Tp`] matrix type.

use crate::comm::Comm;
use ch07_threads::SpinPool;
use ch14_kv_cache::{Config, Layer, Matrix, Model};
use std::ops::Range;

/// What a matrix does after its local product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum After {
    /// Nothing: a column-parallel slice, or a replicated matrix.
    Nothing,
    /// Add the partial outputs of every device: a row-parallel slice.
    AllReduce,
    /// Concatenate every device's output columns: the vocabulary-parallel
    /// LM head.
    AllGather,
}

/// One device's slice of a weight matrix, and the communication that
/// completes its output.
pub struct Tp<'c, W> {
    inner: W,
    after: After,
    comm: &'c Comm,
    /// The full matrix's rows (outputs): for `AllGather`, the whole
    /// vocabulary; otherwise the slice's own rows.
    rows: usize,
}

impl<W: Matrix> Matrix for Tp<'_, W> {
    fn rows(&self) -> usize {
        self.rows
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
        match self.after {
            After::Nothing => self.inner.matmul(pool, x, y, m, scratch),
            After::AllReduce => {
                self.inner.matmul(pool, x, y, m, scratch);
                self.comm.all_reduce(&mut y[..m * self.rows]);
            }
            After::AllGather => {
                let mine = self.inner.rows();
                let mut local = vec![0.0; m * mine];
                self.inner.matmul(pool, x, &mut local, m, scratch);
                // Every device's `[m × its rows]` block, put side by side
                // in each row of `y`.
                let pieces = self.comm.all_gather(&local);
                let mut col = 0;
                for piece in &pieces {
                    let width = piece.len() / m;
                    for r in 0..m {
                        y[r * self.rows + col..][..width]
                            .copy_from_slice(&piece[r * width..][..width]);
                    }
                    col += width;
                }
            }
        }
    }
}

/// A matrix as plain `f32` values (any `Matrix`, through `row_to_f32`).
pub fn to_f32(w: &impl Matrix) -> Vec<f32> {
    let mut out = vec![0.0; w.rows() * w.cols()];
    for (r, row) in out.chunks_exact_mut(w.cols()).enumerate() {
        w.row_to_f32(r, row);
    }
    out
}

/// The rows `rows` of a `[_ × cols]` matrix, all columns.
fn rows_of(values: &[f32], cols: usize, rows: Range<usize>) -> Vec<f32> {
    values[rows.start * cols..rows.end * cols].to_vec()
}

/// The columns `cols` of every row of a `[rows × width]` matrix.
fn cols_of(values: &[f32], width: usize, cols: Range<usize>) -> Vec<f32> {
    values
        .chunks_exact(width)
        .flat_map(|row| row[cols.clone()].iter().copied())
        .collect()
}

/// The `i`-th of `n` equal parts of `0..len`.
fn part(len: usize, i: usize, n: usize) -> Range<usize> {
    i * len / n..(i + 1) * len / n
}

/// Device `comm.rank()`'s share of `model` for tensor parallelism over
/// `comm.size()` devices. `make` builds a matrix of whatever type from
/// `f32` values, rows and columns. The number of KV heads and the MLP's
/// width must divide evenly.
pub fn shard<'c, W: Matrix, V: Matrix>(
    model: &Model<W>,
    comm: &'c Comm,
    make: impl Fn(&[f32], usize, usize) -> V,
) -> Model<Tp<'c, V>> {
    let (r, n) = (comm.rank(), comm.size());
    let c = &model.config;
    assert_eq!(c.num_kv_heads % n, 0, "KV heads must split evenly");
    assert_eq!(c.intermediate_size % n, 0, "the MLP must split evenly");
    let (h, d) = (c.hidden_size, c.head_dim);
    let q = part(c.num_heads, r, n);
    let kv = part(c.num_kv_heads, r, n);
    let mlp = part(c.intermediate_size, r, n);
    let (q_rows, kv_rows) = (q.start * d..q.end * d, kv.start * d..kv.end * d);

    let tp = |values: Vec<f32>, rows: usize, cols: usize, after: After, full_rows: usize| Tp {
        inner: make(&values, rows, cols),
        after,
        comm,
        rows: full_rows,
    };
    // A column-parallel slice: some rows, every column.
    let column = |w: &W, rows: Range<usize>| {
        let n_rows = rows.len();
        tp(
            rows_of(&to_f32(w), w.cols(), rows),
            n_rows,
            w.cols(),
            After::Nothing,
            n_rows,
        )
    };
    // A row-parallel slice: every row, some columns; outputs all-reduced.
    let row = |w: &W, cols: Range<usize>| {
        let n_cols = cols.len();
        tp(
            cols_of(&to_f32(w), w.cols(), cols),
            w.rows(),
            n_cols,
            After::AllReduce,
            w.rows(),
        )
    };
    let layers = model
        .layers
        .iter()
        .map(|l| Layer {
            attn_norm: l.attn_norm.clone(),
            wq: column(&l.wq, q_rows.clone()),
            wk: column(&l.wk, kv_rows.clone()),
            wv: column(&l.wv, kv_rows.clone()),
            wo: row(&l.wo, q_rows.clone()),
            mlp_norm: l.mlp_norm.clone(),
            w_gate: column(&l.w_gate, mlp.clone()),
            w_up: column(&l.w_up, mlp.clone()),
            w_down: row(&l.w_down, mlp.clone()),
        })
        .collect();
    let embed = &model.embed;
    let vocab = part(c.vocab_size, r, n);
    let head = model.lm_head();
    let lm_head = tp(
        rows_of(&to_f32(head), h, vocab.clone()),
        vocab.len(),
        h,
        After::AllGather,
        c.vocab_size,
    );
    let config = Config {
        num_heads: q.len(),
        num_kv_heads: kv.len(),
        intermediate_size: mlp.len(),
        tie_embeddings: false,
        ..c.clone()
    };
    Model::new(
        config,
        tp(
            to_f32(embed),
            embed.rows(),
            embed.cols(),
            After::Nothing,
            embed.rows(),
        ),
        layers,
        model.final_norm.clone(),
        Some(lm_head),
    )
}
