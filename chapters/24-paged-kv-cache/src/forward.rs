//! Chapter 23's batched forward pass, over the paged cache.
//!
//! Three things change: a sequence brings a block table instead of its own
//! `KvCache`; keys and values are stored into the shared `BlockPool`
//! through that table; attention reads them back block by block
//! ([`crate::attention`]). Everything else (the stacked matrix products,
//! the per-sequence positions, the logits of each sequence's last token)
//! is chapter 23's code.

use crate::attention::{PagedInput, paged_chunk, paged_decode_many};
use crate::blocks::{BlockPool, BlockTable};
use ch07_threads::SpinPool;
use ch08_operators::{add_inplace, rms_norm, swiglu};
use ch14_kv_cache::{Config, Matrix, Model};
use ch20_flash_attention::{FlashOptions, MAX_TILE};
use std::ops::Range;

/// One sequence's part of a step.
pub struct PagedSeq<'a> {
    /// The tokens to process: the next chunk of its prompt, or the token it
    /// sampled last.
    pub tokens: &'a [u32],
    /// Its blocks. The tokens continue the sequence from `table.len`; the
    /// table must already have blocks for them.
    pub table: &'a mut BlockTable,
    /// Whether to compute logits for its last token. A prompt chunk that is
    /// not the prompt's last needs none.
    pub logits: bool,
}

/// Buffers for up to `max_tokens` tokens per step and `max_logits`
/// sequences with logits, allocated once.
pub struct PagedScratch {
    max_tokens: usize,
    max_logits: usize,
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    /// The final hidden state of each sequence that needs logits.
    last: Vec<f32>,
    logits: Vec<f32>,
    matmul: Vec<f32>,
    /// Attention outputs and partial states of the decoding sequences.
    decoded: Vec<f32>,
    states: Vec<f32>,
}

impl PagedScratch {
    pub fn new(config: &Config, max_tokens: usize, max_logits: usize) -> Self {
        let (c, n) = (config, max_tokens.max(1));
        Self {
            max_tokens: n,
            max_logits,
            x: vec![0.0; n * c.hidden_size],
            normed: vec![0.0; n * c.hidden_size],
            q: vec![0.0; n * c.q_dim()],
            k: vec![0.0; n * c.kv_dim()],
            v: vec![0.0; n * c.kv_dim()],
            attn: vec![0.0; n * c.q_dim()],
            proj: vec![0.0; n * c.hidden_size],
            gate: vec![0.0; n * c.intermediate_size],
            up: vec![0.0; n * c.intermediate_size],
            act: vec![0.0; n * c.intermediate_size],
            last: vec![0.0; max_logits * c.hidden_size],
            logits: vec![0.0; max_logits * c.vocab_size],
            matmul: Vec::new(),
            decoded: Vec::new(),
            states: Vec::new(),
        }
    }
}

/// Runs one step for every sequence in `seqs`, storing keys and values in
/// `kv`, and advances their tables. Returns the logits of each sequence
/// with `logits: true`, in order, as `[n × vocab]`.
pub fn forward_paged<'s, W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    kv: &mut BlockPool,
    seqs: &mut [PagedSeq<'_>],
    s: &'s mut PagedScratch,
    attention: &FlashOptions,
) -> &'s [f32] {
    let c = &model.config;
    let h = c.hidden_size;
    // Where each sequence's tokens sit in the stacked `[M × ...]` buffers.
    let mut rows: Vec<Range<usize>> = Vec::with_capacity(seqs.len());
    let mut m = 0;
    for seq in seqs.iter() {
        let n = seq.tokens.len();
        assert!(n > 0, "a sequence with no tokens in the step");
        assert!(
            seq.table.len + n <= seq.table.capacity(kv.block_size()),
            "the block table has no room for the step"
        );
        rows.push(m..m + n);
        m += n;
    }
    assert!(m <= s.max_tokens, "{m} tokens exceed the step's scratch");
    assert!(kv.block_size() <= MAX_TILE, "blocks larger than a tile");
    let wanted = seqs.iter().filter(|q| q.logits).count();
    assert!(wanted <= s.max_logits, "too many sequences want logits");

    let mut r = 0;
    for seq in seqs.iter() {
        for &token in seq.tokens {
            model
                .embed
                .row_to_f32(token as usize, &mut s.x[r * h..(r + 1) * h]);
            r += 1;
        }
    }
    for l in 0..c.num_layers {
        layer(model, pool, kv, l, seqs, &rows, m, s, attention);
    }
    for (seq, rows) in seqs.iter_mut().zip(&rows) {
        seq.table.len += rows.len();
    }

    // Final norm and LM head, only for the last token of each sequence that
    // needs logits: those rows are gathered first, so the LM head (the
    // largest matrix) runs once for all of them.
    let mut n = 0;
    for (seq, rows) in seqs.iter().zip(&rows) {
        if seq.logits {
            let last = rows.end - 1;
            rms_norm(
                &s.x[last * h..(last + 1) * h],
                &model.final_norm,
                c.rms_norm_eps,
                &mut s.last[n * h..(n + 1) * h],
            );
            n += 1;
        }
    }
    let vocab = c.vocab_size;
    if n > 0 {
        model.lm_head().matmul(
            pool,
            &s.last[..n * h],
            &mut s.logits[..n * vocab],
            n,
            &mut s.matmul,
        );
    }
    &s.logits[..n * vocab]
}

/// One transformer layer for the whole batch.
#[expect(
    clippy::too_many_arguments,
    reason = "the pieces of one layer step, passed separately so the borrows stay visible"
)]
fn layer<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    kv: &mut BlockPool,
    l: usize,
    seqs: &mut [PagedSeq<'_>],
    rows: &[Range<usize>],
    m: usize,
    s: &mut PagedScratch,
    attention: &FlashOptions,
) {
    let c = &model.config;
    let (h, q_dim, kv_dim, inter) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size);
    let layer = &model.layers[l];

    // Attention block: the projections for all M rows at once...
    norm_rows(
        &s.x[..m * h],
        &layer.attn_norm,
        c.rms_norm_eps,
        &mut s.normed[..m * h],
    );
    W::matmul_many(
        pool,
        &[&layer.wq, &layer.wk, &layer.wv],
        &s.normed[..m * h],
        &mut [
            &mut s.q[..m * q_dim],
            &mut s.k[..m * kv_dim],
            &mut s.v[..m * kv_dim],
        ],
        m,
        &mut s.matmul,
    );
    // ...then, per sequence: its positions, and its blocks...
    for (seq, rows) in seqs.iter_mut().zip(rows) {
        let start = seq.table.len;
        for (t, r) in rows.clone().enumerate() {
            let pos = start + t;
            let (qr, kr) = (r * q_dim..(r + 1) * q_dim, r * kv_dim..(r + 1) * kv_dim);
            model.rope().apply_heads(&mut s.q[qr], pos);
            model.rope().apply_heads(&mut s.k[kr.clone()], pos);
            seq.table.store(kv, pos, l, &s.k[kr.clone()], &s.v[kr]);
        }
    }
    // ...and its attention. Decoding sequences (one token each) go to the
    // pool together, in one parallel pass; prompt chunks one at a time.
    let kv: &BlockPool = kv;
    let mut decoding = Vec::new();
    let mut decoding_rows = Vec::new();
    for (seq, rows) in seqs.iter().zip(rows) {
        let q_rows = rows.start * q_dim..rows.end * q_dim;
        let input = PagedInput {
            table: seq.table,
            start: seq.table.len,
            m: rows.len(),
            q: &s.q[q_rows.clone()],
        };
        if rows.len() == 1 {
            decoding.push(input);
            decoding_rows.push(rows.start);
        } else {
            paged_chunk(
                pool,
                kv,
                c,
                l,
                &input,
                &mut s.attn[q_rows],
                attention,
                &mut s.states,
            );
        }
    }
    s.decoded.resize(decoding.len() * q_dim, 0.0);
    paged_decode_many(
        pool,
        kv,
        c,
        l,
        &decoding,
        &mut s.decoded,
        attention,
        &mut s.states,
    );
    for (&r, out) in decoding_rows.iter().zip(s.decoded.chunks_exact(q_dim)) {
        s.attn[r * q_dim..(r + 1) * q_dim].copy_from_slice(out);
    }
    layer.wo.matmul(
        pool,
        &s.attn[..m * q_dim],
        &mut s.proj[..m * h],
        m,
        &mut s.matmul,
    );
    add_inplace(&mut s.x[..m * h], &s.proj[..m * h]);

    // MLP block: no sequence boundaries at all.
    norm_rows(
        &s.x[..m * h],
        &layer.mlp_norm,
        c.rms_norm_eps,
        &mut s.normed[..m * h],
    );
    W::matmul_many(
        pool,
        &[&layer.w_gate, &layer.w_up],
        &s.normed[..m * h],
        &mut [&mut s.gate[..m * inter], &mut s.up[..m * inter]],
        m,
        &mut s.matmul,
    );
    swiglu(
        &s.gate[..m * inter],
        &s.up[..m * inter],
        &mut s.act[..m * inter],
    );
    layer.w_down.matmul(
        pool,
        &s.act[..m * inter],
        &mut s.proj[..m * h],
        m,
        &mut s.matmul,
    );
    add_inplace(&mut s.x[..m * h], &s.proj[..m * h]);
}

/// RMSNorm applied to each `hidden`-sized row.
fn norm_rows(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let h = weight.len();
    for (xi, oi) in x.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
        rms_norm(xi, weight, eps, oi);
    }
}
