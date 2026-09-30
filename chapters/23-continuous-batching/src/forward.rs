//! One forward pass for a batch of sequences.
//!
//! Each sequence brings some tokens (one when decoding, a chunk of its
//! prompt when prefilling) and its own KV cache. All their tokens are
//! stacked into one `[M × hidden]` matrix, so each weight matrix is read
//! once per step for the whole batch, exactly as chapter 14 does for the
//! tokens of one prompt. Only attention is per sequence: each sequence
//! attends to its own cache.

use ch07_threads::SpinPool;
use ch08_operators::{add_inplace, rms_norm, swiglu};
use ch14_kv_cache::{AttentionInput, Config, KvCache, Matrix, Model};
use ch20_flash_attention::{FlashOptions, flash_attention, flash_decode_many};
use std::ops::Range;

/// One sequence's part of a step.
pub struct BatchSeq<'a> {
    /// The tokens to process: the next chunk of its prompt, or the token it
    /// sampled last.
    pub tokens: &'a [u32],
    /// Its cache. The tokens continue the sequence from `cache.len()`.
    pub cache: &'a mut KvCache,
    /// Whether to compute logits for its last token. A prompt chunk that is
    /// not the prompt's last needs none.
    pub logits: bool,
}

/// Buffers for up to `max_tokens` tokens per step and `max_logits`
/// sequences with logits, allocated once.
pub struct BatchScratch {
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

impl BatchScratch {
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

/// Runs one step for every sequence in `seqs` and advances their caches.
/// Returns the logits of each sequence with `logits: true`, in order, as
/// `[n × vocab]`.
pub fn forward_batch<'s, W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    seqs: &mut [BatchSeq<'_>],
    s: &'s mut BatchScratch,
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
        assert!(seq.cache.len() + n <= seq.cache.capacity(), "KV cache full");
        rows.push(m..m + n);
        m += n;
    }
    assert!(m <= s.max_tokens, "{m} tokens exceed the step's scratch");
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
        layer(model, pool, l, seqs, &rows, m, s, attention);
    }
    for (seq, rows) in seqs.iter_mut().zip(&rows) {
        seq.cache.advance(rows.len());
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
    l: usize,
    seqs: &mut [BatchSeq<'_>],
    rows: &[Range<usize>],
    m: usize,
    s: &mut BatchScratch,
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
    // ...then, per sequence: its positions and its cache...
    for (seq, rows) in seqs.iter_mut().zip(rows) {
        let start = seq.cache.len();
        for (t, r) in rows.clone().enumerate() {
            let pos = start + t;
            let (qr, kr) = (r * q_dim..(r + 1) * q_dim, r * kv_dim..(r + 1) * kv_dim);
            model.rope().apply_heads(&mut s.q[qr], pos);
            model.rope().apply_heads(&mut s.k[kr.clone()], pos);
            seq.cache.store(l, pos, &s.k[kr.clone()], &s.v[kr]);
        }
    }
    // ...and its attention. Decoding sequences (one token each) go to the
    // pool together, in one parallel pass; prompt chunks one at a time.
    let mut decoding = Vec::new();
    let mut decoding_rows = Vec::new();
    for (seq, rows) in seqs.iter().zip(rows) {
        let q_rows = rows.start * q_dim..rows.end * q_dim;
        let input = AttentionInput {
            layer: l,
            start: seq.cache.len(),
            m: rows.len(),
            q: &s.q[q_rows.clone()],
            cache: seq.cache,
            config: c,
        };
        if rows.len() == 1 {
            decoding.push(input);
            decoding_rows.push(rows.start);
        } else {
            flash_attention(pool, &input, &mut s.attn[q_rows], attention);
        }
    }
    s.decoded.resize(decoding.len() * q_dim, 0.0);
    flash_decode_many(pool, &decoding, &mut s.decoded, attention, &mut s.states);
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
