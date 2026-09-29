//! Chapter 14: the inference engine, built around a KV cache.
//!
//! Chapter 13's reference model recomputes every position at every step.
//! Because attention is causal, the keys and values of earlier positions
//! never change, so this engine computes them once, stores them in a
//! [`KvCache`], and afterwards processes only the new tokens:
//!
//! - **prefill**: the whole prompt in one call (many tokens, matmul-shaped),
//! - **decode**: one new token per call (matrix-vector-shaped).
//!
//! The same [`Model::forward_last`] does both. The rest of the course builds
//! on this crate: chapter 16 plugs in `bf16` weights through the [`Matrix`]
//! trait, chapters 18-19 quantized ones, chapter 23 batches several
//! sequences, chapter 24 pages the cache.

use ch06_simd::{AlignedVec, dot};
use ch07_threads::SpinPool;
use ch08_operators::{add_inplace, rms_norm, softmax, swiglu};
use ch12_attention::Rope;
pub use ch13_transformer::Config;
use std::time::{Duration, Instant};

pub mod matmul;
pub use matmul::matmul_pooled;

/// A weight matrix of `rows × cols`, stored row by row in some format.
///
/// The model is generic over this trait, so the same forward pass runs
/// `f32`, `bf16` (chapter 16) and quantized weights (chapters 18-19): only
/// the storage and the dot-product kernel change.
pub trait Matrix: Send + Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Bytes of weight data: what one full pass over the matrix reads.
    fn bytes(&self) -> usize;
    /// Row `r` converted to `f32` (used for embedding lookups).
    fn row_to_f32(&self, r: usize, out: &mut [f32]);
    /// `y[m × rows] = x[m × cols] · selfᵀ`, split across the pool's threads.
    /// `scratch` is reusable temporary space.
    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    );

    /// Multiplies the same `x` by several matrices with the same number of
    /// columns: `outputs[i] = x · matrices[i]ᵀ`. The default simply calls
    /// `matmul` for each; chapter 17 overrides it to do all of them in one
    /// parallel pass, which matters when the matrices are small.
    fn matmul_many(
        pool: &mut SpinPool,
        matrices: &[&Self],
        x: &[f32],
        outputs: &mut [&mut [f32]],
        m: usize,
        scratch: &mut Vec<f32>,
    ) where
        Self: Sized,
    {
        for (w, y) in matrices.iter().zip(outputs.iter_mut()) {
            w.matmul(pool, x, y, m, scratch);
        }
    }
}

/// Plain `f32` weights, 64-byte aligned.
pub struct DenseF32 {
    data: AlignedVec<f32>,
    rows: usize,
    cols: usize,
}

impl DenseF32 {
    pub fn values(&self) -> &[f32] {
        &self.data
    }

    pub fn new(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data has the wrong size");
        Self {
            data: AlignedVec::from_slice(data),
            rows,
            cols,
        }
    }
}

impl Matrix for DenseF32 {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn bytes(&self) -> usize {
        self.data.len() * 4
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        out.copy_from_slice(&self.data[r * self.cols..(r + 1) * self.cols]);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        matmul_pooled(
            pool, x, &self.data, y, m, self.cols, self.rows, scratch, dot,
        );
    }
}

/// The key/value cache of one sequence.
///
/// Layout: `[layer][kv_head][position][head_dim]`. Keeping each head's
/// positions contiguous means attention reads one head's keys as a single
/// run of memory.
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Allocates room for `capacity` positions (all layers, keys and values).
    pub fn new(config: &Config, capacity: usize) -> Self {
        let size = config.num_layers * config.num_kv_heads * capacity * config.head_dim;
        Self {
            k: vec![0.0; size],
            v: vec![0.0; size],
            len: 0,
            capacity,
            kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }

    /// Positions currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes allocated for keys and values.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }

    /// Forgets every position, keeping the memory for the next sequence.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Forgets positions `len..`: used to roll back rejected speculative
    /// tokens (chapter 26). The stored numbers stay; they are overwritten
    /// when those positions are used again.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    fn offset(&self, layer: usize, head: usize, pos: usize) -> usize {
        ((layer * self.kv_heads + head) * self.capacity + pos) * self.head_dim
    }

    /// Stores one position's keys and values (all KV heads) for a layer.
    fn store(&mut self, layer: usize, pos: usize, k_row: &[f32], v_row: &[f32]) {
        let d = self.head_dim;
        for head in 0..self.kv_heads {
            let at = self.offset(layer, head, pos);
            self.k[at..at + d].copy_from_slice(&k_row[head * d..(head + 1) * d]);
            self.v[at..at + d].copy_from_slice(&v_row[head * d..(head + 1) * d]);
        }
    }

    /// Keys of one head for positions `0..upto`, as `[upto × head_dim]`.
    fn keys(&self, layer: usize, head: usize, upto: usize) -> &[f32] {
        let at = self.offset(layer, head, 0);
        &self.k[at..at + upto * self.head_dim]
    }

    fn values(&self, layer: usize, head: usize, upto: usize) -> &[f32] {
        let at = self.offset(layer, head, 0);
        &self.v[at..at + upto * self.head_dim]
    }
}

/// Per-head scratch space for the parallel attention step.
struct HeadScratch {
    /// This head's outputs for the tokens of the current chunk.
    out: Vec<f32>,
    /// Attention scores, one per visible position.
    scores: Vec<f32>,
}

/// Every temporary buffer the forward pass needs, allocated once.
///
/// `max_chunk` is the most tokens processed in one pass through the layers.
/// Longer inputs (long prompts) are processed in chunks of that size, which
/// bounds this memory whatever the prompt length.
pub struct Scratch {
    max_chunk: usize,
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
    heads: Vec<HeadScratch>,
    logits: Vec<f32>,
    matmul: Vec<f32>,
}

impl Scratch {
    pub fn new(config: &Config, max_chunk: usize, cache_capacity: usize) -> Self {
        let c = config;
        let n = max_chunk.max(1);
        Self {
            max_chunk: n,
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
            heads: (0..c.num_heads)
                .map(|_| HeadScratch {
                    out: vec![0.0; n * c.head_dim],
                    scores: vec![0.0; cache_capacity],
                })
                .collect(),
            logits: vec![0.0; n * c.vocab_size],
            matmul: Vec::new(),
        }
    }

    /// Bytes currently allocated for all the buffers.
    pub fn bytes(&self) -> usize {
        let buffers = [
            &self.x,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.proj,
            &self.gate,
            &self.up,
            &self.act,
            &self.logits,
            &self.matmul,
        ];
        let shared: usize = buffers.iter().map(|b| b.capacity()).sum();
        let heads: usize = self
            .heads
            .iter()
            .map(|h| h.out.capacity() + h.scores.capacity())
            .sum();
        (shared + heads) * size_of::<f32>()
    }
}

/// One transformer layer's weights.
pub struct Layer<W> {
    pub attn_norm: Vec<f32>,
    pub wq: W,
    pub wk: W,
    pub wv: W,
    pub wo: W,
    pub mlp_norm: Vec<f32>,
    pub w_gate: W,
    pub w_up: W,
    pub w_down: W,
}

/// A Llama-style model whose matrices are stored as `W`.
pub struct Model<W: Matrix> {
    pub config: Config,
    pub embed: W,
    pub layers: Vec<Layer<W>>,
    pub final_norm: Vec<f32>,
    /// `None` when the output layer is tied to the embedding.
    pub lm_head: Option<W>,
    rope: Rope,
}

impl<W: Matrix> Model<W> {
    pub fn new(
        config: Config,
        embed: W,
        layers: Vec<Layer<W>>,
        final_norm: Vec<f32>,
        lm_head: Option<W>,
    ) -> Self {
        assert_eq!(layers.len(), config.num_layers, "wrong number of layers");
        let rope = Rope::new(
            config.head_dim,
            config.max_positions,
            config.rope_theta,
            config.rope_layout,
        );
        Self {
            config,
            embed,
            layers,
            final_norm,
            lm_head,
            rope,
        }
    }

    pub fn lm_head(&self) -> &W {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// Takes the model apart, so that it can be rebuilt with [`Model::new`]
    /// around different matrices (chapter 17 wraps each one in a timer).
    pub fn into_parts(self) -> (Config, W, Vec<Layer<W>>, Vec<f32>, Option<W>) {
        (
            self.config,
            self.embed,
            self.layers,
            self.final_norm,
            self.lm_head,
        )
    }

    /// Bytes of weights read to process one token (every matrix once; the
    /// embedding is only read for the LM head, one row is negligible).
    pub fn weight_bytes_per_token(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                    .iter()
                    .map(|m| m.bytes())
                    .sum::<usize>()
            })
            .sum();
        layers + self.lm_head().bytes()
    }

    /// Processes `tokens` (continuing the sequence in `cache`) and returns
    /// the logits for the last one: a prefill when `tokens` is the prompt, a
    /// decode step when it is a single token.
    pub fn forward_last<'s>(
        &self,
        pool: &mut SpinPool,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &'s mut Scratch,
    ) -> &'s [f32] {
        self.forward(pool, tokens, cache, scratch, false);
        &scratch.logits[..self.config.vocab_size]
    }

    /// Like [`Model::forward_last`], but returns logits for every token in
    /// `tokens` (`[tokens.len() × vocab]`). Used to verify several draft
    /// tokens at once (chapter 26). `tokens.len()` must not exceed the
    /// scratch's chunk size.
    pub fn forward_all<'s>(
        &self,
        pool: &mut SpinPool,
        tokens: &[u32],
        cache: &mut KvCache,
        scratch: &'s mut Scratch,
    ) -> &'s [f32] {
        assert!(
            tokens.len() <= scratch.max_chunk,
            "forward_all needs one chunk"
        );
        self.forward(pool, tokens, cache, scratch, true);
        &scratch.logits[..tokens.len() * self.config.vocab_size]
    }

    fn forward(
        &self,
        pool: &mut SpinPool,
        tokens: &[u32],
        cache: &mut KvCache,
        s: &mut Scratch,
        all: bool,
    ) {
        assert!(!tokens.is_empty(), "nothing to process");
        assert!(
            cache.len() + tokens.len() <= cache.capacity(),
            "KV cache full: {} + {} > {}",
            cache.len(),
            tokens.len(),
            cache.capacity()
        );
        // Long inputs go through the layers in chunks of at most `max_chunk`
        // tokens. Only the last chunk needs logits.
        let chunks = tokens.len().div_ceil(s.max_chunk);
        for (i, chunk) in tokens.chunks(s.max_chunk).enumerate() {
            let last_chunk = i + 1 == chunks;
            self.forward_chunk(pool, chunk, cache, s, all && last_chunk, last_chunk);
        }
    }

    /// One pass through every layer for up to `max_chunk` tokens.
    fn forward_chunk(
        &self,
        pool: &mut SpinPool,
        tokens: &[u32],
        cache: &mut KvCache,
        s: &mut Scratch,
        all_logits: bool,
        want_logits: bool,
    ) {
        let c = &self.config;
        let m = tokens.len();
        let (h, q_dim, kv_dim, inter) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size);
        let start = cache.len();

        for (t, &token) in tokens.iter().enumerate() {
            self.embed
                .row_to_f32(token as usize, &mut s.x[t * h..(t + 1) * h]);
        }
        // Each `s.field[..]` below borrows one field of the scratch; the
        // borrow checker sees that different fields never overlap.
        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block.
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
            for t in 0..m {
                let pos = start + t;
                let (qt, kt) = (t * q_dim..(t + 1) * q_dim, t * kv_dim..(t + 1) * kv_dim);
                self.rope.apply_heads(&mut s.q[qt], pos);
                self.rope.apply_heads(&mut s.k[kt.clone()], pos);
                cache.store(l, pos, &s.k[kt.clone()], &s.v[kt]);
            }
            self.attention(pool, l, start, m, cache, s);
            layer.wo.matmul(
                pool,
                &s.attn[..m * q_dim],
                &mut s.proj[..m * h],
                m,
                &mut s.matmul,
            );
            add_inplace(&mut s.x[..m * h], &s.proj[..m * h]);

            // MLP block.
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
        cache.len = start + m;

        if !want_logits {
            return;
        }
        // Final norm and LM head: for every row, or only the last one.
        let rows = if all_logits { 0..m } else { m - 1..m };
        let n = rows.len();
        norm_rows(
            &s.x[rows.start * h..rows.end * h],
            &self.final_norm,
            c.rms_norm_eps,
            &mut s.normed[..n * h],
        );
        self.lm_head().matmul(
            pool,
            &s.normed[..n * h],
            &mut s.logits[..n * c.vocab_size],
            n,
            &mut s.matmul,
        );
    }

    /// Causal attention for the chunk's `m` new tokens against every cached
    /// position, one query head per task on the pool.
    fn attention(
        &self,
        pool: &mut SpinPool,
        layer: usize,
        start: usize,
        m: usize,
        cache: &KvCache,
        s: &mut Scratch,
    ) {
        let c = &self.config;
        let (d, q_dim) = (c.head_dim, c.q_dim());
        let heads = c.heads();
        let scale = 1.0 / (d as f32).sqrt();
        let q = &s.q;
        pool.for_each_chunk_mut(&mut s.heads, 1, |first, my_heads| {
            for (i, hs) in my_heads.iter_mut().enumerate() {
                let head = first + i;
                let kvh = heads.kv_head(head);
                for t in 0..m {
                    let visible = start + t + 1;
                    let query = &q[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                    let keys = cache.keys(layer, kvh, visible);
                    let scores = &mut hs.scores[..visible];
                    for (score, key) in scores.iter_mut().zip(keys.chunks_exact(d)) {
                        *score = dot(query, key) * scale;
                    }
                    softmax(scores);
                    let out = &mut hs.out[t * d..(t + 1) * d];
                    out.fill(0.0);
                    for (&p, value) in scores
                        .iter()
                        .zip(cache.values(layer, kvh, visible).chunks_exact(d))
                    {
                        for (o, &v) in out.iter_mut().zip(value) {
                            *o += p * v;
                        }
                    }
                }
            }
        });
        // Gather the head-major results into the [token × head × dim] layout
        // the output projection expects.
        for (head, hs) in s.heads.iter().enumerate() {
            for t in 0..m {
                s.attn[t * q_dim + head * d..t * q_dim + (head + 1) * d]
                    .copy_from_slice(&hs.out[t * d..(t + 1) * d]);
            }
        }
    }
}

/// RMSNorm applied to each `hidden`-sized row.
fn norm_rows(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let h = weight.len();
    for (xi, oi) in x.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
        rms_norm(xi, weight, eps, oi);
    }
}

impl Model<DenseF32> {
    /// Builds the engine's model from chapter 13's reference weights.
    pub fn from_reference(w: &ch13_transformer::Weights) -> Self {
        let c = &w.config;
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        let layers = w
            .layers
            .iter()
            .map(|l| Layer {
                attn_norm: l.attn_norm.clone(),
                wq: DenseF32::new(&l.wq, c.q_dim(), h),
                wk: DenseF32::new(&l.wk, c.kv_dim(), h),
                wv: DenseF32::new(&l.wv, c.kv_dim(), h),
                wo: DenseF32::new(&l.wo, h, c.q_dim()),
                mlp_norm: l.mlp_norm.clone(),
                w_gate: DenseF32::new(&l.w_gate, inter, h),
                w_up: DenseF32::new(&l.w_up, inter, h),
                w_down: DenseF32::new(&l.w_down, h, inter),
            })
            .collect();
        Self::new(
            c.clone(),
            DenseF32::new(&w.embed, c.vocab_size, h),
            layers,
            w.final_norm.clone(),
            w.lm_head
                .as_ref()
                .map(|head| DenseF32::new(head, c.vocab_size, h)),
        )
    }
}

/// Index of the largest logit.
pub fn argmax(logits: &[f32]) -> u32 {
    ch13_transformer::argmax(logits)
}

/// Greedy generation with the cache: one prefill of the whole prompt, then
/// one decode step per new token. `on_token` sees each token as soon as it
/// exists, with the time the step took. Returns prompt + generated tokens.
pub fn generate_greedy<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    prompt: &[u32],
    new_tokens: usize,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    mut on_token: impl FnMut(u32, Duration),
) -> Vec<u32> {
    cache.clear();
    let mut tokens = prompt.to_vec();
    for step in 0..new_tokens {
        let start = Instant::now();
        // First step: the whole prompt. Afterwards: only the newest token.
        let input = if step == 0 {
            prompt
        } else {
            &tokens[tokens.len() - 1..]
        };
        let next = argmax(model.forward_last(pool, input, cache, scratch));
        on_token(next, start.elapsed());
        tokens.push(next);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch13_transformer::Weights;

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
    }

    fn setup(seed: u64) -> (Weights, Model<DenseF32>, SpinPool) {
        let w = Weights::random(&Config::tiny(), seed);
        let m = Model::from_reference(&w);
        (w, m, SpinPool::new(3))
    }

    #[test]
    fn cached_decoding_matches_the_reference_at_every_step() {
        let (w, model, mut pool) = setup(1);
        let vocab = w.config.vocab_size;
        let mut tokens = vec![5u32, 17, 3, 90, 42, 8, 11];
        let mut cache = KvCache::new(&w.config, 64);
        let mut scratch = Scratch::new(&w.config, 16, 64);

        // Prefill the prompt, then decode 6 tokens one at a time.
        let mut logits = model
            .forward_last(&mut pool, &tokens, &mut cache, &mut scratch)
            .to_vec();
        for _ in 0..6 {
            let reference = w.forward(&mut pool, &tokens);
            let want = &reference[(tokens.len() - 1) * vocab..];
            assert!(close(&logits, want), "diverged at length {}", tokens.len());
            let next = argmax(&logits);
            tokens.push(next);
            logits = model
                .forward_last(&mut pool, &[next], &mut cache, &mut scratch)
                .to_vec();
        }
        assert_eq!(cache.len(), tokens.len());
    }

    #[test]
    fn chunked_prefill_matches_one_big_prefill() {
        let (w, model, mut pool) = setup(2);
        let prompt: Vec<u32> = (0..23).map(|i| (i * 7 + 1) % 97).collect();
        let mut cache_a = KvCache::new(&w.config, 64);
        let mut cache_b = KvCache::new(&w.config, 64);
        let mut big = Scratch::new(&w.config, 32, 64);
        let mut small = Scratch::new(&w.config, 4, 64); // 6 chunks
        let a = model
            .forward_last(&mut pool, &prompt, &mut cache_a, &mut big)
            .to_vec();
        let b = model
            .forward_last(&mut pool, &prompt, &mut cache_b, &mut small)
            .to_vec();
        assert!(close(&a, &b));
    }

    #[test]
    fn forward_all_matches_every_reference_row() {
        let (w, model, mut pool) = setup(3);
        let tokens = [1u32, 2, 3, 4, 5];
        let mut cache = KvCache::new(&w.config, 16);
        let mut scratch = Scratch::new(&w.config, 8, 16);
        let all = model.forward_all(&mut pool, &tokens, &mut cache, &mut scratch);
        assert!(close(all, &w.forward(&mut pool, &tokens)));
    }

    #[test]
    fn truncating_the_cache_rolls_the_sequence_back() {
        let (w, model, mut pool) = setup(4);
        let mut cache = KvCache::new(&w.config, 32);
        let mut scratch = Scratch::new(&w.config, 8, 32);
        model.forward_last(&mut pool, &[1, 2, 3], &mut cache, &mut scratch);
        let after_4 = model
            .forward_last(&mut pool, &[4], &mut cache, &mut scratch)
            .to_vec();
        model.forward_last(&mut pool, &[60, 61], &mut cache, &mut scratch); // a wrong turn
        cache.truncate(3);
        let again = model.forward_last(&mut pool, &[4], &mut cache, &mut scratch);
        assert!(close(again, &after_4));
    }

    #[test]
    fn greedy_generation_matches_generation_without_a_cache() {
        let (w, model, mut pool) = setup(6);
        let prompt = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut cache = KvCache::new(&w.config, 32);
        let mut scratch = Scratch::new(&w.config, 8, 32);
        let cached = generate_greedy(
            &model,
            &mut pool,
            &prompt,
            12,
            &mut cache,
            &mut scratch,
            |_, _| {},
        );
        let uncached =
            ch13_transformer::generate_without_cache(&w, &mut pool, &prompt, 12, |_, _| {});
        assert_eq!(cached, uncached);
    }

    #[test]
    #[should_panic(expected = "KV cache full")]
    fn overflowing_the_cache_is_an_error() {
        let (w, model, mut pool) = setup(5);
        let mut cache = KvCache::new(&w.config, 4);
        let mut scratch = Scratch::new(&w.config, 8, 4);
        model.forward_last(&mut pool, &[1, 2, 3, 4, 5], &mut cache, &mut scratch);
    }
}
