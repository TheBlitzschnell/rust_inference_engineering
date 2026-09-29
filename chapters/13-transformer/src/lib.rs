//! Chapter 13: a complete decoder-only transformer in the Llama style.
//!
//! This is the *reference* implementation: it recomputes everything for
//! every token and favours clarity over speed. Chapter 14 builds the fast
//! engine and tests it against this one.
//!
//! One layer:
//!
//! ```text
//! x ─┬─ RMSNorm ─ q,k,v projections ─ RoPE ─ causal GQA attention ─ o projection ─(+)─┐
//!    └──────────────────────────────── residual ─────────────────────────────────────┘ │
//! x ─┬─ RMSNorm ─ gate, up projections ─ SwiGLU ─ down projection ──(+)──► next layer ─┘
//!    └──────────────────────── residual ─────────────────────────────┘
//! ```
//!
//! The model: token embedding, `num_layers` of those layers, a final RMSNorm,
//! and an output projection ("LM head") to one logit per vocabulary entry.

use ch07_threads::{SpinPool, matmul_nt_pool};
use ch08_operators::{add_inplace, rms_norm, swiglu};
use ch12_attention::{Heads, Rope, RopeLayout, attention};

/// The hyperparameters that define a model's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_layout: RopeLayout,
    pub rms_norm_eps: f32,
    pub max_positions: usize,
    /// If true, the output projection reuses the embedding matrix.
    pub tie_embeddings: bool,
}

impl Config {
    /// SmolLM2-135M, as in its `config.json`.
    pub fn smollm2_135m() -> Self {
        Self {
            vocab_size: 49_152,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 30,
            num_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            rope_theta: 100_000.0,
            rope_layout: RopeLayout::HalfSplit,
            rms_norm_eps: 1e-5,
            max_positions: 8192,
            tie_embeddings: true,
        }
    }

    /// A tiny model for tests: every feature, almost no weights.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 97,
            hidden_size: 48,
            intermediate_size: 96,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 12,
            rope_theta: 10_000.0,
            rope_layout: RopeLayout::HalfSplit,
            rms_norm_eps: 1e-5,
            max_positions: 256,
            tie_embeddings: true,
        }
    }

    pub fn heads(&self) -> Heads {
        Heads {
            n_heads: self.num_heads,
            n_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Parameters of one transformer layer.
    pub fn layer_params(&self) -> usize {
        let h = self.hidden_size;
        let attention = h * self.q_dim() + 2 * h * self.kv_dim() + self.q_dim() * h;
        let mlp = 3 * h * self.intermediate_size;
        let norms = 2 * h;
        attention + mlp + norms
    }

    /// Every parameter in the model.
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.hidden_size;
        let head = if self.tie_embeddings { 0 } else { embed };
        embed + self.num_layers * self.layer_params() + self.hidden_size + head
    }

    /// Floating-point operations to process one token that attends to
    /// `context` earlier tokens: two per weight used in a matmul (the
    /// embedding lookup is free), plus attention's scores and weighted sums.
    pub fn flops_per_token(&self, context: usize) -> f64 {
        let matmul_weights = self.num_layers * (self.layer_params() - 2 * self.hidden_size)
            + self.vocab_size * self.hidden_size;
        let attention = 4 * self.num_layers * self.num_heads * self.head_dim * context;
        2.0 * matmul_weights as f64 + attention as f64
    }
}

/// The weights of one layer, each matrix stored `[out × in]` (PyTorch's
/// layout), in plain `f32`.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

/// All of a model's weights.
#[derive(Debug, Clone)]
pub struct Weights {
    pub config: Config,
    /// `[vocab × hidden]`.
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    /// `[vocab × hidden]`, or `None` when tied to `embed`.
    pub lm_head: Option<Vec<f32>>,
}

/// A small deterministic generator for random weights (xorshift64*).
struct Rng(u64);

impl Rng {
    fn normal(&mut self) -> f32 {
        // Sum of 12 uniforms minus 6: close enough to a standard normal.
        (0..12)
            .map(|_| {
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
            })
            .sum::<f32>()
            - 6.0
    }
}

impl Weights {
    /// Random weights with the given shape: normal with standard deviation
    /// 0.02 for matrices, ones for norms. Useless as a language model, but
    /// every computation is the same as for real weights.
    pub fn random(config: &Config, seed: u64) -> Self {
        let mut rng = Rng(seed.max(1));
        let mut matrix = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.normal() * 0.02).collect()
        };
        let h = config.hidden_size;
        let layers = (0..config.num_layers)
            .map(|_| LayerWeights {
                attn_norm: vec![1.0; h],
                wq: matrix(config.q_dim(), h),
                wk: matrix(config.kv_dim(), h),
                wv: matrix(config.kv_dim(), h),
                wo: matrix(h, config.q_dim()),
                mlp_norm: vec![1.0; h],
                w_gate: matrix(config.intermediate_size, h),
                w_up: matrix(config.intermediate_size, h),
                w_down: matrix(h, config.intermediate_size),
            })
            .collect();
        let embed = matrix(config.vocab_size, h);
        let lm_head = (!config.tie_embeddings).then(|| matrix(config.vocab_size, h));
        Self {
            config: config.clone(),
            embed,
            layers,
            final_norm: vec![1.0; h],
            lm_head,
        }
    }

    /// The output projection: its own matrix, or the embedding if tied.
    pub fn lm_head(&self) -> &[f32] {
        self.lm_head.as_deref().unwrap_or(&self.embed)
    }

    /// Runs the whole model on `tokens` and returns the logits for every
    /// position: `[tokens.len() × vocab_size]`. Row `t` scores every
    /// possible token at position `t + 1`.
    pub fn forward(&self, pool: &mut SpinPool, tokens: &[u32]) -> Vec<f32> {
        let c = &self.config;
        let n = tokens.len();
        let (h, q_dim, kv_dim) = (c.hidden_size, c.q_dim(), c.kv_dim());
        assert!(
            n <= c.max_positions,
            "sequence longer than the model supports"
        );
        // Only positions 0..n are needed; building the table for all
        // `max_positions` would cost more than a short forward pass.
        let rope = Rope::new(c.head_dim, n.max(1), c.rope_theta, c.rope_layout);

        // 1. Embedding lookup: one row per token.
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| {
                self.embed[t as usize * h..(t as usize + 1) * h]
                    .iter()
                    .copied()
            })
            .collect();

        let mut normed = vec![0.0; n * h];
        let mut q = vec![0.0; n * q_dim];
        let mut k = vec![0.0; n * kv_dim];
        let mut v = vec![0.0; n * kv_dim];
        let mut attn = vec![0.0; n * q_dim];
        let mut proj = vec![0.0; n * h];
        let mut gate = vec![0.0; n * c.intermediate_size];
        let mut up = vec![0.0; n * c.intermediate_size];
        let mut act = vec![0.0; n * c.intermediate_size];
        let mut scores = vec![0.0; n];

        for layer in &self.layers {
            // 2. Attention block.
            for (xi, ni) in x.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
                rms_norm(xi, &layer.attn_norm, c.rms_norm_eps, ni);
            }
            matmul_nt_pool(pool, &normed, &layer.wq, &mut q, n, h, q_dim);
            matmul_nt_pool(pool, &normed, &layer.wk, &mut k, n, h, kv_dim);
            matmul_nt_pool(pool, &normed, &layer.wv, &mut v, n, h, kv_dim);
            for (pos, (qt, kt)) in q
                .chunks_exact_mut(q_dim)
                .zip(k.chunks_exact_mut(kv_dim))
                .enumerate()
            {
                rope.apply_heads(qt, pos);
                rope.apply_heads(kt, pos);
            }
            attention(&q, &k, &v, &mut attn, c.heads(), 0, true, &mut scores);
            matmul_nt_pool(pool, &attn, &layer.wo, &mut proj, n, q_dim, h);
            add_inplace(&mut x, &proj);

            // 3. Feed-forward (MLP) block.
            for (xi, ni) in x.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
                rms_norm(xi, &layer.mlp_norm, c.rms_norm_eps, ni);
            }
            matmul_nt_pool(
                pool,
                &normed,
                &layer.w_gate,
                &mut gate,
                n,
                h,
                c.intermediate_size,
            );
            matmul_nt_pool(
                pool,
                &normed,
                &layer.w_up,
                &mut up,
                n,
                h,
                c.intermediate_size,
            );
            swiglu(&gate, &up, &mut act);
            matmul_nt_pool(
                pool,
                &act,
                &layer.w_down,
                &mut proj,
                n,
                c.intermediate_size,
                h,
            );
            add_inplace(&mut x, &proj);
        }

        // 4. Final norm and output projection.
        for (xi, ni) in x.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
            rms_norm(xi, &self.final_norm, c.rms_norm_eps, ni);
        }
        let mut logits = vec![0.0; n * c.vocab_size];
        matmul_nt_pool(
            pool,
            &normed,
            self.lm_head(),
            &mut logits,
            n,
            h,
            c.vocab_size,
        );
        logits
    }
}

/// Index of the largest logit.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0 as u32
}

/// Greedy generation with no cache: every step runs the model on the whole
/// sequence so far and keeps only the last position's logits.
pub fn generate_without_cache(
    weights: &Weights,
    pool: &mut SpinPool,
    prompt: &[u32],
    new_tokens: usize,
    mut on_step: impl FnMut(usize, std::time::Duration),
) -> Vec<u32> {
    let vocab = weights.config.vocab_size;
    let mut tokens = prompt.to_vec();
    for _ in 0..new_tokens {
        let start = std::time::Instant::now();
        let logits = weights.forward(pool, &tokens);
        let last = &logits[(tokens.len() - 1) * vocab..];
        tokens.push(argmax(last));
        on_step(tokens.len() - 1, start.elapsed());
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smollm2_parameter_count_matches_the_checkpoint() {
        // chapter 9 counted 134,515,008 parameters in model.safetensors.
        assert_eq!(Config::smollm2_135m().param_count(), 134_515_008);
    }

    #[test]
    fn forward_has_one_row_of_logits_per_token() {
        let w = Weights::random(&Config::tiny(), 1);
        let mut pool = SpinPool::new(2);
        let logits = w.forward(&mut pool, &[1, 5, 9]);
        assert_eq!(logits.len(), 3 * 97);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(logits, w.forward(&mut pool, &[1, 5, 9]), "deterministic");
    }

    #[test]
    fn appending_tokens_never_changes_earlier_logits() {
        let w = Weights::random(&Config::tiny(), 2);
        let mut pool = SpinPool::new(2);
        let short = w.forward(&mut pool, &[3, 1, 4, 1]);
        let long = w.forward(&mut pool, &[3, 1, 4, 1, 5, 9, 2]);
        for (a, b) in short.iter().zip(&long[..short.len()]) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn untied_head_is_used_when_present() {
        let mut config = Config::tiny();
        config.tie_embeddings = false;
        let w = Weights::random(&config, 3);
        assert!(w.lm_head.is_some());
        assert!(!std::ptr::eq(w.lm_head().as_ptr(), w.embed.as_ptr()));
        assert_eq!(config.param_count(), Config::tiny().param_count() + 97 * 48);
    }
}
