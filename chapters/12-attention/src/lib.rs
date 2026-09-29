//! Chapter 12: attention, the operation that lets tokens look at each other.
//!
//! For each token, attention compares a *query* vector with the *key*
//! vectors of the tokens it may look at, turns the similarity scores into
//! weights with softmax, and returns the weighted average of those tokens'
//! *value* vectors. This crate implements:
//!
//! - [`attention`]: scaled dot-product attention with causal masking and
//!   grouped-query attention (several query heads sharing one key/value head),
//! - [`Rope`]: rotary position embeddings, in both memory layouts found in
//!   real checkpoints.
//!
//! Layouts used throughout, all row-major:
//! - queries `q`: `[n_q_tokens × n_heads × head_dim]`
//! - keys and values `k`, `v`: `[n_kv_tokens × n_kv_heads × head_dim]`
//! - output: `[n_q_tokens × n_heads × head_dim]`

use ch06_simd::dot;
use ch08_operators::softmax;

/// Shapes shared by every attention call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heads {
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads. Equal to `n_heads` for multi-head
    /// attention (MHA), 1 for multi-query attention (MQA), in between for
    /// grouped-query attention (GQA).
    pub n_kv_heads: usize,
    /// Size of each head's vectors.
    pub head_dim: usize,
}

impl Heads {
    /// How many query heads share each key/value head.
    pub fn group_size(&self) -> usize {
        assert!(
            self.n_heads.is_multiple_of(self.n_kv_heads),
            "query heads must divide evenly among key/value heads"
        );
        self.n_heads / self.n_kv_heads
    }

    /// Which key/value head query head `h` reads.
    pub fn kv_head(&self, h: usize) -> usize {
        h / self.group_size()
    }
}

/// Scaled dot-product attention.
///
/// Query token `t` sits at absolute position `q_start + t`. Keys and values
/// are for positions `0..n_kv_tokens`. With `causal`, a query may only look
/// at keys at its own position or earlier: the model cannot peek at tokens
/// that have not been generated yet.
///
/// `scores` is scratch space of at least `n_kv_tokens` floats.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: Heads,
    q_start: usize,
    causal: bool,
    scores: &mut [f32],
) {
    let Heads {
        n_heads,
        n_kv_heads,
        head_dim: d,
    } = heads;
    let q_row = n_heads * d;
    let kv_row = n_kv_heads * d;
    let n_q = q.len() / q_row;
    let n_kv = k.len() / kv_row;
    assert_eq!(q.len(), n_q * q_row);
    assert_eq!(k.len(), n_kv * kv_row);
    assert_eq!(v.len(), k.len());
    assert_eq!(out.len(), q.len());
    // Dividing scores by sqrt(d) keeps their size independent of the head
    // dimension, so softmax does not saturate for large heads.
    let scale = 1.0 / (d as f32).sqrt();

    for t in 0..n_q {
        let pos = q_start + t;
        // Causal: keys 0..=pos. Otherwise: every key.
        let visible = if causal { (pos + 1).min(n_kv) } else { n_kv };
        for h in 0..n_heads {
            let kvh = heads.kv_head(h);
            let query = &q[t * q_row + h * d..t * q_row + (h + 1) * d];
            let s = &mut scores[..visible];
            for (j, score) in s.iter_mut().enumerate() {
                let key = &k[j * kv_row + kvh * d..j * kv_row + (kvh + 1) * d];
                *score = dot(query, key) * scale;
            }
            softmax(s);
            let o = &mut out[t * q_row + h * d..t * q_row + (h + 1) * d];
            o.fill(0.0);
            for (j, &p) in s.iter().enumerate() {
                let value = &v[j * kv_row + kvh * d..j * kv_row + (kvh + 1) * d];
                for (oi, &vi) in o.iter_mut().zip(value) {
                    *oi += p * vi;
                }
            }
        }
    }
}

/// The attention weights (after softmax) for one head, as a
/// `[n_q_tokens × n_kv_tokens]` matrix. Masked entries are 0. Used to look
/// at what attention does; the fast path never builds this matrix.
pub fn attention_weights(
    q: &[f32],
    k: &[f32],
    heads: Heads,
    head: usize,
    q_start: usize,
    causal: bool,
) -> Vec<f32> {
    let d = heads.head_dim;
    let (q_row, kv_row) = (heads.n_heads * d, heads.n_kv_heads * d);
    let (n_q, n_kv) = (q.len() / q_row, k.len() / kv_row);
    let kvh = heads.kv_head(head);
    let scale = 1.0 / (d as f32).sqrt();
    let mut w = vec![0.0; n_q * n_kv];
    for t in 0..n_q {
        let visible = if causal {
            (q_start + t + 1).min(n_kv)
        } else {
            n_kv
        };
        let row = &mut w[t * n_kv..t * n_kv + visible];
        let query = &q[t * q_row + head * d..t * q_row + (head + 1) * d];
        for (j, s) in row.iter_mut().enumerate() {
            *s = dot(query, &k[j * kv_row + kvh * d..j * kv_row + (kvh + 1) * d]) * scale;
        }
        softmax(row);
    }
    w
}

/// How the two halves of each rotated pair are laid out in a head vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Pairs are neighbours: (x0, x1), (x2, x3), ... Used by the original
    /// Llama release and by llama.cpp's GGUF files.
    Interleaved,
    /// Pairs are split across the halves: (x0, x_{d/2}), (x1, x_{d/2+1}), ...
    /// Used by Hugging Face's Llama implementation ("rotate_half"), and so
    /// by SmolLM2's safetensors weights.
    HalfSplit,
}

/// Rotary position embeddings (RoPE).
///
/// Each pair of dimensions `i` of a query or key vector is treated as a point
/// in a plane and rotated by the angle `pos × θ^(-2i/d)`. Low dimensions
/// rotate quickly with position, high dimensions slowly. Because rotating
/// both a query and a key by their positions changes their dot product only
/// through the *difference* of the positions, attention scores depend on how
/// far apart two tokens are, not where they are.
#[derive(Debug, Clone)]
pub struct Rope {
    head_dim: usize,
    layout: RopeLayout,
    /// `cos[pos * half + i]` and `sin[...]` for every position and pair.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// Precomputes cos and sin for positions `0..max_positions`.
    ///
    /// The rounding follows the Hugging Face reference: the inverse
    /// frequencies and the angles are computed in `f32`, and only the
    /// cosine and sine themselves are evaluated precisely. Doing it "better"
    /// (all in `f64`) would make our numbers drift away from the reference
    /// at long positions.
    pub fn new(head_dim: usize, max_positions: usize, theta: f32, layout: RopeLayout) -> Self {
        assert!(
            head_dim.is_multiple_of(2),
            "RoPE needs an even head dimension"
        );
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(max_positions * half);
        let mut sin = Vec::with_capacity(max_positions * half);
        for pos in 0..max_positions {
            for &f in &inv_freq {
                let angle = f64::from(pos as f32 * f);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            head_dim,
            layout,
            cos,
            sin,
        }
    }

    pub fn max_positions(&self) -> usize {
        self.cos.len() / (self.head_dim / 2)
    }

    /// Rotates one head vector in place for position `pos`.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        assert_eq!(x.len(), self.head_dim);
        assert!(
            pos < self.max_positions(),
            "position {pos} beyond the RoPE table"
        );
        let cos = &self.cos[pos * half..(pos + 1) * half];
        let sin = &self.sin[pos * half..(pos + 1) * half];
        match self.layout {
            RopeLayout::Interleaved => {
                for (pair, (&c, &s)) in x.chunks_exact_mut(2).zip(cos.iter().zip(sin)) {
                    let (a, b) = (pair[0], pair[1]);
                    pair[0] = a * c - b * s;
                    pair[1] = a * s + b * c;
                }
            }
            RopeLayout::HalfSplit => {
                let (lo, hi) = x.split_at_mut(half);
                for ((a, b), (&c, &s)) in lo.iter_mut().zip(hi.iter_mut()).zip(cos.iter().zip(sin))
                {
                    let (x0, x1) = (*a, *b);
                    *a = x0 * c - x1 * s;
                    *b = x0 * s + x1 * c;
                }
            }
        }
    }

    /// Rotates every head of a `[n_heads × head_dim]` vector.
    pub fn apply_heads(&self, x: &mut [f32], pos: usize) {
        for head in x.chunks_exact_mut(self.head_dim) {
            self.apply(head, pos);
        }
    }
}

/// Converts a head vector between the two RoPE layouts: interleaved
/// `(x0, x1, x2, x3, ...)` pairs become `(x0, x2, ..., x1, x3, ...)` halves.
/// Loading a checkpoint written for one layout into code expecting the other
/// requires permuting the rows of the query and key weights this way.
pub fn interleaved_to_half_split(x: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = vec![0.0; x.len()];
    for i in 0..half {
        out[i] = x[2 * i];
        out[half + i] = x[2 * i + 1];
    }
    out
}

/// Bytes of key/value cache per token: keys and values, for every layer and
/// key/value head, at `bytes_per_value` each.
pub fn kv_bytes_per_token(layers: usize, heads: Heads, bytes_per_value: usize) -> usize {
    2 * layers * heads.n_kv_heads * heads.head_dim * bytes_per_value
}

/// Deterministic pseudo-random numbers in [-1, 1), for tests and demos.
pub fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    ch06_simd::random_vec(len, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADS: Heads = Heads {
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
    };

    fn run(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: Heads,
        q_start: usize,
        causal: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0; q.len()];
        let mut scores = vec![0.0; k.len() / (heads.n_kv_heads * heads.head_dim)];
        attention(q, k, v, &mut out, heads, q_start, causal, &mut scores);
        out
    }

    #[test]
    fn weights_are_probabilities_and_respect_the_mask() {
        let (n, row_q, row_kv) = (5, 32, 16);
        let q = random_vec(n * row_q, 1);
        let k = random_vec(n * row_kv, 2);
        let w = attention_weights(&q, &k, HEADS, 1, 0, true);
        for t in 0..n {
            let row = &w[t * n..(t + 1) * n];
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(row[t + 1..].iter().all(|&p| p == 0.0), "future is masked");
        }
    }

    #[test]
    fn the_future_cannot_change_the_past() {
        let n = 6;
        let (q, k, v) = (
            random_vec(n * 32, 3),
            random_vec(n * 16, 4),
            random_vec(n * 16, 5),
        );
        let before = run(&q, &k, &v, HEADS, 0, true);
        // Change everything about the last two tokens.
        let (mut q2, mut k2, mut v2) = (q.clone(), k.clone(), v.clone());
        for x in q2[4 * 32..]
            .iter_mut()
            .chain(&mut k2[4 * 16..])
            .chain(&mut v2[4 * 16..])
        {
            *x = -*x * 3.0;
        }
        let after = run(&q2, &k2, &v2, HEADS, 0, true);
        assert_eq!(&before[..4 * 32], &after[..4 * 32]);
        assert_ne!(&before[4 * 32..], &after[4 * 32..]);
    }

    #[test]
    fn gqa_equals_mha_with_repeated_kv_heads() {
        let n = 4;
        let (q, k, v) = (
            random_vec(n * 32, 6),
            random_vec(n * 16, 7),
            random_vec(n * 16, 8),
        );
        let gqa = run(&q, &k, &v, HEADS, 0, true);
        // Expand 2 KV heads to 4 by repeating each for its group of 2.
        let expand = |x: &[f32]| -> Vec<f32> {
            x.chunks_exact(16)
                .flat_map(|tok| {
                    let (h0, h1) = tok.split_at(8);
                    [h0, h0, h1, h1].concat()
                })
                .collect()
        };
        let mha_heads = Heads {
            n_kv_heads: 4,
            ..HEADS
        };
        let mha = run(&q, &expand(&k), &expand(&v), mha_heads, 0, true);
        for (a, b) in gqa.iter().zip(&mha) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn one_visible_key_returns_its_value() {
        // Token 0 can only see itself, so its output is exactly its value.
        let (q, k, v) = (
            random_vec(3 * 32, 9),
            random_vec(3 * 16, 10),
            random_vec(3 * 16, 11),
        );
        let out = run(&q, &k, &v, HEADS, 0, true);
        for h in 0..4 {
            let kvh = HEADS.kv_head(h);
            for (a, b) in out[h * 8..(h + 1) * 8]
                .iter()
                .zip(&v[kvh * 8..(kvh + 1) * 8])
            {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn query_offset_matches_the_full_computation() {
        // Computing only the last 2 queries (q_start = 4) must give the same
        // outputs as the last 2 rows of computing all 6. This is what makes
        // the KV cache of chapter 14 possible.
        let n = 6;
        let (q, k, v) = (
            random_vec(n * 32, 12),
            random_vec(n * 16, 13),
            random_vec(n * 16, 14),
        );
        let full = run(&q, &k, &v, HEADS, 0, true);
        let tail = run(&q[4 * 32..], &k, &v, HEADS, 4, true);
        for (a, b) in tail.iter().zip(&full[4 * 32..]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_length_and_depends_only_on_distance() {
        for layout in [RopeLayout::Interleaved, RopeLayout::HalfSplit] {
            let rope = Rope::new(64, 4096, 10_000.0, layout);
            let (q, k) = (random_vec(64, 15), random_vec(64, 16));
            let norm = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let score = |m: usize, n: usize| {
                let (mut qm, mut kn) = (q.clone(), k.clone());
                rope.apply(&mut qm, m);
                rope.apply(&mut kn, n);
                assert!((norm(&qm) - norm(&q)).abs() < 1e-4);
                dot(&qm, &kn)
            };
            // Same distance (3), very different absolute positions.
            let a = score(10, 7);
            let b = score(2010, 2007);
            assert!((a - b).abs() < 1e-3, "{layout:?}: {a} vs {b}");
            // Position 0 is the identity.
            let mut x = q.clone();
            rope.apply(&mut x, 0);
            assert_eq!(x, q);
        }
    }

    #[test]
    fn the_two_layouts_agree_after_permuting() {
        let x = random_vec(64, 17);
        let mut a = x.clone();
        Rope::new(64, 100, 10_000.0, RopeLayout::Interleaved).apply(&mut a, 37);
        let mut b = interleaved_to_half_split(&x);
        Rope::new(64, 100, 10_000.0, RopeLayout::HalfSplit).apply(&mut b, 37);
        for (p, q) in interleaved_to_half_split(&a).iter().zip(&b) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn kv_cache_size_formula() {
        // SmolLM2-135M: 30 layers, 3 KV heads of 64, f32.
        let smol = Heads {
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
        };
        assert_eq!(kv_bytes_per_token(30, smol, 4), 46_080);
    }
}
