//! Layer-by-layer numerical debugging: record the hidden state that enters
//! every layer, and compare two recordings to find where they part.
//!
//! When a model's output is wrong, the logits say only *that* something is
//! wrong. Comparing the hidden state layer by layer with a trusted
//! implementation (PyTorch, a previous version, the unquantized model)
//! says *where*: a bug shows as a jump at one layer, numerical drift
//! (quantization, a different summation order) as a slow growth.

use ch07_threads::SpinPool;
use ch08_operators::rms_norm;
use ch14_kv_cache::{KvCache, Matrix, Model};
use ch28_parallelism::pp::{Buffers, run_layers};

/// The hidden state of the last token of `tokens` as it enters each layer
/// (`num_layers` rows), then after the final norm: `num_layers + 1` rows,
/// the layout of chapter 16's PyTorch fixture.
pub fn trace_last<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    let c = &model.config;
    let mut cache = KvCache::new(c, tokens.len());
    let mut buf = Buffers::new(model, tokens.len());
    let mut x = vec![0.0; c.hidden_size];
    let mut rows = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        let last = pos + 1 == tokens.len();
        model.embed.row_to_f32(token as usize, &mut x);
        for l in 0..c.num_layers {
            if last {
                rows.push(x.clone());
            }
            run_layers(model, l..l + 1, pool, &mut x, pos, &mut cache, &mut buf);
        }
        cache.advance(1);
    }
    let mut normed = vec![0.0; c.hidden_size];
    rms_norm(&x, &model.final_norm, c.rms_norm_eps, &mut normed);
    rows.push(normed);
    rows
}

/// How far one vector is from a reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Diff {
    /// `|a − b| / |b|`, with Euclidean norms.
    pub relative: f64,
    pub max_abs: f64,
    /// Cosine of the angle between `a` and `b`.
    pub cosine: f64,
}

pub fn diff(a: &[f32], b: &[f32]) -> Diff {
    let (mut dd, mut bb, mut aa, mut ab, mut max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dd += (x - y) * (x - y);
        aa += x * x;
        bb += y * y;
        ab += x * y;
        max = max.max((x - y).abs());
    }
    Diff {
        relative: (dd / bb).sqrt(),
        max_abs: max,
        cosine: ab / (aa.sqrt() * bb.sqrt()),
    }
}

/// The first row whose relative difference exceeds `factor` times the
/// largest one before it, counting anything below `floor` as `floor` (so
/// that rounding noise growing from 10⁻⁹ to 10⁻⁷ is not a jump): where a
/// bug, rather than drift, enters.
pub fn first_jump(diffs: &[Diff], factor: f64, floor: f64) -> Option<usize> {
    let mut worst = floor;
    for (i, d) in diffs.iter().enumerate() {
        if d.relative > factor * worst {
            return Some(i);
        }
        worst = worst.max(d.relative);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, Scratch};

    #[test]
    fn the_trace_ends_in_the_models_own_final_state() {
        let model: Model<DenseF32> = Model::from_reference(&Weights::random(&Config::tiny(), 30));
        let tokens = [5, 17, 3, 88, 42];
        let mut pool = SpinPool::new(1);
        let rows = trace_last(&model, &mut pool, &tokens);
        assert_eq!(rows.len(), model.config.num_layers + 1);
        // The final row through the LM head gives chapter 14's logits.
        let mut logits = vec![0.0; model.config.vocab_size];
        model
            .lm_head()
            .matmul(&mut pool, &rows[2], &mut logits, 1, &mut Vec::new());
        let mut cache = KvCache::new(&model.config, 16);
        let mut scratch = Scratch::new(&model.config, 16, 16);
        let want = model.forward_last(&mut pool, &tokens, &mut cache, &mut scratch);
        assert!(diff(&logits, want).relative < 1e-5);
        // The first row is the last token's embedding.
        let mut e = vec![0.0; model.config.hidden_size];
        model.embed.row_to_f32(42, &mut e);
        assert_eq!(rows[0], e);
    }

    #[test]
    fn a_jump_is_found_and_drift_is_not() {
        let d = |relative| Diff {
            relative,
            max_abs: 0.0,
            cosine: 1.0,
        };
        let drift: Vec<Diff> = (1..10).map(|i| d(1e-3 * f64::from(i))).collect();
        assert_eq!(first_jump(&drift, 10.0, 1e-4), None);
        let mut bug = drift.clone();
        bug[6] = d(0.5);
        assert_eq!(first_jump(&bug, 10.0, 1e-4), Some(6));
        // Rounding noise rising from 1e-9 to 1e-7 stays under the floor.
        let noise: Vec<Diff> = [1e-9, 1e-8, 1e-7].into_iter().map(d).collect();
        assert_eq!(first_jump(&noise, 10.0, 1e-4), None);
    }
}
