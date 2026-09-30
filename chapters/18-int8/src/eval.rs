//! Measuring what a cheaper model loses, against a reference.

use ch07_threads::SpinPool;
use ch08_operators::log_softmax;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};

/// Log-probabilities (natural log) of every vocabulary token, for every
/// position of `tokens`, processed as one sequence from an empty cache:
/// `tokens.len() × vocab` values.
pub fn log_probs<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    tokens: &[u32],
    cache: &mut KvCache,
    scratch: &mut Scratch,
) -> Vec<f32> {
    cache.clear();
    let mut out = model.forward_all(pool, tokens, cache, scratch).to_vec();
    for row in out.chunks_exact_mut(model.config.vocab_size) {
        log_softmax(row);
    }
    out
}

/// Accumulated comparison of a candidate model with a reference over many
/// predicted tokens.
#[derive(Clone, Copy, Debug, Default)]
pub struct Quality {
    /// Next-token predictions compared.
    pub tokens: usize,
    /// Sum of the reference's negative log-likelihood of the actual text.
    pub reference_nll: f64,
    /// The same for the candidate.
    pub candidate_nll: f64,
    /// Sum over positions of KL(reference ‖ candidate).
    pub kl: f64,
    /// Positions where both models' most likely token is the same.
    pub same_top1: usize,
}

impl Quality {
    /// Adds one window. `reference` and `candidate` are log-probability
    /// rows for the window's positions; position `t` predicts `tokens[t + 1]`.
    pub fn add(&mut self, reference: &[f32], candidate: &[f32], tokens: &[u32], vocab: usize) {
        let rows = reference
            .chunks_exact(vocab)
            .zip(candidate.chunks_exact(vocab));
        for (t, (r, c)) in rows.enumerate().take(tokens.len() - 1) {
            let next = tokens[t + 1] as usize;
            self.reference_nll -= f64::from(r[next]);
            self.candidate_nll -= f64::from(c[next]);
            // KL(p ‖ q) = Σ p · (log p − log q), with p the reference.
            self.kl += r
                .iter()
                .zip(c)
                .map(|(&lp, &lq)| f64::from(lp.exp()) * f64::from(lp - lq))
                .sum::<f64>();
            if argmax(r) == argmax(c) {
                self.same_top1 += 1;
            }
            self.tokens += 1;
        }
    }

    /// `exp(mean negative log-likelihood)`: the average number of tokens
    /// the model is "choosing between" at each step. Lower is better.
    pub fn reference_perplexity(&self) -> f64 {
        (self.reference_nll / self.tokens as f64).exp()
    }

    pub fn candidate_perplexity(&self) -> f64 {
        (self.candidate_nll / self.tokens as f64).exp()
    }

    /// Mean KL divergence per token, in nats.
    pub fn mean_kl(&self) -> f64 {
        self.kl / self.tokens as f64
    }

    pub fn top1_agreement(&self) -> f64 {
        self.same_top1 as f64 / self.tokens as f64
    }
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0
}
