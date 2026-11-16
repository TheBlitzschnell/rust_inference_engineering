//! Deciding which draft tokens to keep.
//!
//! The target model has scored the draft: for draft tokens `d[0..k]` it
//! returned `k + 1` rows of logits, row `i` being its prediction for the
//! position of `d[i]` (and row `k` for the position after the last one).
//! Two rules decide how many draft tokens to accept and which token comes
//! next:
//!
//! - **Greedy:** keep draft tokens while each equals the target's most
//!   likely token; the first disagreement is replaced by the target's
//!   choice. The output is exactly what the target alone would produce.
//! - **Rejection sampling:** keep `d[i]` with probability
//!   `min(1, p(d[i]) / q(d[i]))`, `p` being the target's distribution and
//!   `q` the one the draft token was drawn from; on rejection, draw from
//!   the leftover `max(0, p − q)`, renormalized. The output then has
//!   exactly the target's distribution (Leviathan et al., Chen et al.,
//!   2023).
//!
//! Either way, at least one token comes out of every verification: the
//! replacement for the first rejected draft token, or, when all are
//! accepted, one more from the last row.

use ch15_sampling::Rng;

/// How a draft token was chosen, for rejection sampling.
#[derive(Clone, Debug, PartialEq)]
pub enum Proposal {
    /// Drawn from this distribution over the vocabulary.
    Sampled(Vec<f32>),
    /// Chosen without randomness (an n-gram lookup): `q` puts all its
    /// probability on the token.
    Exact,
}

impl Proposal {
    /// The probability `q` gave `token`, for a proposal of `proposed`.
    fn q(&self, proposed: u32, token: u32) -> f32 {
        match self {
            Self::Sampled(q) => q[token as usize],
            Self::Exact => f32::from(u8::from(token == proposed)),
        }
    }
}

/// The index of the largest value (the first one, on ties).
pub fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Softmax of `logits / temperature`: the distribution a temperature
/// sampler draws from.
pub fn probabilities(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

/// Draws a token from `p` (which sums to 1, up to rounding).
pub fn draw(p: &[f32], rng: &mut Rng) -> u32 {
    let mut u = rng.next_f64() as f32;
    for (i, &v) in p.iter().enumerate() {
        if u < v {
            return i as u32;
        }
        u -= v;
    }
    // Rounding left a little probability unspent: take the last token that
    // had any.
    p.iter().rposition(|&v| v > 0.0).unwrap_or(0) as u32
}

/// Greedy verification of `draft` against `logits` (`draft.len() + 1`
/// rows of `vocab`). Returns how many draft tokens to keep and the token
/// that follows them.
pub fn verify_greedy(draft: &[u32], logits: &[f32], vocab: usize) -> (usize, u32) {
    let rows: Vec<&[f32]> = logits.chunks_exact(vocab).collect();
    assert_eq!(
        rows.len(),
        draft.len() + 1,
        "one row per draft token, plus one"
    );
    for (i, &d) in draft.iter().enumerate() {
        let best = argmax(rows[i]);
        if best != d {
            return (i, best);
        }
    }
    (draft.len(), argmax(rows[draft.len()]))
}

/// Rejection-sampling verification. `p[i]` is the target's distribution
/// for the position of `draft[i]` (`p[draft.len()]` for the one after),
/// `q[i]` how `draft[i]` was proposed. Returns how many draft tokens to
/// keep and the token that follows them.
pub fn verify_sampled(
    draft: &[u32],
    q: &[Proposal],
    p: &[Vec<f32>],
    rng: &mut Rng,
) -> (usize, u32) {
    assert_eq!(q.len(), draft.len(), "one proposal per draft token");
    assert_eq!(
        p.len(),
        draft.len() + 1,
        "one distribution per draft token, plus one"
    );
    for (i, &d) in draft.iter().enumerate() {
        let (pd, qd) = (p[i][d as usize], q[i].q(d, d));
        // Accept with probability min(1, p/q).
        if (rng.next_f64() as f32) * qd < pd {
            continue;
        }
        // Rejected: draw from what p has and q lacks.
        let mut residual: Vec<f32> = p[i]
            .iter()
            .enumerate()
            .map(|(t, &pt)| (pt - q[i].q(d, t as u32)).max(0.0))
            .collect();
        let sum: f32 = residual.iter().sum();
        if sum <= 0.0 {
            // p and q agree to rounding: rejection was a rounding accident.
            return (i, draw(&p[i], rng));
        }
        for v in &mut residual {
            *v /= sum;
        }
        return (i, draw(&residual, rng));
    }
    (draft.len(), draw(&p[draft.len()], rng))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_keeps_the_agreeing_prefix() {
        let vocab = 4;
        // Target's choices per row: 2, 0, 3, 1.
        let logits = [
            0.0, 0.1, 0.9, 0.2, //
            0.8, 0.1, 0.0, 0.2, //
            0.0, 0.1, 0.2, 0.7, //
            0.1, 0.9, 0.0, 0.2,
        ];
        assert_eq!(verify_greedy(&[2, 0, 3], &logits, vocab), (3, 1));
        assert_eq!(verify_greedy(&[2, 1, 3], &logits, vocab), (1, 0));
        assert_eq!(verify_greedy(&[3, 0, 3], &logits, vocab), (0, 2));
        assert_eq!(verify_greedy(&[], &logits[..4], vocab), (0, 2));
    }

    /// The first token out of a verification, over many trials.
    fn first_tokens(q: &[f32], p: &[f32], exact: Option<u32>, trials: usize) -> Vec<f64> {
        let mut rng = Rng::new(42);
        let mut counts = vec![0usize; p.len()];
        for _ in 0..trials {
            let (d, proposal) = match exact {
                Some(t) => (t, Proposal::Exact),
                None => (draw(q, &mut rng), Proposal::Sampled(q.to_vec())),
            };
            let (kept, next) =
                verify_sampled(&[d], &[proposal], &[p.to_vec(), vec![0.25; 4]], &mut rng);
            counts[if kept == 1 { d } else { next } as usize] += 1;
        }
        counts.iter().map(|&c| c as f64 / trials as f64).collect()
    }

    #[test]
    fn rejection_sampling_produces_the_target_distribution() {
        let p = [0.5, 0.3, 0.15, 0.05];
        for q in [
            [0.25, 0.25, 0.25, 0.25],
            [0.05, 0.15, 0.3, 0.5],
            [0.5, 0.3, 0.15, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ] {
            let got = first_tokens(&q, &p, None, 200_000);
            for (g, want) in got.iter().zip(p) {
                assert!((g - f64::from(want)).abs() < 0.005, "q {q:?}: {got:?}");
            }
        }
        // A deterministic proposal of token 3, then of token 0.
        for t in [3, 0] {
            let got = first_tokens(&[], &p, Some(t), 200_000);
            for (g, want) in got.iter().zip(p) {
                assert!((g - f64::from(want)).abs() < 0.005, "exact {t}: {got:?}");
            }
        }
    }

    #[test]
    fn a_draft_from_the_target_distribution_is_always_accepted() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let d = draw(&p, &mut rng);
            let q = Proposal::Sampled(p.clone());
            let (kept, _) = verify_sampled(&[d], &[q], &[p.clone(), p.clone()], &mut rng);
            assert_eq!(kept, 1);
        }
    }

    #[test]
    fn probabilities_follow_the_temperature() {
        let p = probabilities(&[1.0, 2.0, 3.0], 1.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
        let sharp = probabilities(&[1.0, 2.0, 3.0], 0.1);
        assert!(sharp[2] > 0.999);
    }
}
