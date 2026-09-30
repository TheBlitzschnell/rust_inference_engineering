//! Online softmax: a weighted average over a stream of scores, in one pass.
//!
//! Softmax needs the maximum score before it can exponentiate anything
//! (chapter 8), which seems to require two passes over the scores, and a
//! third to weight the values. Online softmax keeps a running maximum `m`,
//! a running sum `l` of `exp(score − m)` and a running weighted sum `acc`
//! of the values; when a larger score arrives it rescales `l` and `acc` by
//! `exp(m_old − m_new)`. Partial states over different parts of the stream
//! merge the same way, which is what lets threads split a long context.

/// The running state for one query: `acc / l` is the attention output.
#[derive(Clone, Debug, PartialEq)]
pub struct Online {
    pub max: f32,
    pub sum: f32,
    pub acc: Vec<f32>,
}

impl Online {
    pub fn new(dim: usize) -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
            acc: vec![0.0; dim],
        }
    }

    /// Adds a batch of `scores` with their value rows (`values` is
    /// `scores.len() × dim`). Rescales once per batch, not once per score.
    pub fn add(&mut self, scores: &[f32], values: &[f32]) {
        let dim = self.acc.len();
        let batch_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if batch_max == f32::NEG_INFINITY {
            return;
        }
        let new_max = self.max.max(batch_max);
        let rescale = (self.max - new_max).exp();
        self.sum *= rescale;
        for a in &mut self.acc {
            *a *= rescale;
        }
        for (&s, v) in scores.iter().zip(values.chunks_exact(dim)) {
            let p = (s - new_max).exp();
            self.sum += p;
            for (a, &x) in self.acc.iter_mut().zip(v) {
                *a += p * x;
            }
        }
        self.max = new_max;
    }

    /// Combines two states over disjoint parts of the same stream.
    pub fn merge(&mut self, other: &Online) {
        if other.max == f32::NEG_INFINITY {
            return;
        }
        let new_max = self.max.max(other.max);
        let (a, b) = ((self.max - new_max).exp(), (other.max - new_max).exp());
        self.sum = self.sum * a + other.sum * b;
        for (x, &y) in self.acc.iter_mut().zip(&other.acc) {
            *x = *x * a + y * b;
        }
        self.max = new_max;
    }

    /// The weighted average of the values: the attention output.
    pub fn finish(&self, out: &mut [f32]) {
        let inv = 1.0 / self.sum;
        for (o, &a) in out.iter_mut().zip(&self.acc) {
            *o = a * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax_average(scores: &[f32], values: &[f32], dim: usize) -> Vec<f32> {
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f64> = scores.iter().map(|&s| f64::from(s - max).exp()).collect();
        let total: f64 = w.iter().sum();
        (0..dim)
            .map(|j| {
                (w.iter()
                    .zip(values.chunks_exact(dim))
                    .map(|(p, v)| p * f64::from(v[j]))
                    .sum::<f64>()
                    / total) as f32
            })
            .collect()
    }

    #[test]
    fn one_pass_in_pieces_equals_the_three_pass_softmax() {
        let dim = 4;
        let scores: Vec<f32> = (0..37)
            .map(|i| ((i * 17 % 23) as f32 - 11.0) * 0.7)
            .collect();
        let values: Vec<f32> = (0..37 * dim).map(|i| (i % 7) as f32 - 3.0).collect();
        let want = softmax_average(&scores, &values, dim);
        for piece in [1, 5, 16, 37] {
            let mut o = Online::new(dim);
            for (s, v) in scores.chunks(piece).zip(values.chunks(piece * dim)) {
                o.add(s, v);
            }
            let mut got = vec![0.0; dim];
            o.finish(&mut got);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "piece {piece}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn merging_split_states_equals_one_state() {
        let dim = 3;
        let scores: Vec<f32> = (0..20).map(|i| (i as f32 * 1.3).sin() * 20.0).collect();
        let values: Vec<f32> = (0..20 * dim).map(|i| (i as f32).cos()).collect();
        let mut whole = Online::new(dim);
        whole.add(&scores, &values);
        let (mut left, mut right) = (Online::new(dim), Online::new(dim));
        left.add(&scores[..7], &values[..7 * dim]);
        right.add(&scores[7..], &values[7 * dim..]);
        left.merge(&right);
        let (mut a, mut b) = (vec![0.0; dim], vec![0.0; dim]);
        whole.finish(&mut a);
        left.finish(&mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn huge_scores_do_not_overflow() {
        let mut o = Online::new(1);
        o.add(&[1000.0, 999.0], &[1.0, 3.0]);
        let mut out = [0.0];
        o.finish(&mut out);
        assert!(out[0].is_finite());
    }
}
