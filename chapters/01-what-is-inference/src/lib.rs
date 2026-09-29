//! Chapter 1: a tiny model, and the numbers that say how fast it runs.
//!
//! The model is a single linear layer, `y = W · x`. That one operation is
//! where large neural networks spend most of their time, so even this toy
//! shows the effects an inference engineer deals with every day: latency,
//! throughput, batching, and the cost of moving memory around.

use std::time::Duration;

/// A single linear layer.
///
/// `weights` holds `out_dim` rows of `in_dim` numbers each, one row after
/// another ("row-major"). Row `o` is the set of weights that produces
/// output number `o`.
#[derive(Clone)]
pub struct LinearModel {
    weights: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl LinearModel {
    /// Builds a model with made-up but deterministic weights.
    ///
    /// Real weights come from training. Here we only need numbers that are
    /// not all equal, so the computation cannot be skipped.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = (0..in_dim * out_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        Self {
            weights,
            in_dim,
            out_dim,
        }
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// How many bytes of weights one full pass over the model reads.
    pub fn weight_bytes(&self) -> usize {
        self.weights.len() * size_of::<f32>()
    }

    /// Floating-point operations for one input: one multiply and one add
    /// per weight.
    pub fn flops_per_input(&self) -> usize {
        2 * self.weights.len()
    }

    /// Serves one request: `out = W · x`.
    ///
    /// Both arguments are borrowed. The model is read, never copied, and the
    /// caller owns the output buffer so it can be reused across requests.
    pub fn predict(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim, "input has the wrong length");
        assert_eq!(out.len(), self.out_dim, "output has the wrong length");
        for (row, y) in self.weights.chunks_exact(self.in_dim).zip(out.iter_mut()) {
            *y = dot(row, x);
        }
    }

    /// Serves `batch` requests at once.
    ///
    /// `xs` holds the inputs one after another, and `out` receives the
    /// outputs the same way. The loop order is the point: each weight row is
    /// fetched from memory once and then used for every input in the batch
    /// while it is still sitting in the CPU cache.
    pub fn predict_batch(&self, xs: &[f32], batch: usize, out: &mut [f32]) {
        assert_eq!(
            xs.len(),
            batch * self.in_dim,
            "inputs have the wrong length"
        );
        assert_eq!(
            out.len(),
            batch * self.out_dim,
            "outputs have the wrong length"
        );
        for (o, row) in self.weights.chunks_exact(self.in_dim).enumerate() {
            for (b, x) in xs.chunks_exact(self.in_dim).enumerate() {
                out[b * self.out_dim + o] = dot(row, x);
            }
        }
    }
}

/// Serves one request after being handed its own copy of the model.
///
/// This is the mistake the borrow checker makes visible: a function that
/// takes `LinearModel` by value forces every caller to either give up their
/// model or write `.clone()`, which copies every weight.
#[expect(
    clippy::needless_pass_by_value,
    reason = "this function exists to demonstrate the cost of taking ownership"
)]
pub fn predict_with_owned_model(model: LinearModel, x: &[f32], out: &mut [f32]) {
    model.predict(x, out);
    // `model` is dropped here: its weights are freed at the end of every call.
}

/// Dot product of two equal-length slices.
///
/// It keeps eight running sums instead of one. With a single sum, every
/// addition has to wait for the previous one to finish; with eight, the CPU
/// can work on several at once. Chapter 6 explains this properly.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for (x, y) in a8.iter().zip(b8) {
        for lane in 0..8 {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        total += x * y;
    }
    total
}

/// A summary of many latency measurements.
#[derive(Debug, Clone, Copy)]
pub struct LatencySummary {
    pub count: usize,
    pub mean: Duration,
    pub p50: Duration,
    pub p90: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Sorts the samples and reads off the percentiles.
pub fn summarize(samples: &mut [Duration]) -> LatencySummary {
    assert!(!samples.is_empty(), "need at least one sample");
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    LatencySummary {
        count: samples.len(),
        mean: total / samples.len() as u32,
        p50: percentile(samples, 50.0),
        p90: percentile(samples, 90.0),
        p99: percentile(samples, 99.0),
        max: samples[samples.len() - 1],
    }
}

/// The value below which `p` percent of the (sorted) samples fall.
///
/// Uses the "nearest rank" definition: the smallest sample such that at
/// least `p` percent of samples are less than or equal to it.
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty());
    assert!((0.0..=100.0).contains(&p));
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_the_obvious_loop() {
        for len in [0, 1, 7, 8, 9, 31, 100] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| 1.0 - i as f32 * 0.25).collect();
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let got = dot(&a, &b);
            assert!((got - expected).abs() <= 1e-3 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn batch_gives_the_same_answers_as_one_at_a_time() {
        let model = LinearModel::new(33, 5);
        let batch = 4;
        let xs: Vec<f32> = (0..batch * 33).map(|i| (i % 7) as f32 - 3.0).collect();
        let mut batched = vec![0.0; batch * 5];
        model.predict_batch(&xs, batch, &mut batched);
        for b in 0..batch {
            let mut single = vec![0.0; 5];
            model.predict(&xs[b * 33..(b + 1) * 33], &mut single);
            assert_eq!(&batched[b * 5..(b + 1) * 5], &single[..]);
        }
    }

    #[test]
    fn owned_model_gives_the_same_answer() {
        let model = LinearModel::new(16, 4);
        let x = vec![1.0; 16];
        let (mut a, mut b) = (vec![0.0; 4], vec![0.0; 4]);
        model.predict(&x, &mut a);
        predict_with_owned_model(model.clone(), &x, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let mut samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        samples.reverse();
        let s = summarize(&mut samples);
        assert_eq!(s.p50, Duration::from_millis(50));
        assert_eq!(s.p90, Duration::from_millis(90));
        assert_eq!(s.p99, Duration::from_millis(99));
        assert_eq!(s.max, Duration::from_millis(100));
        assert_eq!(percentile(&samples, 0.0), Duration::from_millis(1));
    }
}
