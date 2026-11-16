//! What clients experienced, as numbers: time to first token, time per
//! output token, the gaps between tokens, and how many requests met a
//! service-level objective.

use std::time::Duration;

/// What one client saw, measured on the client's side.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// When the request was sent, from the start of the run.
    pub sent: Duration,
    /// When each token arrived, from `sent`.
    pub tokens: Vec<Duration>,
    /// When the last event arrived, from `sent`.
    pub done: Duration,
    pub prompt_tokens: usize,
    /// The engine refused the request instead of serving it.
    pub rejected: bool,
}

impl Trace {
    /// Time to first token.
    pub fn ttft(&self) -> Option<Duration> {
        self.tokens.first().copied()
    }

    /// Time per output token after the first: the average gap.
    pub fn tpot(&self) -> Option<Duration> {
        let n = self.tokens.len();
        (n >= 2).then(|| self.tokens[n - 1].saturating_sub(self.tokens[0]) / (n as u32 - 1))
    }

    /// Every gap between consecutive tokens (inter-token latencies).
    pub fn gaps(&self) -> impl Iterator<Item = Duration> + '_ {
        self.tokens.windows(2).map(|w| w[1].saturating_sub(w[0]))
    }
}

/// A service-level objective: the latencies a request must stay within to
/// count as served well.
#[derive(Clone, Copy, Debug)]
pub struct Slo {
    pub ttft: Duration,
    pub tpot: Duration,
}

impl Slo {
    /// Served, with its first token and its average gap within the targets.
    pub fn met(&self, t: &Trace) -> bool {
        !t.rejected
            && t.ttft().is_some_and(|x| x <= self.ttft)
            && t.tpot().is_none_or(|x| x <= self.tpot)
    }
}

/// The `p`-th percentile (`0 < p <= 100`) of `sorted`, by the nearest-rank
/// method: the smallest value with at least `p`% of all values at or below
/// it. Always one of the values, never an interpolation.
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty(), "percentile of nothing");
    assert!(p > 0.0 && p <= 100.0, "percentile {p} out of range");
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The usual summary of a latency distribution.
#[derive(Clone, Copy, Debug, Default)]
pub struct Percentiles {
    pub n: usize,
    pub p50: Duration,
    pub p90: Duration,
    pub p99: Duration,
    pub max: Duration,
}

impl Percentiles {
    pub fn of(mut values: Vec<Duration>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_unstable();
        Self {
            n: values.len(),
            p50: percentile(&values, 50.0),
            p90: percentile(&values, 90.0),
            p99: percentile(&values, 99.0),
            max: values[values.len() - 1],
        }
    }
}

/// A run, summarized.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub sent: usize,
    pub served: usize,
    pub rejected: usize,
    /// Served within the SLO.
    pub good: usize,
    /// From the first request sent to the last one finished.
    pub elapsed: Duration,
    pub output_tokens: usize,
    pub ttft: Percentiles,
    /// One value per request: its average gap between tokens.
    pub tpot: Percentiles,
    /// One value per gap, all requests together.
    pub itl: Percentiles,
}

impl Report {
    pub fn new(traces: &[Trace], slo: &Slo) -> Self {
        let served: Vec<&Trace> = traces.iter().filter(|t| !t.rejected).collect();
        let first = traces.iter().map(|t| t.sent).min().unwrap_or_default();
        let last = traces
            .iter()
            .map(|t| t.sent + t.done)
            .max()
            .unwrap_or_default();
        Self {
            sent: traces.len(),
            served: served.len(),
            rejected: traces.len() - served.len(),
            good: traces.iter().filter(|t| slo.met(t)).count(),
            elapsed: last.saturating_sub(first),
            output_tokens: served.iter().map(|t| t.tokens.len()).sum(),
            ttft: Percentiles::of(served.iter().filter_map(|t| t.ttft()).collect()),
            tpot: Percentiles::of(served.iter().filter_map(|t| t.tpot()).collect()),
            itl: Percentiles::of(served.iter().flat_map(|t| t.gaps()).collect()),
        }
    }

    /// Output tokens per second over the whole run.
    pub fn throughput(&self) -> f64 {
        self.output_tokens as f64 / self.elapsed.as_secs_f64()
    }

    /// The share of sent requests that met the SLO.
    pub fn attainment(&self) -> f64 {
        self.good as f64 / self.sent.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    #[test]
    fn nearest_rank_percentiles() {
        let v: Vec<Duration> = (1..=10).map(ms).collect();
        assert_eq!(percentile(&v, 50.0), ms(5));
        assert_eq!(percentile(&v, 90.0), ms(9));
        assert_eq!(percentile(&v, 91.0), ms(10));
        assert_eq!(percentile(&v, 100.0), ms(10));
        // With fewer than 100 values, the 99th percentile is the maximum.
        assert_eq!(percentile(&v, 99.0), ms(10));
        assert_eq!(percentile(&[ms(7)], 1.0), ms(7));
    }

    #[test]
    fn a_trace_gives_ttft_tpot_and_gaps() {
        let t = Trace {
            tokens: vec![ms(100), ms(110), ms(130), ms(160)],
            ..Trace::default()
        };
        assert_eq!(t.ttft(), Some(ms(100)));
        assert_eq!(t.tpot(), Some(ms(20)));
        assert_eq!(t.gaps().collect::<Vec<_>>(), [ms(10), ms(20), ms(30)]);
        let one = Trace {
            tokens: vec![ms(50)],
            ..Trace::default()
        };
        assert_eq!(one.tpot(), None);
    }

    #[test]
    fn the_slo_counts_rejections_and_both_latencies() {
        let slo = Slo {
            ttft: ms(200),
            tpot: ms(20),
        };
        let fast = Trace {
            tokens: vec![ms(150), ms(160), ms(175)],
            ..Trace::default()
        };
        let slow_start = Trace {
            tokens: vec![ms(250), ms(260)],
            ..Trace::default()
        };
        let slow_tokens = Trace {
            tokens: vec![ms(100), ms(150)],
            ..Trace::default()
        };
        let rejected = Trace {
            rejected: true,
            ..Trace::default()
        };
        assert!(slo.met(&fast));
        assert!(!slo.met(&slow_start));
        assert!(!slo.met(&slow_tokens));
        assert!(!slo.met(&rejected));
        let report = Report::new(&[fast, slow_start, slow_tokens, rejected], &slo);
        assert_eq!((report.sent, report.served, report.good), (4, 3, 1));
        assert_eq!(report.output_tokens, 7);
        assert_eq!(report.ttft.p50, ms(150));
        assert_eq!(report.itl.n, 4);
    }
}
