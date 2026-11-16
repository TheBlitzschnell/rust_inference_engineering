//! Metrics in the Prometheus format: counters, gauges and histograms that
//! a server exposes as text at `GET /metrics`, and the quantile estimate a
//! monitoring system computes from a histogram's buckets.
//!
//! A server cannot keep every request's latency forever; it keeps counts.
//! A histogram counts observations per bucket (`le`: less than or equal to
//! the bucket's upper bound), and the monitoring system estimates a
//! percentile from those counts by assuming observations are spread evenly
//! within a bucket. The estimate is only as fine as the buckets.

use std::collections::BTreeMap;
use std::fmt::Write;

/// Upper bounds of buckets, in seconds; observations above the last go in
/// an implicit `+Inf` bucket.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub bounds: Vec<f64>,
    /// `counts[i]`: observations in bucket `i` (not cumulative); the last
    /// entry is the `+Inf` bucket.
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    pub fn new(bounds: &[f64]) -> Self {
        assert!(
            bounds.windows(2).all(|w| w[0] < w[1]),
            "bounds must increase"
        );
        Self {
            bounds: bounds.to_vec(),
            counts: vec![0; bounds.len() + 1],
            sum: 0.0,
            count: 0,
        }
    }

    pub fn observe(&mut self, v: f64) {
        let i = self.bounds.partition_point(|&b| b < v);
        self.counts[i] += 1;
        self.sum += v;
        self.count += 1;
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// The `q`-quantile (`0 < q < 1`) estimated from the buckets, as
    /// Prometheus's `histogram_quantile` does: find the bucket holding the
    /// rank `q · count`, and interpolate linearly inside it. A rank in the
    /// `+Inf` bucket returns the largest finite bound.
    pub fn quantile(&self, q: f64) -> f64 {
        let rank = q * self.count as f64;
        let mut below = 0u64;
        for (i, &c) in self.counts.iter().enumerate() {
            if (below + c) as f64 >= rank && c > 0 {
                let Some(&upper) = self.bounds.get(i) else {
                    return *self.bounds.last().expect("at least one bound");
                };
                let lower = if i == 0 { 0.0 } else { self.bounds[i - 1] };
                return lower + (upper - lower) * (rank - below as f64) / c as f64;
            }
            below += c;
        }
        f64::NAN
    }
}

/// Counters, gauges and histograms, by name and label set.
#[derive(Default)]
pub struct Registry {
    help: BTreeMap<String, (String, &'static str)>,
    values: BTreeMap<(String, String), f64>,
    histograms: BTreeMap<String, Histogram>,
}

impl Registry {
    fn declare(&mut self, name: &str, kind: &'static str, help: &str) {
        self.help
            .entry(name.to_string())
            .or_insert_with(|| (help.to_string(), kind));
    }

    /// Adds `by` to a counter (which only goes up).
    pub fn inc(&mut self, name: &str, labels: &str, by: f64, help: &str) {
        self.declare(name, "counter", help);
        *self
            .values
            .entry((name.to_string(), labels.to_string()))
            .or_default() += by;
    }

    /// Sets a gauge (a value that goes up and down).
    pub fn set(&mut self, name: &str, labels: &str, v: f64, help: &str) {
        self.declare(name, "gauge", help);
        self.values
            .insert((name.to_string(), labels.to_string()), v);
    }

    /// Records an observation in a histogram with these bucket bounds.
    pub fn observe(&mut self, name: &str, bounds: &[f64], v: f64, help: &str) {
        self.declare(name, "histogram", help);
        self.histograms
            .entry(name.to_string())
            .or_insert_with(|| Histogram::new(bounds))
            .observe(v);
    }

    pub fn histogram(&self, name: &str) -> Option<&Histogram> {
        self.histograms.get(name)
    }

    /// The Prometheus text exposition format.
    pub fn exposition(&self) -> String {
        let mut out = String::new();
        for (name, (help, kind)) in &self.help {
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} {kind}");
            for ((n, labels), v) in &self.values {
                if n == name {
                    let labels = if labels.is_empty() {
                        String::new()
                    } else {
                        format!("{{{labels}}}")
                    };
                    let _ = writeln!(out, "{name}{labels} {v}");
                }
            }
            if let Some(h) = self.histograms.get(name) {
                let mut cumulative = 0;
                for (i, c) in h.counts.iter().enumerate() {
                    cumulative += c;
                    let le = h.bounds.get(i).map_or("+Inf".to_string(), f64::to_string);
                    let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {cumulative}");
                }
                let _ = writeln!(out, "{name}_sum {}", h.sum);
                let _ = writeln!(out, "{name}_count {}", h.count);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_interpolate_within_a_bucket() {
        let mut h = Histogram::new(&[1.0, 2.0, 4.0]);
        for v in [0.5, 1.5, 1.5, 1.5, 3.0, 10.0] {
            h.observe(v);
        }
        // Rank 3 of 6 is the second of the three values in (1, 2].
        assert!((h.quantile(0.5) - (1.0 + 2.0 / 3.0)).abs() < 1e-12);
        // Rank 5.94 falls in +Inf: the largest finite bound.
        assert!((h.quantile(0.99) - 4.0).abs() < 1e-12);
        // A value on a bound goes in that bound's bucket (le).
        let mut h = Histogram::new(&[1.0, 2.0]);
        h.observe(1.0);
        assert!(h.quantile(0.5) <= 1.0);
    }

    #[test]
    fn the_exposition_has_cumulative_buckets() {
        let mut r = Registry::default();
        r.inc("requests_total", "status=\"200\"", 3.0, "Requests served.");
        r.inc("requests_total", "status=\"429\"", 1.0, "Requests served.");
        r.set("queue_depth", "", 2.0, "Requests waiting.");
        for v in [0.1, 0.3, 0.3] {
            r.observe("ttft_seconds", &[0.25, 0.5], v, "Time to first token.");
        }
        let text = r.exposition();
        for line in [
            "# TYPE requests_total counter",
            "requests_total{status=\"200\"} 3",
            "requests_total{status=\"429\"} 1",
            "# TYPE queue_depth gauge",
            "queue_depth 2",
            "# TYPE ttft_seconds histogram",
            "ttft_seconds_bucket{le=\"0.25\"} 1",
            "ttft_seconds_bucket{le=\"0.5\"} 3",
            "ttft_seconds_bucket{le=\"+Inf\"} 3",
            "ttft_seconds_count 3",
        ] {
            assert!(
                text.lines().any(|l| l == line),
                "missing {line:?} in\n{text}"
            );
        }
    }
}
