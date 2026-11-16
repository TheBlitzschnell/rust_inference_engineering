//! Chapter 30: inference in production.
//!
//! - [`trace`]: layer-by-layer comparison of hidden states, to find where
//!   two implementations of a model part.
//! - [`metrics`]: Prometheus-style counters, gauges and histograms, and
//!   quantiles estimated from buckets.
//! - [`capacity`]: replicas for a forecast load, with the M/M/c queue.
//! - [`outage`]: a simulation of timeouts and retries around a stall, and
//!   the policies that keep it from becoming a lasting outage.

pub mod capacity;
pub mod metrics;
pub mod outage;
pub mod trace;
