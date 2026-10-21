//! Load generators: clients that send requests to an engine and record
//! what they see.
//!
//! - **Open loop** ([`run_open`]): requests are sent at times fixed in
//!   advance, whatever the engine is doing, as independent users do. The
//!   times usually come from a Poisson process ([`poisson_arrivals`]).
//! - **Closed loop** ([`run_closed`]): a fixed number of users, each
//!   sending its next request when its previous one is done. The load
//!   adapts to the server: when it slows down, they send less.
//!
//! Both run every client as a task on one thread (a single-threaded tokio
//! runtime), so hundreds of clients cost one thread, and each token is
//! timestamped when the client receives it.

use crate::metrics::Trace;
use ch15_sampling::{Rng, SamplingParams};
use ch21_engine_thread::{EngineHandle, Event, Request};
use futures_util::future::join_all;
use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

/// One request of a workload: when to send it, and what it asks for.
#[derive(Clone, Debug)]
pub struct Planned {
    /// From the start of the run (ignored by closed-loop clients).
    pub at: Duration,
    pub prompt: Vec<u32>,
    /// Exactly this many tokens are generated: there are no stop tokens,
    /// so the work per request is known.
    pub max_tokens: usize,
}

/// The first `n` arrival times of a Poisson process with `rate` arrivals
/// per second: the gaps between arrivals are independent and
/// exponentially distributed, with mean `1 / rate`.
pub fn poisson_arrivals(rate: f64, n: usize, seed: u64) -> Vec<Duration> {
    let mut rng = Rng::new(seed);
    let mut t = 0.0;
    (0..n)
        .map(|_| {
            // Inverse transform: if u is uniform in (0, 1], -ln(u) / rate
            // is exponential with mean 1 / rate.
            let u = 1.0 - rng.next_f64();
            t += -u.ln() / rate;
            Duration::from_secs_f64(t)
        })
        .collect()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("building the client runtime")
}

/// Sends one request, waits for its last event, and records the arrival
/// time of every token.
async fn client(engine: &EngineHandle, p: &Planned, start: Instant) -> Trace {
    let sent = Instant::now();
    let mut trace = Trace {
        sent: sent - start,
        prompt_tokens: p.prompt.len(),
        ..Trace::default()
    };
    let request = Request {
        prompt: p.prompt.clone(),
        params: SamplingParams::greedy(),
        max_tokens: p.max_tokens,
        stop_tokens: Vec::new(),
    };
    match engine.submit(request) {
        Ok(mut events) => {
            while let Some(event) = events.recv().await {
                match event {
                    Event::Token(_) => trace.tokens.push(sent.elapsed()),
                    Event::Done(_) => break,
                    Event::Rejected(_) => {
                        trace.rejected = true;
                        break;
                    }
                }
            }
        }
        Err(_) => trace.rejected = true,
    }
    trace.done = sent.elapsed();
    trace
}

/// Open loop: sends each planned request at its time, and returns what
/// each client saw, in the order of `plan`. Blocks until all are done.
pub fn run_open(engine: &EngineHandle, plan: &[Planned]) -> Vec<Trace> {
    runtime().block_on(async {
        let start = Instant::now();
        let clients = plan.iter().map(|p| async move {
            tokio::time::sleep_until((start + p.at).into()).await;
            client(engine, p, start).await
        });
        join_all(clients).await
    })
}

/// Closed loop: `users` clients, each sending the next request of `plan`
/// as soon as its previous one is done (after `think` time), until `plan`
/// is used up. Returns the traces in the order of `plan`.
pub fn run_closed(
    engine: &EngineHandle,
    users: usize,
    plan: &[Planned],
    think: Duration,
) -> Vec<Trace> {
    runtime().block_on(async {
        let start = Instant::now();
        let next = Cell::new(0);
        let traces = RefCell::new(vec![Trace::default(); plan.len()]);
        let user = || async {
            loop {
                let i = next.get();
                if i == plan.len() {
                    break;
                }
                next.set(i + 1);
                let trace = client(engine, &plan[i], start).await;
                traces.borrow_mut()[i] = trace;
                tokio::time::sleep(think).await;
            }
        };
        join_all((0..users).map(|_| user())).await;
        traces.into_inner()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_arrivals_have_the_right_rate_and_spread() {
        let rate = 4.0;
        let n = 20_000;
        let at = poisson_arrivals(rate, n, 1);
        assert!(at.windows(2).all(|w| w[0] <= w[1]));
        let gaps: Vec<f64> = std::iter::once(at[0])
            .chain(at.windows(2).map(|w| w[1].saturating_sub(w[0])))
            .map(|d| d.as_secs_f64())
            .collect();
        let mean = gaps.iter().sum::<f64>() / n as f64;
        assert!((mean - 1.0 / rate).abs() < 0.01, "mean gap {mean}");
        // Exponential gaps: about e^-1 = 37% are longer than the mean, and
        // the standard deviation equals the mean.
        let longer = gaps.iter().filter(|&&g| g > 1.0 / rate).count() as f64 / n as f64;
        assert!((longer - (-1.0f64).exp()).abs() < 0.02, "{longer}");
        let var = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((var.sqrt() / mean - 1.0).abs() < 0.05);
    }
}
