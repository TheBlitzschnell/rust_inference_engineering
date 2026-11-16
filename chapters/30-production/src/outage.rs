//! A short stall, timeouts and retries: how a server can stay down after
//! the cause of an outage is gone.
//!
//! Clients send requests at a steady rate and give up on an attempt after a
//! timeout, then retry. The server completes up to `capacity` requests per
//! second from a FIFO queue, and stalls completely during the outage.
//! During the stall the queue fills with attempts whose clients will have
//! given up by the time they are served; after it, the server spends its
//! capacity on that stale work while the clients keep adding retries, so
//! every request again waits longer than the timeout. The overload sustains
//! itself: a **metastable failure**. The remedies are the policies below.

use ch15_sampling::Rng;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// When a client retries an attempt that timed out or was rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Retry {
    /// At once.
    Immediately,
    /// After a random delay in `[0, min(cap, base · 2^k))` for the `k`-th
    /// retry: exponential backoff with "full jitter".
    Backoff { base: f64, cap: f64 },
}

/// One simulated situation.
#[derive(Clone, Debug)]
pub struct Scenario {
    /// New requests per second.
    pub rate: f64,
    /// Requests the server completes per second.
    pub capacity: f64,
    /// Seconds a client waits for one attempt.
    pub timeout: f64,
    pub max_attempts: u32,
    pub retry: Retry,
    /// The server skips queued attempts whose client has already given up
    /// (the client's deadline travels with the request).
    pub drop_expired: bool,
    /// The server rejects attempts when this many are queued.
    pub max_queue: Option<usize>,
    /// The server completes nothing in `[start, end)` seconds.
    pub outage: (f64, f64),
    pub seconds: usize,
    pub seed: u64,
}

/// What happened in one second.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Second {
    /// Attempts sent (new requests and retries).
    pub attempts: usize,
    /// Requests answered within their client's timeout.
    pub good: usize,
    /// Attempts the server completed for clients that had given up.
    pub wasted: usize,
    /// Requests abandoned after their last attempt.
    pub lost: usize,
    /// Queue length at the end of the second.
    pub queue: usize,
}

struct Request {
    attempts: u32,
    done: bool,
}

/// Runs the scenario in steps of 10 ms and returns one record per second.
pub fn simulate(s: &Scenario) -> Vec<Second> {
    const STEPS: usize = 100; // per second
    let dt = 1.0 / STEPS as f64;
    let mut rng = Rng::new(s.seed);
    let mut requests: Vec<Request> = Vec::new();
    // Attempts in the server's queue: (request, client's deadline).
    let mut queue: VecDeque<(usize, f64)> = VecDeque::new();
    // Attempts waiting for an answer, in order of deadline.
    let mut waiting: VecDeque<(usize, f64)> = VecDeque::new();
    // Retries to send: (time in ms, request).
    let mut retries: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let (mut arrivals, mut budget) = (0.0, 0.0);
    let mut out = vec![Second::default(); s.seconds];

    // A request needs another attempt (or is lost) at `now`.
    let retry_or_lose = |r: usize,
                         now: f64,
                         requests: &mut Vec<Request>,
                         rng: &mut Rng,
                         retries: &mut BinaryHeap<Reverse<(u64, usize)>>,
                         sec: &mut Second| {
        let req = &mut requests[r];
        if req.attempts >= s.max_attempts {
            sec.lost += 1;
            req.done = true;
            return;
        }
        let delay = match s.retry {
            Retry::Immediately => 0.0,
            Retry::Backoff { base, cap } => {
                let k = req.attempts - 1;
                rng.next_f64() * cap.min(base * f64::from(1u32 << k.min(20)))
            }
        };
        let at = ((now + delay) * 1000.0).round() as u64;
        retries.push(Reverse((at, r)));
    };

    for step in 0..s.seconds * STEPS {
        let now = step as f64 * dt;
        let sec = &mut out[step / STEPS];
        // New requests, and retries that are due.
        arrivals += s.rate * dt;
        let mut sends: Vec<usize> = Vec::new();
        while arrivals >= 1.0 {
            arrivals -= 1.0;
            requests.push(Request {
                attempts: 0,
                done: false,
            });
            sends.push(requests.len() - 1);
        }
        let now_ms = (now * 1000.0).round() as u64;
        while let Some(&Reverse((at, r))) = retries.peek() {
            if at > now_ms {
                break;
            }
            retries.pop();
            sends.push(r);
        }
        for r in sends {
            requests[r].attempts += 1;
            sec.attempts += 1;
            if s.max_queue.is_some_and(|m| queue.len() >= m) {
                // Rejected at once: the client learns it now.
                retry_or_lose(r, now, &mut requests, &mut rng, &mut retries, sec);
                continue;
            }
            queue.push_back((r, now + s.timeout));
            waiting.push_back((r, now + s.timeout));
        }
        // The server works, unless stalled.
        if now < s.outage.0 || now >= s.outage.1 {
            budget += s.capacity * dt;
            while budget >= 1.0 {
                let Some((r, deadline)) = queue.pop_front() else {
                    budget = 0.0; // idle capacity is not saved up
                    break;
                };
                if s.drop_expired && deadline < now {
                    continue; // skipped without doing the work
                }
                budget -= 1.0;
                if deadline >= now && !requests[r].done {
                    requests[r].done = true;
                    sec.good += 1;
                } else {
                    sec.wasted += 1;
                }
            }
        }
        // Clients whose attempt timed out retry or give up.
        while let Some(&(r, deadline)) = waiting.front() {
            if deadline >= now {
                break;
            }
            waiting.pop_front();
            if !requests[r].done {
                retry_or_lose(r, now, &mut requests, &mut rng, &mut retries, sec);
            }
        }
        sec.queue = queue.len();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(retry: Retry, drop_expired: bool, max_queue: Option<usize>) -> Scenario {
        Scenario {
            rate: 80.0,
            capacity: 100.0,
            timeout: 1.0,
            max_attempts: 4,
            retry,
            drop_expired,
            max_queue,
            outage: (10.0, 15.0),
            seconds: 40,
            seed: 1,
        }
    }

    fn good(run: &[Second], from: usize, to: usize) -> usize {
        run[from..to].iter().map(|s| s.good).sum()
    }

    #[test]
    fn without_an_outage_everything_is_served() {
        let mut s = scenario(Retry::Immediately, false, None);
        s.outage = (0.0, 0.0);
        let run = simulate(&s);
        assert_eq!(good(&run, 1, 40), 80 * 39);
        assert!(run.iter().all(|s| s.wasted == 0 && s.lost == 0));
    }

    #[test]
    fn immediate_retries_stay_down_and_dropping_expired_work_recovers() {
        let stuck = simulate(&scenario(Retry::Immediately, false, None));
        assert_eq!(
            good(&stuck, 30, 40),
            0,
            "metastable: nothing good at the end"
        );
        let fixed = simulate(&scenario(Retry::Immediately, true, None));
        assert!(good(&fixed, 30, 40) >= 790, "{}", good(&fixed, 30, 40));
    }
}
