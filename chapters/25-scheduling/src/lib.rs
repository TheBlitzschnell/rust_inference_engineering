//! Chapter 25: scheduling and service-level objectives.
//!
//! - [`metrics`]: per-request traces, percentiles, SLOs and goodput.
//! - [`loadgen`]: open-loop (Poisson) and closed-loop load generators.
//! - [`engine`]: chapter 24's engine with a scheduling policy and a
//!   bounded queue.

pub mod engine;
pub mod loadgen;
pub mod metrics;

pub use engine::{Policy, SchedulerConfig, Stats, spawn};
pub use loadgen::{Planned, poisson_arrivals, run_closed, run_open};
pub use metrics::{Percentiles, Report, Slo, Trace, percentile};

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Model, Scratch};
    use ch15_sampling::SamplingParams;
    use ch21_engine_thread::{EngineHandle, Event, Request, collect};
    use ch24_paged_kv_cache::PagedConfig;
    use std::time::Instant;

    fn model() -> Model<DenseF32> {
        Model::from_reference(&Weights::random(&Config::tiny(), 25))
    }

    fn prompt(len: usize, seed: u32) -> Vec<u32> {
        (0..len as u32).map(|i| (i * 13 + seed * 7) % 97).collect()
    }

    fn request(prompt: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            prompt,
            params: SamplingParams::greedy(),
            max_tokens,
            stop_tokens: Vec::new(),
        }
    }

    fn config(
        max_batch: usize,
        max_step_tokens: usize,
        policy: Policy,
        max_waiting: Option<usize>,
    ) -> SchedulerConfig {
        SchedulerConfig {
            paged: PagedConfig {
                threads: 2,
                max_batch,
                context: 256,
                max_step_tokens,
                num_blocks: 128,
                block_size: 4,
                prefix_caching: false,
            },
            policy,
            max_waiting,
        }
    }

    /// Chapter 14's greedy answer, computed directly.
    fn greedy(prompt: &[u32], n: usize) -> Vec<u32> {
        let model = model();
        let mut pool = SpinPool::new(2);
        let mut cache = KvCache::new(&model.config, 256);
        let mut scratch = Scratch::new(&model.config, 64, 256);
        let all = ch14_kv_cache::generate_greedy(
            &model,
            &mut pool,
            prompt,
            n,
            &mut cache,
            &mut scratch,
            |_, _| {},
        );
        all[prompt.len()..].to_vec()
    }

    /// Submits `a`, waits until it is producing tokens (so it is running),
    /// and returns its receiver.
    fn start_running(engine: &EngineHandle, a: Request) -> impl FnOnce() {
        let mut events = engine.submit(a).unwrap();
        assert!(matches!(events.blocking_recv(), Some(Event::Token(_))));
        move || while events.blocking_recv().is_some() {}
    }

    /// When the engine admitted a request and when it produced its first
    /// token. The engine measures both on its own thread, so they are
    /// ordered exactly as its steps were, whatever the test threads do.
    struct Served {
        tokens: Vec<u32>,
        admitted: Instant,
        first_token: Instant,
    }

    /// Submits `requests` one after another while another request keeps
    /// the engine busy, and waits for all of them.
    fn race(policy: Policy, max_batch: usize, requests: Vec<Request>) -> Vec<Served> {
        let (engine, _thread, _) = spawn(model(), config(max_batch, 8, policy, None));
        let finish = start_running(&engine, request(prompt(4, 0), 120));
        // The engine's times are relative to each submission; `submit`
        // records it within a microsecond of `Instant::now()` here, far
        // less than one step of the engine.
        let submitted: Vec<_> = requests
            .into_iter()
            .map(|r| (Instant::now(), engine.submit(r).unwrap()))
            .collect();
        let served = submitted
            .into_iter()
            .map(|(at, events)| {
                let (tokens, summary) = collect(events).unwrap();
                Served {
                    tokens,
                    admitted: at + summary.queued,
                    first_token: at + summary.time_to_first_token,
                }
            })
            .collect();
        finish();
        served
    }

    #[test]
    fn shortest_first_admits_a_short_prompt_before_a_long_one() {
        // One request at a time: the long prompt arrives first.
        let (long, short) = (prompt(60, 1), prompt(5, 2));
        let reqs = || vec![request(long.clone(), 3), request(short.clone(), 3)];
        let fcfs = race(Policy::Fcfs, 1, reqs());
        assert!(fcfs[0].admitted < fcfs[1].admitted, "arrival order");
        let spf = race(Policy::ShortestFirst, 1, reqs());
        assert!(spf[1].admitted < spf[0].admitted, "the short prompt first");
        // The order changes nothing about the answers.
        for run in [&fcfs, &spf] {
            assert_eq!(run[0].tokens, greedy(&long, 3));
            assert_eq!(run[1].tokens, greedy(&short, 3));
        }
    }

    #[test]
    fn shortest_first_gives_prompt_chunks_to_the_shortest_prompt() {
        // Both run together; 8 tokens per step, 1 of them for the busy
        // request's decode token. The long prompt needs 7 steps.
        let (long, short) = (prompt(48, 3), prompt(6, 4));
        let reqs = || vec![request(long.clone(), 2), request(short.clone(), 2)];
        let fcfs = race(Policy::Fcfs, 4, reqs());
        assert!(fcfs[0].first_token < fcfs[1].first_token);
        let spf = race(Policy::ShortestFirst, 4, reqs());
        assert!(spf[1].first_token < spf[0].first_token);
        assert_eq!(spf[0].tokens, greedy(&long, 2));
        assert_eq!(spf[1].tokens, greedy(&short, 2));
    }

    #[test]
    fn a_full_queue_rejects_new_requests_at_once() {
        let (engine, _thread, stats) = spawn(model(), config(1, 8, Policy::Fcfs, Some(1)));
        let finish = start_running(&engine, request(prompt(4, 0), 100));
        // The batch is full (one request): the next one waits...
        let waiting = engine.submit(request(prompt(5, 1), 4)).unwrap();
        // ...and the queue is full: this one is refused immediately.
        let refused = collect(engine.submit(request(prompt(5, 2), 4)).unwrap());
        assert!(refused.unwrap_err().contains("overloaded"));
        assert_eq!(Stats::get(&stats.rejected), 1);
        finish();
        assert_eq!(collect(waiting).unwrap().0, greedy(&prompt(5, 1), 4));
        assert_eq!(Stats::get(&stats.peak_waiting), 1);
    }

    #[test]
    fn the_load_generators_record_every_token() {
        let (engine, _thread, _) = spawn(model(), config(4, 16, Policy::Fcfs, None));
        let at = poisson_arrivals(50.0, 6, 3);
        let plan: Vec<Planned> = at
            .iter()
            .enumerate()
            .map(|(i, &at)| Planned {
                at,
                prompt: prompt(5 + i, i as u32),
                max_tokens: 4 + i,
            })
            .collect();
        for traces in [
            run_open(&engine, &plan),
            run_closed(&engine, 2, &plan, std::time::Duration::ZERO),
        ] {
            for (t, p) in traces.iter().zip(&plan) {
                assert!(!t.rejected);
                assert_eq!(t.tokens.len(), p.max_tokens);
                assert_eq!(t.prompt_tokens, p.prompt.len());
                assert!(t.tokens.windows(2).all(|w| w[0] <= w[1]));
                assert!(t.done >= *t.tokens.last().unwrap());
            }
        }
        // Open-loop requests are sent at their planned times.
        let open = run_open(&engine, &plan);
        for (t, p) in open.iter().zip(&plan) {
            assert!(t.sent >= p.at);
        }
    }
}
