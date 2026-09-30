//! Chapter 21: the engine on its own thread.
//!
//! The model, its thread pool, its KV cache and its scratch space belong to
//! one engine thread. Everything else (an HTTP server, a command line, a
//! test) talks to it through channels:
//!
//! - requests go in over one channel ([`EngineHandle::submit`]);
//! - each request gets its own channel of [`Event`]s back: tokens as they
//!   are produced, then one `Done`;
//! - dropping the receiver cancels the request: the engine notices its next
//!   send fail and moves on.
//!
//! One request runs at a time here; chapter 23 batches them.

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::{FinishReason, Sampler, SamplingParams};
use std::ops::ControlFlow;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::mpsc as async_mpsc;

/// What a client asks for.
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    /// Tokens that end generation (the model's end-of-turn token).
    pub stop_tokens: Vec<u32>,
}

/// What the engine sends back, in order: `Token`s, then exactly one final
/// `Done` or `Rejected`.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Token(u32),
    Done(Summary),
    /// The request could not run (for example, its prompt does not fit).
    Rejected(String),
}

/// How a finished request went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub finish: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// From submission to the start of processing (waiting behind others).
    pub queued: Duration,
    /// From submission to the first token.
    pub time_to_first_token: Duration,
    /// From submission to the end.
    pub total: Duration,
}

/// A request inside the engine: what was asked, where to send events, and
/// when it arrived.
pub struct Job {
    pub request: Request,
    pub events: async_mpsc::UnboundedSender<Event>,
    pub submitted: Instant,
}

/// The client side: cheap to clone, usable from any thread or task.
#[derive(Clone)]
pub struct EngineHandle {
    jobs: mpsc::Sender<Job>,
}

/// The engine has stopped (it was shut down, or it panicked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineGone;

impl std::fmt::Display for EngineGone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the engine thread has stopped")
    }
}

impl std::error::Error for EngineGone {}

impl EngineHandle {
    /// Queues a request and returns the receiving end of its events. Works
    /// from synchronous code (`blocking_recv`) and async code (`recv().await`).
    /// Drop the receiver to cancel.
    pub fn submit(
        &self,
        request: Request,
    ) -> Result<async_mpsc::UnboundedReceiver<Event>, EngineGone> {
        let (events, receiver) = async_mpsc::unbounded_channel();
        let job = Job {
            request,
            events,
            submitted: Instant::now(),
        };
        self.jobs.send(job).map_err(|_| EngineGone)?;
        Ok(receiver)
    }
}

/// A handle and the queue of jobs it feeds. [`spawn`] uses it; so can any
/// other engine loop (chapter 23's batching engine), which then works with
/// every client of `EngineHandle`, such as chapter 22's server.
pub fn channel() -> (EngineHandle, mpsc::Receiver<Job>) {
    let (jobs, queue) = mpsc::channel();
    (EngineHandle { jobs }, queue)
}

/// Starts the engine on a new thread. It runs until every `EngineHandle`
/// is dropped, then finishes the requests already queued and returns.
pub fn spawn<W: Matrix + 'static>(
    model: Model<W>,
    threads: usize,
    context: usize,
) -> (EngineHandle, JoinHandle<()>) {
    let (handle, queue) = channel();
    let thread = std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            // Everything the engine owns is created here, on its thread.
            let mut cache = KvCache::new(&model.config, context);
            let mut scratch = Scratch::new(&model.config, 256, context);
            // `recv` sleeps without using the CPU until a job arrives, and
            // fails once every handle is gone: that is the shutdown.
            while let Ok(job) = queue.recv() {
                // Pool workers spin while they wait for work: right while
                // requests run, a waste of every core while none do. So the
                // pool lives only while there is work (starting its threads
                // takes well under a millisecond).
                let mut pool = SpinPool::new(threads);
                run(&model, &mut pool, &mut cache, &mut scratch, job);
                while let Ok(job) = queue.try_recv() {
                    run(&model, &mut pool, &mut cache, &mut scratch, job);
                }
            }
        })
        .expect("spawning the engine thread");
    (handle, thread)
}

/// Runs one request to completion (or cancellation).
fn run<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    job: Job,
) {
    let Job {
        request,
        events,
        submitted,
    } = job;
    let queued = submitted.elapsed();
    let room = cache.capacity().saturating_sub(request.prompt.len());
    if request.prompt.is_empty() || room == 0 {
        // The client may be gone already; nothing to do about that.
        let _ = events.send(Event::Rejected(format!(
            "prompt of {} tokens does not fit a context of {}",
            request.prompt.len(),
            cache.capacity()
        )));
        return;
    }
    let mut sampler = Sampler::new(request.params.clone());
    let mut first_token = None;
    let mut completion_tokens = 0;
    let finish = ch15_sampling::generate(
        model,
        pool,
        &request.prompt,
        &mut sampler,
        cache,
        scratch,
        request.max_tokens.min(room),
        &request.stop_tokens,
        |token| {
            first_token.get_or_insert_with(|| submitted.elapsed());
            completion_tokens += 1;
            // A failed send means the receiver was dropped: the client has
            // gone, so stop spending compute on it.
            if events.send(Event::Token(token)).is_err() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    let _ = events.send(Event::Done(Summary {
        finish,
        prompt_tokens: request.prompt.len(),
        completion_tokens,
        queued,
        time_to_first_token: first_token.unwrap_or_else(|| submitted.elapsed()),
        total: submitted.elapsed(),
    }));
}

/// Collects a request's events into its tokens and summary (for tests and
/// simple callers). Blocks the current thread.
pub fn collect(
    mut events: async_mpsc::UnboundedReceiver<Event>,
) -> Result<(Vec<u32>, Summary), String> {
    let mut tokens = Vec::new();
    while let Some(event) = events.blocking_recv() {
        match event {
            Event::Token(t) => tokens.push(t),
            Event::Done(summary) => return Ok((tokens, summary)),
            Event::Rejected(why) => return Err(why),
        }
    }
    Err("the engine stopped before finishing the request".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::DenseF32;

    fn engine() -> (EngineHandle, JoinHandle<()>, Model<DenseF32>) {
        let w = Weights::random(&Config::tiny(), 1);
        let (handle, thread) = spawn(Model::from_reference(&w), 2, 128);
        (handle, thread, Model::from_reference(&w))
    }

    fn request(prompt: &[u32], max_tokens: usize) -> Request {
        Request {
            prompt: prompt.to_vec(),
            params: SamplingParams::greedy(),
            max_tokens,
            stop_tokens: Vec::new(),
        }
    }

    #[test]
    fn tokens_stream_back_and_match_direct_generation() {
        let (handle, _thread, model) = engine();
        let (tokens, summary) = collect(handle.submit(request(&[1, 2, 3], 10)).unwrap()).unwrap();
        let mut pool = SpinPool::new(2);
        let mut cache = KvCache::new(&model.config, 128);
        let mut scratch = Scratch::new(&model.config, 16, 128);
        let want = ch14_kv_cache::generate_greedy(
            &model,
            &mut pool,
            &[1, 2, 3],
            10,
            &mut cache,
            &mut scratch,
            |_, _| {},
        );
        assert_eq!(tokens, want[3..]);
        assert_eq!(summary.finish, FinishReason::Length);
        assert_eq!((summary.prompt_tokens, summary.completion_tokens), (3, 10));
    }

    #[test]
    fn requests_from_many_threads_are_served_in_turn() {
        let (handle, _thread, _) = engine();
        let clients: Vec<_> = (0..4u32)
            .map(|i| {
                let h = handle.clone();
                std::thread::spawn(move || collect(h.submit(request(&[i + 1, 5], 6)).unwrap()))
            })
            .collect();
        for c in clients {
            let (tokens, summary) = c.join().unwrap().unwrap();
            assert_eq!(tokens.len(), 6);
            assert_eq!(summary.completion_tokens, 6);
        }
    }

    #[test]
    fn dropping_the_receiver_cancels_the_request() {
        let (handle, _thread, _) = engine();
        let mut events = handle.submit(request(&[1, 2], 100)).unwrap();
        assert!(matches!(events.blocking_recv(), Some(Event::Token(_))));
        drop(events);
        // The engine stops that request and serves the next one.
        let (tokens, _) = collect(handle.submit(request(&[3], 4)).unwrap()).unwrap();
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn a_prompt_that_does_not_fit_is_rejected() {
        let (handle, _thread, _) = engine();
        let long: Vec<u32> = (0..200).map(|i| i % 90).collect();
        assert!(collect(handle.submit(request(&long, 4)).unwrap()).is_err());
    }

    #[test]
    fn dropping_every_handle_stops_the_engine() {
        let (handle, thread, _) = engine();
        drop(handle);
        thread.join().unwrap();
    }
}
