//! Continuous batching: the engine thread of chapter 21, serving many
//! requests at once.
//!
//! Every step, the engine runs one [`forward_batch`] over all the requests
//! it holds: each decoding request contributes its last token, each new
//! request a chunk of its prompt. Between steps, finished requests leave
//! and waiting ones join, so the batch changes at every step. That is the
//! "continuous" (or "iteration-level") part: nobody waits for the rest of
//! a batch to finish before starting.
//!
//! Clients use chapter 21's `EngineHandle`, unchanged, so chapter 22's
//! server works with this engine too.

use crate::forward::{BatchScratch, BatchSeq, forward_batch};
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model};
use ch15_sampling::{FinishReason, Sampler};
use ch20_flash_attention::FlashOptions;
use ch21_engine_thread::{EngineHandle, Event, Job, Summary, channel};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

/// How the engine batches.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub threads: usize,
    /// Requests in the batch at most. Each owns a KV cache slot of
    /// `context` positions while it runs.
    pub max_batch: usize,
    /// Tokens per request, prompt and answer together.
    pub context: usize,
    /// Tokens processed in one step at most: one per decoding request, and
    /// prompt chunks up to this limit. Must be at least `max_batch`.
    pub max_step_tokens: usize,
}

/// Starts the batching engine on a new thread. Like chapter 21's, it runs
/// until every handle is dropped and its requests are finished.
pub fn spawn<W: Matrix + 'static>(
    model: Model<W>,
    config: BatchConfig,
) -> (EngineHandle, JoinHandle<()>) {
    assert!(
        config.max_step_tokens >= config.max_batch,
        "a step must fit one token per request"
    );
    let (handle, queue) = channel();
    let thread = std::thread::Builder::new()
        .name("batching engine".into())
        .spawn(move || Engine::new(model, config).run(&queue))
        .expect("spawning the engine thread");
    (handle, thread)
}

/// A request in the batch.
struct Active {
    job: Job,
    /// Its KV cache slot.
    slot: usize,
    sampler: Sampler,
    /// Prompt tokens processed so far.
    prefilled: usize,
    /// The token sampled last: the input of its next decode step.
    next: u32,
    generated: usize,
    max_new: usize,
    queued: Duration,
    first_token: Option<Duration>,
    /// Set when the request is done; it leaves after the step.
    finish: Option<FinishReason>,
}

impl Active {
    fn decoding(&self) -> bool {
        self.prefilled == self.job.request.prompt.len()
    }
}

struct Engine<W: Matrix> {
    model: Model<W>,
    config: BatchConfig,
    caches: Vec<KvCache>,
    /// Slots not in use.
    free: Vec<usize>,
    waiting: VecDeque<Job>,
    /// In order of arrival.
    active: Vec<Active>,
    scratch: BatchScratch,
    attention: FlashOptions,
}

impl<W: Matrix> Engine<W> {
    fn new(model: Model<W>, config: BatchConfig) -> Self {
        let caches = (0..config.max_batch)
            .map(|_| KvCache::new(&model.config, config.context))
            .collect();
        let scratch = BatchScratch::new(&model.config, config.max_step_tokens, config.max_batch);
        Self {
            free: (0..config.max_batch).rev().collect(),
            caches,
            waiting: VecDeque::new(),
            active: Vec::new(),
            scratch,
            attention: FlashOptions::default(),
            model,
            config,
        }
    }

    fn run(mut self, queue: &Receiver<Job>) {
        let mut pool: Option<SpinPool> = None;
        let mut open = true;
        loop {
            if self.active.is_empty() && self.waiting.is_empty() {
                // Nothing to do: drop the spinning pool (chapter 21) and
                // sleep until a job arrives or every handle is gone.
                pool = None;
                if !open {
                    return;
                }
                match queue.recv() {
                    Ok(job) => self.waiting.push_back(job),
                    Err(_) => return,
                }
            }
            // Everything else that has arrived, without waiting.
            loop {
                match queue.try_recv() {
                    Ok(job) => self.waiting.push_back(job),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            self.admit();
            if !self.active.is_empty() {
                let pool = pool.get_or_insert_with(|| SpinPool::new(self.config.threads));
                self.step(pool);
            }
        }
    }

    /// Moves waiting requests into free slots, in order of arrival.
    fn admit(&mut self) {
        // A client that left while waiting costs nothing.
        self.waiting.retain(|job| !job.events.is_closed());
        while let Some(&slot) = self.free.last() {
            let Some(job) = self.waiting.pop_front() else {
                break;
            };
            let prompt = job.request.prompt.len();
            let room = self.config.context.saturating_sub(prompt);
            if prompt == 0 || room == 0 {
                let _ = job.events.send(Event::Rejected(format!(
                    "prompt of {prompt} tokens does not fit a context of {}",
                    self.config.context
                )));
                continue;
            }
            self.free.pop();
            self.caches[slot].clear();
            let mut sampler = Sampler::new(job.request.params.clone());
            sampler.start(&job.request.prompt);
            let max_new = job.request.max_tokens.min(room);
            let mut active = Active {
                queued: job.submitted.elapsed(),
                job,
                slot,
                sampler,
                prefilled: 0,
                next: 0,
                generated: 0,
                max_new,
                first_token: None,
                finish: None,
            };
            if max_new == 0 {
                active.finish = Some(FinishReason::Length);
            }
            self.active.push(active);
        }
    }

    /// Plans one step, runs it, samples, and retires finished requests.
    fn step(&mut self, pool: &mut SpinPool) {
        let plan = self.plan();
        let Self {
            model,
            caches,
            active,
            scratch,
            attention,
            ..
        } = self;

        // One `BatchSeq` per planned request. Each borrows its own cache:
        // `slots` hands out each `&mut KvCache` at most once.
        let mut slots: Vec<Option<&mut KvCache>> = caches.iter_mut().map(Some).collect();
        let mut seqs: Vec<BatchSeq<'_>> = plan
            .iter()
            .map(|(i, chunk)| {
                let a = &active[*i];
                let prompt = &a.job.request.prompt;
                let (tokens, logits) = match chunk {
                    Some(r) => (&prompt[r.clone()], r.end == prompt.len()),
                    None => (std::slice::from_ref(&a.next), true),
                };
                BatchSeq {
                    tokens,
                    cache: slots[a.slot].take().expect("one request per slot"),
                    logits,
                }
            })
            .collect();
        let logits = if seqs.is_empty() {
            &[][..]
        } else {
            forward_batch(model, pool, &mut seqs, scratch, attention)
        };
        drop(seqs);

        // Sample the next token of every request that got logits.
        let vocab = model.config.vocab_size;
        let mut rows = logits.chunks_exact(vocab);
        for (i, chunk) in &plan {
            let a = &mut active[*i];
            if let Some(r) = chunk {
                a.prefilled = r.end;
                if !a.decoding() {
                    continue;
                }
            }
            let token = a.sampler.sample(rows.next().expect("a row per request"));
            if a.job.request.stop_tokens.contains(&token) {
                a.finish = Some(FinishReason::StopToken);
                continue;
            }
            let now = a.job.submitted.elapsed();
            a.first_token.get_or_insert(now);
            if a.job.events.send(Event::Token(token)).is_err() {
                // The client has gone (chapter 21).
                a.finish = Some(FinishReason::Stopped);
                continue;
            }
            a.generated += 1;
            a.next = token;
            if a.generated == a.max_new {
                a.finish = Some(FinishReason::Length);
            }
        }
        self.retire();
    }

    /// What each request contributes to the next step: `None` for one
    /// decode token, `Some(range)` for a chunk of its prompt. Decoding
    /// requests come first, one token each, so that a long new prompt
    /// never stalls them; prompt chunks fill the rest of the step, oldest
    /// request first.
    fn plan(&mut self) -> Vec<(usize, Option<Range<usize>>)> {
        let mut budget = self.config.max_step_tokens;
        let mut plan = Vec::with_capacity(self.active.len());
        for (i, a) in self.active.iter_mut().enumerate() {
            if a.finish.is_none() && a.job.events.is_closed() {
                a.finish = Some(FinishReason::Stopped);
            }
            if a.finish.is_none() && a.decoding() {
                plan.push((i, None));
                budget -= 1;
            }
        }
        for (i, a) in self.active.iter().enumerate() {
            if a.finish.is_none() && !a.decoding() && budget > 0 {
                let n = (a.job.request.prompt.len() - a.prefilled).min(budget);
                plan.push((i, Some(a.prefilled..a.prefilled + n)));
                budget -= n;
            }
        }
        plan
    }

    /// Sends the final event of every finished request and frees its slot.
    fn retire(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            let Some(finish) = self.active[i].finish else {
                i += 1;
                continue;
            };
            // `remove`, not `swap_remove`: the order of arrival decides
            // whose prompt goes first.
            let a = self.active.remove(i);
            let total = a.job.submitted.elapsed();
            let _ = a.job.events.send(Event::Done(Summary {
                finish,
                prompt_tokens: a.job.request.prompt.len(),
                completion_tokens: a.generated,
                queued: a.queued,
                time_to_first_token: a.first_token.unwrap_or(total),
                total,
            }));
            self.free.push(a.slot);
        }
    }
}
