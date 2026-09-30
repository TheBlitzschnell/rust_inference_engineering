//! Chapter 23's batching engine, with the KV cache in blocks.
//!
//! What changes:
//!
//! - **Admission** asks for blocks, not a slot: a request is admitted when
//!   the pool has blocks for its prompt (plus a small reserve), so the
//!   number of requests running depends on their actual lengths.
//! - **Growth:** a request takes one more block each time it crosses a
//!   block boundary.
//! - **Preemption:** if a running request needs a block and none is free,
//!   the most recently admitted request gives all of its blocks back and
//!   returns to the front of the queue; when readmitted, it recomputes its
//!   keys and values (prompt and the tokens generated so far) and carries
//!   on where it stopped.
//! - **Prefix caching** (optional): full blocks are registered in a
//!   [`PrefixCache`] as soon as they are filled; a new request starts from
//!   the longest cached prefix of its tokens and computes only the rest.

use crate::blocks::{BlockPool, BlockTable};
use crate::forward::{PagedScratch, PagedSeq, forward_paged};
use crate::prefix::PrefixCache;
use ch07_threads::SpinPool;
use ch14_kv_cache::{Matrix, Model};
use ch15_sampling::{FinishReason, Sampler};
use ch20_flash_attention::FlashOptions;
use ch21_engine_thread::{EngineHandle, Event, Job, Summary, channel};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

/// How the engine batches and stores keys and values.
#[derive(Clone, Debug)]
pub struct PagedConfig {
    pub threads: usize,
    /// Requests running at most.
    pub max_batch: usize,
    /// Tokens per request at most, prompt and answer together.
    pub context: usize,
    /// Tokens per step at most (decode tokens first, then prompt chunks).
    /// Must be at least `max_batch`.
    pub max_step_tokens: usize,
    /// The KV cache: `num_blocks` blocks of `block_size` positions.
    pub num_blocks: usize,
    pub block_size: usize,
    pub prefix_caching: bool,
}

/// Counters the engine updates as it runs, readable from any thread.
#[derive(Debug, Default)]
pub struct Stats {
    pub steps: AtomicUsize,
    /// Prompt tokens of admitted requests.
    pub prompt_tokens: AtomicUsize,
    /// Of those, tokens whose keys and values came from the prefix cache.
    pub cached_tokens: AtomicUsize,
    pub preemptions: AtomicUsize,
    /// Positions whose keys and values were thrown away by preemption, to
    /// be computed again.
    pub recomputed_tokens: AtomicUsize,
    /// The most requests in one step, and the most blocks in use at once.
    pub peak_batch: AtomicUsize,
    pub peak_blocks: AtomicUsize,
}

impl Stats {
    fn add(counter: &AtomicUsize, n: usize) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    fn max(counter: &AtomicUsize, n: usize) {
        counter.fetch_max(n, Ordering::Relaxed);
    }

    pub fn get(counter: &AtomicUsize) -> usize {
        counter.load(Ordering::Relaxed)
    }
}

/// Starts the engine on a new thread; returns its handle, its thread and
/// its counters.
pub fn spawn<W: Matrix + 'static>(
    model: Model<W>,
    config: PagedConfig,
) -> (EngineHandle, JoinHandle<()>, Arc<Stats>) {
    assert!(
        config.max_step_tokens >= config.max_batch,
        "a step must fit one token per request"
    );
    let stats = Arc::new(Stats::default());
    let (handle, queue) = channel();
    let engine = Engine::new(model, config, Arc::clone(&stats));
    let thread = std::thread::Builder::new()
        .name("paged engine".into())
        .spawn(move || engine.run(&queue))
        .expect("spawning the engine thread");
    (handle, thread, stats)
}

/// A request, waiting or running.
struct Active {
    job: Job,
    sampler: Sampler,
    /// The prompt, then every token generated so far.
    tokens: Vec<u32>,
    /// Blocks holding the keys and values of `tokens[..table.len]`.
    table: BlockTable,
    generated: usize,
    max_new: usize,
    queued: Option<Duration>,
    first_token: Option<Duration>,
    finish: Option<FinishReason>,
    /// Full blocks already offered to the prefix cache.
    offered: usize,
    /// Set when the request loses its blocks during planning.
    preempted: bool,
}

impl Active {
    /// Tokens whose keys and values still have to be computed. Exactly 1
    /// for a request that is decoding: its last sampled token.
    fn remaining(&self) -> usize {
        self.tokens.len() - self.table.len
    }
}

struct Engine<W: Matrix> {
    model: Model<W>,
    config: PagedConfig,
    kv: BlockPool,
    prefix: PrefixCache,
    waiting: VecDeque<Active>,
    /// In order of admission.
    active: Vec<Active>,
    scratch: PagedScratch,
    attention: FlashOptions,
    stats: Arc<Stats>,
}

impl<W: Matrix> Engine<W> {
    fn new(model: Model<W>, config: PagedConfig, stats: Arc<Stats>) -> Self {
        let kv = BlockPool::new(&model.config, config.num_blocks, config.block_size);
        let scratch = PagedScratch::new(&model.config, config.max_step_tokens, config.max_batch);
        Self {
            kv,
            prefix: PrefixCache::new(),
            waiting: VecDeque::new(),
            active: Vec::new(),
            scratch,
            attention: FlashOptions::default(),
            stats,
            model,
            config,
        }
    }

    fn run(mut self, queue: &Receiver<Job>) {
        let mut pool: Option<SpinPool> = None;
        let mut open = true;
        loop {
            if self.active.is_empty() && self.waiting.is_empty() {
                pool = None;
                if !open {
                    return;
                }
                match queue.recv() {
                    Ok(job) => self.receive(job),
                    Err(_) => return,
                }
            }
            loop {
                match queue.try_recv() {
                    Ok(job) => self.receive(job),
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

    /// Queues a new job, or rejects it if it can never fit.
    fn receive(&mut self, job: Job) {
        let prompt = job.request.prompt.len();
        let blocks = (prompt + 1).div_ceil(self.config.block_size);
        if prompt == 0 || prompt >= self.config.context || blocks > self.config.num_blocks {
            let _ = job.events.send(Event::Rejected(format!(
                "prompt of {prompt} tokens does not fit a context of {} tokens",
                self.config
                    .context
                    .min(self.config.num_blocks * self.config.block_size)
            )));
            return;
        }
        let mut sampler = Sampler::new(job.request.params.clone());
        sampler.start(&job.request.prompt);
        // A request must fit the whole pool on its own, or it could end up
        // preempting itself forever.
        let room = self
            .config
            .context
            .min(self.config.num_blocks * self.config.block_size);
        let max_new = job.request.max_tokens.min(room - prompt);
        self.waiting.push_back(Active {
            tokens: job.request.prompt.clone(),
            job,
            sampler,
            table: BlockTable::default(),
            generated: 0,
            max_new,
            queued: None,
            first_token: None,
            finish: None,
            offered: 0,
            preempted: false,
        });
    }

    /// Admits waiting requests, oldest first, while there is room in the
    /// batch and blocks for their tokens.
    fn admit(&mut self) {
        let bs = self.config.block_size;
        // Leave a few blocks for running requests to grow into.
        let reserve = (self.config.num_blocks / 100).max(1);
        while self.active.len() < self.config.max_batch {
            let Some(mut a) = self.waiting.pop_front() else {
                break;
            };
            if a.job.events.is_closed() {
                continue; // the client left while waiting
            }
            if a.max_new == 0 {
                a.finish = Some(FinishReason::Length);
                self.active.push(a);
                continue;
            }
            // Start from the longest cached prefix. Keep at least one token
            // to compute: its logits give the next token.
            if self.config.prefix_caching {
                let most = (a.tokens.len() - 1) / bs;
                a.table.blocks = self.prefix.lookup(&mut self.kv, &a.tokens, most);
                a.table.len = a.table.blocks.len() * bs;
            }
            let needed = (a.tokens.len() + 1).div_ceil(bs) - a.table.blocks.len();
            if self.kv.free_blocks() < needed + reserve {
                let short = needed + reserve - self.kv.free_blocks();
                self.prefix.evict(&mut self.kv, short);
            }
            if self.kv.free_blocks() < needed + reserve && !self.active.is_empty() {
                // Not now: give back the cached blocks and wait for requests
                // to finish. (With nothing running, admit anyway: the
                // request fits the pool by itself.)
                a.table.release(&mut self.kv);
                self.waiting.push_front(a);
                break;
            }
            let was_cached = a.table.len;
            if a.queued.is_none() {
                a.queued = Some(a.job.submitted.elapsed());
                Stats::add(&self.stats.prompt_tokens, a.tokens.len());
                Stats::add(&self.stats.cached_tokens, was_cached);
            }
            self.active.push(a);
        }
    }

    /// Plans, allocates blocks, runs one step, samples, retires.
    fn step(&mut self, pool: &mut SpinPool) {
        let mut plan = self.plan();
        // Requests preempted while planning go back to the front of the
        // queue, keeping their order.
        let mut i = self.active.len();
        while i > 0 {
            i -= 1;
            if self.active[i].preempted {
                let mut a = self.active.remove(i);
                plan.remove(i);
                a.preempted = false;
                self.waiting.push_front(a);
            }
        }
        Stats::add(&self.stats.steps, 1);
        Stats::max(
            &self.stats.peak_batch,
            plan.iter().filter(|p| p.is_some()).count(),
        );
        Stats::max(
            &self.stats.peak_blocks,
            self.kv.num_blocks() - self.kv.free_blocks(),
        );

        let Self {
            model,
            kv,
            active,
            scratch,
            attention,
            ..
        } = self;
        let mut seqs: Vec<PagedSeq<'_>> = Vec::new();
        for (a, range) in active.iter_mut().zip(&plan) {
            let Some(r) = range else { continue };
            seqs.push(PagedSeq {
                tokens: &a.tokens[r.clone()],
                logits: r.end == a.tokens.len(),
                table: &mut a.table,
            });
        }
        let logits = if seqs.is_empty() {
            &[][..]
        } else {
            forward_paged(model, pool, kv, &mut seqs, scratch, attention)
        };
        drop(seqs);

        let vocab = model.config.vocab_size;
        let mut rows = logits.chunks_exact(vocab);
        for (a, range) in active.iter_mut().zip(&plan) {
            let Some(r) = range else { continue };
            if r.end < a.tokens.len() {
                continue; // a prompt chunk that is not the last: no logits
            }
            let token = a.sampler.sample(rows.next().expect("a row per request"));
            if a.job.request.stop_tokens.contains(&token) {
                a.finish = Some(FinishReason::StopToken);
                continue;
            }
            let now = a.job.submitted.elapsed();
            a.first_token.get_or_insert(now);
            if a.job.events.send(Event::Token(token)).is_err() {
                a.finish = Some(FinishReason::Stopped);
                continue;
            }
            a.tokens.push(token);
            a.generated += 1;
            if a.generated == a.max_new {
                a.finish = Some(FinishReason::Length);
            }
        }
        self.offer_full_blocks();
        self.retire();
    }

    /// What each running request contributes to the step (by index in
    /// `active`; `None` for nothing): decoding requests first, one token
    /// each, then prompt chunks within the step's token budget. Makes sure
    /// every planned request has blocks for its tokens, evicting from the
    /// prefix cache and then preempting the newest requests if needed.
    #[expect(
        clippy::needless_range_loop,
        reason = "the decode loop preempts other requests by index while it runs"
    )]
    fn plan(&mut self) -> Vec<Option<Range<usize>>> {
        let bs = self.config.block_size;
        let mut budget = self.config.max_step_tokens;
        let mut plan = vec![None; self.active.len()];
        for a in &mut self.active {
            if a.finish.is_none() && a.job.events.is_closed() {
                a.finish = Some(FinishReason::Stopped);
            }
        }
        for i in 0..self.active.len() {
            let a = &self.active[i];
            if a.finish.is_some() || a.preempted || a.remaining() != 1 {
                continue;
            }
            // Room for one more position, preempting others if necessary.
            loop {
                let (kv, a) = (&mut self.kv, &mut self.active[i]);
                if a.table.reserve(kv, a.table.len + 1) {
                    break;
                }
                if self.prefix.evict(&mut self.kv, 1) > 0 {
                    continue;
                }
                let newest = (0..self.active.len())
                    .rev()
                    .find(|&j| self.active[j].finish.is_none() && !self.active[j].preempted)
                    .expect("request i itself is running");
                self.preempt(newest);
                if newest == i {
                    break;
                }
            }
            let a = &self.active[i];
            if !a.preempted {
                plan[i] = Some(a.table.len..a.table.len + 1);
                budget -= 1;
            }
        }
        for (a, planned) in self.active.iter_mut().zip(&mut plan) {
            if a.finish.is_some() || a.preempted || a.remaining() <= 1 || budget == 0 {
                continue;
            }
            // As much of the rest as the budget and the free blocks allow.
            let want = a.remaining().min(budget);
            if !a.table.reserve(&mut self.kv, a.table.len + want) {
                let short = (a.table.len + want).div_ceil(bs) - a.table.blocks.len();
                self.prefix.evict(&mut self.kv, short);
                a.table.reserve(&mut self.kv, a.table.len + want);
            }
            let n = want.min(a.table.capacity(bs) - a.table.len);
            if n > 0 {
                *planned = Some(a.table.len..a.table.len + n);
                budget -= n;
            }
        }
        plan
    }

    /// Takes all of request `i`'s blocks back; it will recompute them.
    fn preempt(&mut self, i: usize) {
        let a = &mut self.active[i];
        Stats::add(&self.stats.recomputed_tokens, a.table.len);
        a.table.release(&mut self.kv);
        a.offered = 0;
        a.preempted = true;
        Stats::add(&self.stats.preemptions, 1);
    }

    /// Registers newly filled blocks with the prefix cache.
    fn offer_full_blocks(&mut self) {
        if !self.config.prefix_caching {
            return;
        }
        let bs = self.config.block_size;
        for a in &mut self.active {
            let full = a.table.len / bs;
            if full > a.offered {
                self.prefix.insert(&mut self.kv, &a.tokens, &a.table);
                a.offered = full;
            }
        }
    }

    /// Sends the final event of every finished request and frees its blocks
    /// (the prefix cache keeps its own references).
    fn retire(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            let Some(finish) = self.active[i].finish else {
                i += 1;
                continue;
            };
            let mut a = self.active.remove(i);
            a.table.release(&mut self.kv);
            let total = a.job.submitted.elapsed();
            let _ = a.job.events.send(Event::Done(Summary {
                finish,
                prompt_tokens: a.job.request.prompt.len(),
                completion_tokens: a.generated,
                queued: a.queued.unwrap_or(total),
                time_to_first_token: a.first_token.unwrap_or(total),
                total,
            }));
        }
    }
}
