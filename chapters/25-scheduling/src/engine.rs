//! Chapter 24's paged engine, with the scheduling decisions made explicit.
//!
//! What changes:
//!
//! - **Policy:** which waiting request is admitted next, and which prompts
//!   get the step's token budget first. Chapter 24 did both in order of
//!   arrival ([`Policy::Fcfs`]); [`Policy::ShortestFirst`] serves the
//!   requests with the fewest prompt tokens left first.
//! - **A bounded queue:** with `max_waiting`, a request that arrives when
//!   that many are already waiting is rejected at once instead of queued.
//! - Two more counters: requests rejected for load, and the longest queue.
//!
//! Everything else (blocks, preemption, prefix caching, decode tokens
//! before prompt chunks in every step) is chapter 24's engine.

use ch07_threads::SpinPool;
use ch14_kv_cache::{Matrix, Model};
use ch15_sampling::{FinishReason, Sampler};
use ch20_flash_attention::FlashOptions;
use ch21_engine_thread::{EngineHandle, Event, Job, Summary, channel};
use ch24_paged_kv_cache::{
    BlockPool, BlockTable, PagedConfig, PagedScratch, PagedSeq, PrefixCache, forward_paged,
};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

/// In what order the engine serves requests that compete for the batch
/// and for the step's prompt tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// First come, first served: admission and prompt chunks in order of
    /// arrival.
    Fcfs,
    /// Shortest remaining prompt first: among waiting requests, admit the
    /// one with the shortest prompt; among admitted ones, give prompt
    /// chunks to those with the fewest prompt tokens left. Requests that
    /// were preempted come back before new ones in both policies.
    ShortestFirst,
}

/// Chapter 24's configuration, plus the scheduling choices.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub paged: PagedConfig,
    pub policy: Policy,
    /// Requests allowed to wait for admission (`None`: no limit). A request
    /// arriving when the queue is full is rejected immediately.
    pub max_waiting: Option<usize>,
}

/// Counters the engine updates as it runs, readable from any thread.
#[derive(Debug, Default)]
pub struct Stats {
    pub steps: AtomicUsize,
    pub prompt_tokens: AtomicUsize,
    pub cached_tokens: AtomicUsize,
    pub preemptions: AtomicUsize,
    pub recomputed_tokens: AtomicUsize,
    pub peak_batch: AtomicUsize,
    pub peak_blocks: AtomicUsize,
    /// Requests rejected because the queue was full.
    pub rejected: AtomicUsize,
    /// The most requests waiting at once.
    pub peak_waiting: AtomicUsize,
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
    config: SchedulerConfig,
) -> (EngineHandle, JoinHandle<()>, Arc<Stats>) {
    assert!(
        config.paged.max_step_tokens >= config.paged.max_batch,
        "a step must fit one token per request"
    );
    let stats = Arc::new(Stats::default());
    let (handle, queue) = channel();
    let engine = Engine::new(model, config, Arc::clone(&stats));
    let thread = std::thread::Builder::new()
        .name("scheduling engine".into())
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

    /// Whether the request has been admitted before (it was preempted).
    fn started(&self) -> bool {
        self.queued.is_some()
    }
}

struct Engine<W: Matrix> {
    model: Model<W>,
    config: PagedConfig,
    policy: Policy,
    max_waiting: Option<usize>,
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
    fn new(model: Model<W>, config: SchedulerConfig, stats: Arc<Stats>) -> Self {
        let SchedulerConfig {
            paged: config,
            policy,
            max_waiting,
        } = config;
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
            policy,
            max_waiting,
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

    /// Queues a new job, or rejects it if it can never fit or if the queue
    /// is full.
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
        if self.max_waiting.is_some_and(|n| self.waiting.len() >= n) {
            Stats::add(&self.stats.rejected, 1);
            let _ = job.events.send(Event::Rejected(format!(
                "overloaded: {} requests are already waiting",
                self.waiting.len()
            )));
            return;
        }
        let mut sampler = Sampler::new(job.request.params.clone());
        sampler.start(&job.request.prompt);
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
        Stats::max(&self.stats.peak_waiting, self.waiting.len());
    }

    /// The waiting request to admit next: preempted requests first, in
    /// queue order; then the oldest (FCFS) or the shortest (ShortestFirst).
    fn next_waiting(&self) -> Option<usize> {
        if let Some(i) = self.waiting.iter().position(Active::started) {
            return Some(i);
        }
        match self.policy {
            Policy::Fcfs => (!self.waiting.is_empty()).then_some(0),
            Policy::ShortestFirst => {
                (0..self.waiting.len()).min_by_key(|&i| self.waiting[i].tokens.len())
            }
        }
    }

    /// Admits waiting requests, in the policy's order, while there is room
    /// in the batch and blocks for their tokens.
    fn admit(&mut self) {
        let bs = self.config.block_size;
        // Leave a few blocks for running requests to grow into.
        let reserve = (self.config.num_blocks / 100).max(1);
        while self.active.len() < self.config.max_batch {
            let Some(i) = self.next_waiting() else {
                break;
            };
            let mut a = self.waiting.remove(i).expect("index from next_waiting");
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
                self.waiting.insert(i, a);
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
    /// each, then prompt chunks within the step's token budget, in the
    /// policy's order. Makes sure every planned request has blocks for its
    /// tokens, evicting from the prefix cache and then preempting the
    /// newest requests if needed.
    #[allow(
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
        // Prompt chunks: in admission order, or fewest tokens left first.
        let mut order: Vec<usize> = (0..self.active.len()).collect();
        if self.policy == Policy::ShortestFirst {
            order.sort_by_key(|&i| self.active[i].remaining());
        }
        for i in order {
            let a = &mut self.active[i];
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
                plan[i] = Some(a.table.len..a.table.len + n);
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
