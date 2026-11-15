//! The speculative decoding loop: draft `k` tokens cheaply, check them
//! all with one forward pass of the target model, keep the ones it
//! agrees with, repeat.

use crate::draft::{Drafter, Mode};
use crate::verify::{probabilities, verify_greedy, verify_sampled};
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::Rng;
use std::time::{Duration, Instant};

/// What happened during one generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecStats {
    /// Forward passes of the target model after the prompt (one per
    /// verification).
    pub target_passes: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub draft_time: Duration,
    pub verify_time: Duration,
}

impl SpecStats {
    /// The share of draft tokens the target kept.
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }

    /// Tokens produced per target pass: the kept draft tokens plus the one
    /// every verification adds.
    pub fn tokens_per_pass(&self) -> f64 {
        (self.accepted + self.target_passes) as f64 / self.target_passes.max(1) as f64
    }

    pub fn add(&mut self, other: &Self) {
        self.target_passes += other.target_passes;
        self.drafted += other.drafted;
        self.accepted += other.accepted;
        self.draft_time += other.draft_time;
        self.verify_time += other.verify_time;
    }
}

/// What to generate.
#[derive(Clone, Debug)]
pub struct Generation {
    pub max_new: usize,
    /// Draft tokens per verification; 0 is ordinary decoding, one token
    /// per forward pass.
    pub k: usize,
    pub mode: Mode,
    pub seed: u64,
    /// Tokens that end the answer (not included in it).
    pub stop: Vec<u32>,
}

/// The target model with its KV cache, ready to generate.
pub struct Target<'m, W: Matrix> {
    pub model: &'m Model<W>,
    cache: KvCache,
    scratch: Scratch,
}

impl<'m, W: Matrix> Target<'m, W> {
    pub fn new(model: &'m Model<W>, context: usize) -> Self {
        Self {
            model,
            cache: KvCache::new(&model.config, context),
            scratch: Scratch::new(&model.config, 256, context),
        }
    }

    /// Generates up to `g.max_new` tokens after `prompt`, drafting with
    /// `drafter`.
    pub fn generate(
        &mut self,
        pool: &mut SpinPool,
        drafter: &mut dyn Drafter,
        prompt: &[u32],
        g: &Generation,
    ) -> (Vec<u32>, SpecStats) {
        assert!(!prompt.is_empty(), "nothing to continue");
        let (max_new, mode) = (g.max_new, g.mode);
        let vocab = self.model.config.vocab_size;
        let mut rng = Rng::new(g.seed);
        let mut stats = SpecStats::default();
        // The cache holds every token but the last: the last one is the
        // first input of each verification, which gives the target's
        // prediction for the first draft token.
        self.cache.clear();
        if prompt.len() > 1 {
            let n = prompt.len() - 1;
            self.model
                .forward_last(pool, &prompt[..n], &mut self.cache, &mut self.scratch);
        }
        let mut tokens = prompt.to_vec();
        while tokens.len() - prompt.len() < max_new {
            // A verification yields the accepted tokens plus one, so never
            // draft more than one less than the tokens still wanted.
            let left = max_new - (tokens.len() - prompt.len());
            let start = Instant::now();
            let draft = drafter.propose(pool, &tokens, g.k.min(left - 1), mode, &mut rng);
            stats.draft_time += start.elapsed();

            let start = Instant::now();
            let mut input = vec![*tokens.last().expect("a prompt")];
            input.extend_from_slice(&draft.tokens);
            let logits = self
                .model
                .forward_all(pool, &input, &mut self.cache, &mut self.scratch);
            let (kept, next) = match mode {
                Mode::Greedy => verify_greedy(&draft.tokens, logits, vocab),
                Mode::Sample { temperature } => {
                    let p: Vec<Vec<f32>> = logits
                        .chunks_exact(vocab)
                        .map(|row| probabilities(row, temperature))
                        .collect();
                    verify_sampled(&draft.tokens, &draft.proposals, &p, &mut rng)
                }
            };
            // The cache now holds the input's positions; forget the ones
            // after the last kept draft token.
            let rejected = draft.tokens.len() - kept;
            self.cache.truncate(self.cache.len() - rejected);
            stats.verify_time += start.elapsed();

            stats.target_passes += 1;
            stats.drafted += draft.tokens.len();
            stats.accepted += kept;
            let new = draft.tokens[..kept].iter().chain([&next]);
            if let Some(stop) = new.clone().position(|t| g.stop.contains(t)) {
                tokens.extend(new.take(stop));
                break;
            }
            tokens.extend(new);
        }
        (tokens.split_off(prompt.len()), stats)
    }
}
