//! Drafters: cheap ways to guess the next few tokens.
//!
//! - [`ModelDrafter`]: a smaller model of the same family (same tokenizer),
//!   generating `k` tokens one at a time with its own KV cache.
//! - [`NgramDrafter`]: no model at all. If the last few tokens appeared
//!   earlier in the text, guess that what followed them then follows them
//!   now ("prompt lookup decoding"). Free, and very good whenever the
//!   answer copies from the prompt.

use crate::verify::{Proposal, argmax, draw, probabilities};
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::Rng;

/// How tokens are chosen: the most likely one, or drawn at a temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Greedy,
    Sample { temperature: f32 },
}

/// Draft tokens, and how each was chosen (used by rejection sampling).
#[derive(Clone, Debug, Default)]
pub struct Draft {
    pub tokens: Vec<u32>,
    pub proposals: Vec<Proposal>,
}

pub trait Drafter {
    /// Guesses up to `k` tokens that follow `context` (the prompt and
    /// every token accepted so far).
    fn propose(
        &mut self,
        pool: &mut SpinPool,
        context: &[u32],
        k: usize,
        mode: Mode,
        rng: &mut Rng,
    ) -> Draft;
}

/// A small model that drafts by generating, with a KV cache it keeps
/// between calls.
pub struct ModelDrafter<W: Matrix> {
    pub model: Model<W>,
    cache: KvCache,
    scratch: Scratch,
    /// The tokens whose keys and values are in `cache`.
    cached: Vec<u32>,
}

impl<W: Matrix> ModelDrafter<W> {
    pub fn new(model: Model<W>, context: usize) -> Self {
        let cache = KvCache::new(&model.config, context);
        let scratch = Scratch::new(&model.config, 256, context);
        Self {
            model,
            cache,
            scratch,
            cached: Vec::new(),
        }
    }
}

impl<W: Matrix> Drafter for ModelDrafter<W> {
    fn propose(
        &mut self,
        pool: &mut SpinPool,
        context: &[u32],
        k: usize,
        mode: Mode,
        rng: &mut Rng,
    ) -> Draft {
        let mut draft = Draft::default();
        if k == 0 {
            return draft;
        }
        // Keep the cached positions that still match the context (the
        // draft tokens the target accepted), drop the rest, and feed what
        // is new: at least the context's last token, whose logits give the
        // first guess.
        let same = self
            .cached
            .iter()
            .zip(context)
            .take_while(|(a, b)| a == b)
            .count()
            .min(context.len() - 1);
        self.cache.truncate(same);
        self.cached.truncate(same);
        self.cached.extend_from_slice(&context[same..]);
        let mut logits =
            self.model
                .forward_last(pool, &context[same..], &mut self.cache, &mut self.scratch);
        for i in 0..k {
            let token = match mode {
                Mode::Greedy => {
                    draft.proposals.push(Proposal::Exact);
                    argmax(logits)
                }
                Mode::Sample { temperature } => {
                    let q = probabilities(logits, temperature);
                    let token = draw(&q, rng);
                    draft.proposals.push(Proposal::Sampled(q));
                    token
                }
            };
            draft.tokens.push(token);
            if i + 1 < k {
                logits =
                    self.model
                        .forward_last(pool, &[token], &mut self.cache, &mut self.scratch);
                self.cached.push(token);
            }
        }
        draft
    }
}

/// Prompt lookup: finds the most recent earlier occurrence of the last
/// `n` tokens (trying `max_n` down to `min_n`) and proposes the tokens that
/// followed it.
pub struct NgramDrafter {
    pub min_n: usize,
    pub max_n: usize,
}

impl Drafter for NgramDrafter {
    fn propose(
        &mut self,
        _pool: &mut SpinPool,
        context: &[u32],
        k: usize,
        _mode: Mode,
        _rng: &mut Rng,
    ) -> Draft {
        let len = context.len();
        for n in (self.min_n.max(1)..=self.max_n.min(len.saturating_sub(1))).rev() {
            let tail = &context[len - n..];
            // The latest start before the tail itself.
            if let Some(start) = (0..len - n).rev().find(|&s| &context[s..s + n] == tail) {
                let tokens: Vec<u32> = context[start + n..].iter().take(k).copied().collect();
                return Draft {
                    proposals: vec![Proposal::Exact; tokens.len()],
                    tokens,
                };
            }
        }
        Draft::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_lookup_proposes_what_followed_the_last_occurrence() {
        let mut d = NgramDrafter { min_n: 1, max_n: 3 };
        let mut pool = SpinPool::new(1);
        let mut rng = Rng::new(0);
        let mut propose = |context: &[u32], k| {
            d.propose(&mut pool, context, k, Mode::Greedy, &mut rng)
                .tokens
        };
        // "5 6 7" appeared before, followed by 8 9 1.
        assert_eq!(propose(&[1, 5, 6, 7, 8, 9, 1, 2, 5, 6, 7], 3), [8, 9, 1]);
        // The latest occurrence wins, and a short continuation is fine.
        assert_eq!(propose(&[4, 2, 4, 3, 4], 4), [3, 4]);
        // No earlier occurrence of even the last token: nothing.
        assert_eq!(propose(&[1, 2, 3], 4), []);
        // With min_n = 2, a single repeated token is not enough.
        let mut d = NgramDrafter { min_n: 2, max_n: 3 };
        let draft = d.propose(&mut pool, &[4, 2, 4, 3, 4], 4, Mode::Greedy, &mut rng);
        assert_eq!(draft.tokens, []);
    }
}
