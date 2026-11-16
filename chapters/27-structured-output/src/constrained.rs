//! Generation that can only produce JSON matching a schema.
//!
//! Every step: compute the token mask from the matcher, set the logits of
//! every other token to minus infinity, sample as usual, and feed the
//! chosen token's bytes to the matcher. The sampler never sees a token
//! that would break the JSON, so the result is valid whatever the model
//! would have preferred. If the token budget runs out before the value is
//! complete, the matcher's shortest completion closes it.

use crate::json::Matcher;
use crate::mask::TokenTrie;
use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::Sampler;
use std::time::{Duration, Instant};

/// The vocabulary as the constrained generator needs it.
pub struct Vocab<'v> {
    /// Each token's bytes (empty for special tokens, which never match).
    pub bytes: Vec<&'v [u8]>,
    pub trie: TokenTrie,
    /// The end-of-sequence token: allowed only once the value is complete.
    pub eos: u32,
}

impl<'v> Vocab<'v> {
    /// `bytes[id]` for every token; `special` tokens are left out of the
    /// trie (they stand for markup, not text).
    pub fn new(bytes: Vec<&'v [u8]>, special: impl Fn(u32) -> bool, eos: u32) -> Self {
        let trie = TokenTrie::new(
            bytes
                .iter()
                .enumerate()
                .map(|(id, &b)| (id as u32, b))
                .filter(|&(id, b)| !b.is_empty() && !special(id)),
        );
        Self { bytes, trie, eos }
    }
}

/// What one constrained generation did.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// The JSON text (the generated tokens' bytes, plus any completion).
    pub text: String,
    pub tokens: Vec<u32>,
    /// Bytes appended by the matcher because the budget ran out.
    pub completed: usize,
    /// Time spent on the prompt, computing masks, and in the model's
    /// decode steps.
    pub prefill_time: Duration,
    pub mask_time: Duration,
    pub model_time: Duration,
}

/// Generates a value that `matcher` accepts (a fresh matcher for the
/// schema wanted) after `prompt`, with at most `max_tokens` sampled tokens.
pub fn generate_json<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    vocab: &Vocab<'_>,
    mut matcher: Matcher<'_>,
    sampler: &mut Sampler,
    prompt: &[u32],
    max_tokens: usize,
) -> Output {
    let mut out = Output::default();
    let mut bytes = Vec::new();
    let mut mask = vec![false; vocab.bytes.len()];
    cache.clear();
    sampler.start(prompt);
    let start = Instant::now();
    let mut logits = model.forward_last(pool, prompt, cache, scratch).to_vec();
    out.prefill_time = start.elapsed();
    for _ in 0..max_tokens {
        if matcher.is_finished() {
            break;
        }
        let start = Instant::now();
        mask.fill(false);
        vocab.trie.allowed(&matcher, &mut mask);
        mask[vocab.eos as usize] = matcher.is_complete();
        for (l, &ok) in logits.iter_mut().zip(&mask) {
            if !ok {
                *l = f32::NEG_INFINITY;
            }
        }
        out.mask_time += start.elapsed();
        let token = sampler.sample(&logits);
        if token == vocab.eos {
            break;
        }
        let ok = matcher.push_all(vocab.bytes[token as usize]);
        assert!(ok, "the mask allowed a token the matcher rejects");
        bytes.extend_from_slice(vocab.bytes[token as usize]);
        out.tokens.push(token);
        let start = Instant::now();
        logits.copy_from_slice(model.forward_last(pool, &[token], cache, scratch));
        out.model_time += start.elapsed();
    }
    if !matcher.is_complete() {
        let tail = matcher.completion();
        out.completed = tail.len();
        bytes.extend_from_slice(&tail);
    }
    // The matcher accepts only valid UTF-8 in strings and ASCII elsewhere,
    // and its completion finishes a character cut by the budget.
    out.text = String::from_utf8(bytes).expect("the matcher only accepts UTF-8");
    out
}
