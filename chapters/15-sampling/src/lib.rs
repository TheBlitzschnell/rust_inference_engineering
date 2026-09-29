//! Chapter 15: sampling, the step that turns the model's logits into the
//! next token.
//!
//! The pipeline, in the order production engines apply it:
//!
//! 1. penalties for tokens already seen (repetition, frequency, presence),
//! 2. temperature (or greedy when it is 0),
//! 3. top-k, then softmax, then top-p and min-p,
//! 4. a random draw from what is left, with a per-request seeded generator.
//!
//! Plus the pieces around it: log-probabilities, stop sequences that span
//! token boundaries, and a generation loop on chapter 14's engine.

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

/// Temperatures below this are treated as greedy (dividing by a tiny
/// temperature would overflow `f32`). vLLM uses the same threshold.
pub const GREEDY_BELOW: f32 = 1e-5;

/// How to choose the next token. The defaults change nothing: temperature 1,
/// every filter and penalty off.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// Divides the logits. 0 means greedy; below 1 sharpens, above 1
    /// flattens the distribution.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens (0 = off).
    pub top_k: usize,
    /// Keep the smallest set of most likely tokens whose probabilities add up
    /// to at least `top_p` (1.0 = off).
    pub top_p: f32,
    /// Drop tokens less likely than `min_p` times the most likely one
    /// (0.0 = off).
    pub min_p: f32,
    /// Hugging Face style: logits of tokens seen in the prompt or the output
    /// are divided by this if positive, multiplied if negative (1.0 = off).
    pub repetition_penalty: f32,
    /// OpenAI style: subtract `frequency_penalty × count` from the logit of
    /// every token already generated (0.0 = off).
    pub frequency_penalty: f32,
    /// OpenAI style: subtract `presence_penalty` once from the logit of every
    /// token already generated (0.0 = off).
    pub presence_penalty: f32,
    /// Seed of this request's random number generator.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

impl SamplingParams {
    /// Always the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature < GREEDY_BELOW
    }

    #[expect(
        clippy::float_cmp,
        reason = "1.0 and 0.0 are the exact 'off' values a user sets, not computed results"
    )]
    pub fn has_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }
}

/// SplitMix64: a small, fast, seedable generator with good statistical
/// quality (it passes BigCrush). Not for cryptography.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1), with 53 random bits (every `f64` step).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A token that may still be chosen, with its adjusted logit and, once
/// computed, its probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub token: u32,
    pub logit: f32,
    pub p: f32,
}

/// Chooses tokens for one request. It owns the request's random generator
/// and the history the penalties need, so requests never influence each
/// other (chapter 23 batches many of them).
pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    prompt_tokens: HashSet<u32>,
    output_counts: HashMap<u32, u32>,
    candidates: Vec<Candidate>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = Rng::new(params.seed);
        Self {
            params,
            rng,
            prompt_tokens: HashSet::new(),
            output_counts: HashMap::new(),
            candidates: Vec::new(),
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Starts a new sequence: remembers the prompt (for the repetition
    /// penalty), forgets earlier output and reseeds the generator, so the
    /// same request with the same seed gives the same tokens.
    pub fn start(&mut self, prompt: &[u32]) {
        self.prompt_tokens = prompt.iter().copied().collect();
        self.output_counts.clear();
        self.rng = Rng::new(self.params.seed);
    }

    /// Records a generated token, for the penalties. [`Sampler::sample`]
    /// does this itself.
    pub fn record(&mut self, token: u32) {
        *self.output_counts.entry(token).or_insert(0) += 1;
    }

    /// Chooses the next token from `logits` and records it.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.distribution(logits);
        let token = draw(&self.candidates, self.rng.next_f64());
        self.record(token);
        token
    }

    /// The distribution the next token is drawn from: every token that can
    /// still be chosen, with its probability. Sorted from most to least
    /// likely when top-p is on; in no particular order otherwise.
    #[expect(
        clippy::float_cmp,
        reason = "1.0 and 0.0 are the exact 'off' values a user sets, not computed results"
    )]
    pub fn distribution(&mut self, logits: &[f32]) -> &[Candidate] {
        let p = &self.params;
        let c = &mut self.candidates;
        c.clear();
        // Fast path: greedy without penalties is a plain argmax over the
        // logits, with no candidate list to build.
        if p.is_greedy() && !p.has_penalties() {
            let (token, &logit) = logits
                .iter()
                .enumerate()
                .reduce(|best, x| if x.1 > best.1 { x } else { best })
                .expect("empty logits");
            c.push(Candidate {
                token: token as u32,
                logit,
                p: 1.0,
            });
            return c;
        }
        c.extend(logits.iter().enumerate().map(|(i, &logit)| Candidate {
            token: i as u32,
            logit,
            p: 0.0,
        }));

        // 1. Penalties. `c` is still indexed by token id here.
        if p.repetition_penalty != 1.0 {
            let seen = self.prompt_tokens.iter().chain(self.output_counts.keys());
            for &token in seen {
                if let Some(cand) = c.get_mut(token as usize) {
                    cand.logit = penalize(cand.logit, p.repetition_penalty);
                }
            }
        }
        if p.frequency_penalty != 0.0 || p.presence_penalty != 0.0 {
            for (&token, &count) in &self.output_counts {
                if let Some(cand) = c.get_mut(token as usize) {
                    cand.logit -= p.frequency_penalty * count as f32 + p.presence_penalty;
                }
            }
        }

        // 2. Greedy, or temperature.
        if p.is_greedy() {
            let best = c
                .iter()
                .copied()
                .reduce(|best, x| if x.logit > best.logit { x } else { best })
                .expect("empty logits");
            c.clear();
            c.push(Candidate { p: 1.0, ..best });
            return c;
        }
        if p.temperature != 1.0 {
            let inv = 1.0 / p.temperature;
            for cand in c.iter_mut() {
                cand.logit *= inv;
            }
        }

        // 3. Top-k: an O(n) partial selection, not a sort.
        if p.top_k > 0 && p.top_k < c.len() {
            c.select_nth_unstable_by(p.top_k - 1, |a, b| b.logit.total_cmp(&a.logit));
            c.truncate(p.top_k);
        }
        softmax(c);

        // Top-p needs the candidates in order; sorting is the expensive part.
        if p.top_p < 1.0 {
            c.sort_unstable_by(|a, b| b.p.total_cmp(&a.p));
            let mut cumulative = 0.0;
            let mut keep = c.len();
            for (i, cand) in c.iter().enumerate() {
                cumulative += cand.p;
                if cumulative >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
            c.truncate(keep);
            renormalize(c);
        }
        if p.min_p > 0.0 {
            let max = c.iter().map(|x| x.p).fold(0.0, f32::max);
            let threshold = p.min_p * max;
            c.retain(|x| x.p >= threshold);
            renormalize(c);
        }
        c
    }
}

/// Hugging Face's repetition penalty. It depends on the sign of the logit,
/// which is arbitrary: adding a constant to every logit leaves the
/// probabilities unchanged but changes what this does (see the lesson).
fn penalize(logit: f32, penalty: f32) -> f32 {
    if logit > 0.0 {
        logit / penalty
    } else {
        logit * penalty
    }
}

/// Softmax over the candidates' logits, into their `p`. Tokens with a logit
/// of −∞ (masked, chapter 27) get probability 0.
fn softmax(c: &mut [Candidate]) {
    let max = c.iter().map(|x| x.logit).fold(f32::NEG_INFINITY, f32::max);
    assert!(max > f32::NEG_INFINITY, "every token is masked");
    let mut sum = 0.0;
    for cand in c.iter_mut() {
        cand.p = (cand.logit - max).exp();
        sum += cand.p;
    }
    for cand in c.iter_mut() {
        cand.p /= sum;
    }
}

fn renormalize(c: &mut [Candidate]) {
    let sum: f32 = c.iter().map(|x| x.p).sum();
    for cand in c.iter_mut() {
        cand.p /= sum;
    }
}

/// Inverse-CDF sampling: walk the candidates, subtracting probabilities from
/// `u × total` until it drops below zero. `u` is uniform in [0, 1).
pub fn draw(candidates: &[Candidate], u: f64) -> u32 {
    let total: f64 = candidates.iter().map(|x| f64::from(x.p)).sum();
    let mut remaining = u * total;
    for cand in candidates {
        remaining -= f64::from(cand.p);
        if remaining < 0.0 {
            return cand.token;
        }
    }
    // Rounding can leave a tiny positive remainder: take the last token
    // with a nonzero probability.
    candidates
        .iter()
        .rev()
        .find(|x| x.p > 0.0)
        .expect("no candidate has a nonzero probability")
        .token
}

/// The model's log-probability of `token` (from the raw logits, before any
/// sampling adjustments), as reported by `logprobs` in OpenAI-style APIs.
pub fn logprob(logits: &[f32], token: u32) -> f32 {
    logits[token as usize] - log_sum_exp(logits)
}

/// The `n` most likely tokens and their log-probabilities, best first.
pub fn top_logprobs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let lse = log_sum_exp(logits);
    let mut all: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l - lse))
        .collect();
    let n = n.min(all.len());
    if n > 0 && n < all.len() {
        all.select_nth_unstable_by(n - 1, |a, b| b.1.total_cmp(&a.1));
    }
    all.truncate(n);
    all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    all
}

/// `ln(Σ exp(x_i))`, computed stably by factoring out the maximum.
pub fn log_sum_exp(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Finds stop sequences in streamed text, which may be split across tokens.
///
/// Text is released only once it cannot be the beginning of a stop
/// sequence: with the stop sequence `"\nUser:"`, a generated `"\n"` is held
/// back until the next piece shows whether `"User:"` follows.
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

/// What [`StopMatcher::push`] found.
#[derive(Debug, PartialEq, Eq)]
pub enum Matched {
    /// No stop sequence yet; this text is safe to show.
    Continue(String),
    /// A stop sequence was found; this is the text before it, and generation
    /// should end. The stop sequence itself is not included.
    Stop(String),
}

impl StopMatcher {
    pub fn new<S: AsRef<str>>(stops: &[S]) -> Self {
        Self {
            stops: stops
                .iter()
                .map(|s| s.as_ref().to_owned())
                .filter(|s| !s.is_empty())
                .collect(),
            pending: String::new(),
        }
    }

    /// Adds newly generated text.
    pub fn push(&mut self, text: &str) -> Matched {
        self.pending.push_str(text);
        // The earliest complete stop sequence wins.
        let found = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min();
        if let Some(at) = found {
            let before = self.pending[..at].to_owned();
            self.pending.clear();
            return Matched::Stop(before);
        }
        // Hold back the longest tail that could still grow into a stop.
        let hold = self.longest_partial_match();
        let release = self.pending.len() - hold;
        let out = self.pending[..release].to_owned();
        self.pending.drain(..release);
        Matched::Continue(out)
    }

    /// Generation ended for another reason: release what was held back.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Length in bytes of the longest suffix of `pending` that is a proper
    /// prefix of some stop sequence.
    fn longest_partial_match(&self) -> usize {
        let text = &self.pending;
        // Candidate suffixes start at character boundaries only.
        text.char_indices()
            .map(|(i, _)| i)
            .find(|&i| {
                let tail = &text[i..];
                self.stops
                    .iter()
                    .any(|s| s.len() > tail.len() && s.starts_with(tail))
            })
            .map_or(0, |i| text.len() - i)
    }
}

/// Why generation ended, as reported by OpenAI-style APIs (`finish_reason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// `max_new_tokens` reached ("length").
    Length,
    /// The model produced a stop token, such as end-of-sequence ("stop").
    StopToken,
    /// The caller's callback asked to stop: a stop string matched, or the
    /// client went away.
    Stopped,
}

/// Samples up to `max_new_tokens` tokens after `prompt`, calling `on_token`
/// with each one (stop tokens excluded). Return `ControlFlow::Break(())` from
/// the callback to end generation early.
pub fn generate<W: Matrix>(
    model: &Model<W>,
    pool: &mut SpinPool,
    prompt: &[u32],
    sampler: &mut Sampler,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    max_new_tokens: usize,
    stop_tokens: &[u32],
    mut on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> FinishReason {
    cache.clear();
    sampler.start(prompt);
    let mut next = None;
    for _ in 0..max_new_tokens {
        let logits = match next {
            None => model.forward_last(pool, prompt, cache, scratch),
            Some(token) => model.forward_last(pool, &[token], cache, scratch),
        };
        let token = sampler.sample(logits);
        if stop_tokens.contains(&token) {
            return FinishReason::StopToken;
        }
        if on_token(token).is_break() {
            return FinishReason::Stopped;
        }
        next = Some(token);
    }
    FinishReason::Length
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "the penalties are exact on these small values"
)]
mod tests {
    use super::*;

    const LOGITS: [f32; 6] = [2.0, 1.0, 0.5, 0.0, -1.0, -3.0];

    fn probs(logits: &[f32]) -> Vec<f64> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = logits.iter().map(|&l| f64::from(l - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        e.iter().map(|x| x / sum).collect()
    }

    fn tokens(c: &[Candidate]) -> Vec<u32> {
        let mut t: Vec<u32> = c.iter().map(|x| x.token).collect();
        t.sort_unstable();
        t
    }

    #[test]
    fn greedy_picks_the_largest_logit() {
        let mut s = Sampler::new(SamplingParams::greedy());
        for _ in 0..10 {
            assert_eq!(s.sample(&LOGITS), 0);
        }
    }

    #[test]
    fn a_tiny_temperature_is_greedy_and_never_nan() {
        let params = SamplingParams {
            temperature: 1e-30,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let d = s.distribution(&LOGITS);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].token, 0);
        assert!(d.iter().all(|x| x.p.is_finite()));
    }

    #[test]
    fn sampled_frequencies_match_the_softmax() {
        let mut s = Sampler::new(SamplingParams::default());
        let n = 200_000;
        let mut counts = [0usize; 6];
        for _ in 0..n {
            counts[s.sample(&LOGITS) as usize] += 1;
        }
        for (count, p) in counts.iter().zip(probs(&LOGITS)) {
            let observed = *count as f64 / f64::from(n);
            // Several standard errors of a binomial proportion.
            let tolerance = 5.0 * (p * (1.0 - p) / f64::from(n)).sqrt() + 1e-4;
            assert!((observed - p).abs() < tolerance, "{observed} vs {p}");
        }
    }

    #[test]
    fn temperature_divides_the_logits() {
        let params = SamplingParams {
            temperature: 2.0,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let d = s.distribution(&LOGITS).to_vec();
        let halved: Vec<f32> = LOGITS.iter().map(|l| l / 2.0).collect();
        for (cand, want) in d.iter().zip(probs(&halved)) {
            assert!((f64::from(cand.p) - want).abs() < 1e-6);
        }
    }

    #[test]
    fn top_k_keeps_the_k_most_likely() {
        let params = SamplingParams {
            top_k: 3,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        let d = s.distribution(&LOGITS);
        assert_eq!(tokens(d), [0, 1, 2]);
        let sum: f32 = d.iter().map(|x| x.p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_p_keeps_the_smallest_set_reaching_p() {
        // Probabilities: 0.557, 0.205, 0.124, 0.075, 0.028, 0.004.
        for (top_p, want) in [(0.5, 1), (0.6, 2), (0.76, 2), (0.77, 3), (0.99, 5)] {
            let params = SamplingParams {
                top_p,
                ..SamplingParams::default()
            };
            let mut s = Sampler::new(params);
            assert_eq!(s.distribution(&LOGITS).len(), want, "top_p = {top_p}");
        }
    }

    #[test]
    fn min_p_drops_tokens_far_below_the_best() {
        let params = SamplingParams {
            min_p: 0.2, // keep p >= 0.2 × 0.557 = 0.111
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(params);
        assert_eq!(tokens(s.distribution(&LOGITS)), [0, 1, 2]);
    }

    #[test]
    fn masked_tokens_are_never_chosen() {
        let mut logits = LOGITS;
        logits[0] = f32::NEG_INFINITY;
        let mut s = Sampler::new(SamplingParams::default());
        for _ in 0..10_000 {
            assert_ne!(s.sample(&logits), 0);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_tokens() {
        let run = |seed| {
            let mut s = Sampler::new(SamplingParams {
                seed,
                ..SamplingParams::default()
            });
            (0..50).map(|_| s.sample(&LOGITS)).collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn restarting_reseeds_and_forgets_history() {
        let mut s = Sampler::new(SamplingParams {
            seed: 3,
            frequency_penalty: 0.5,
            ..SamplingParams::default()
        });
        s.start(&[]);
        let first: Vec<u32> = (0..20).map(|_| s.sample(&LOGITS)).collect();
        s.start(&[]);
        let second: Vec<u32> = (0..20).map(|_| s.sample(&LOGITS)).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn penalties_lower_the_logits_of_seen_tokens() {
        let mut s = Sampler::new(SamplingParams {
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        });
        s.start(&[0, 4]); // token 0 has a positive logit, token 4 a negative one
        let d = s.distribution(&LOGITS).to_vec();
        assert_eq!(d[0].logit, 1.0); // 2.0 / 2
        assert_eq!(d[4].logit, -2.0); // -1.0 × 2
        assert_eq!(d[1].logit, 1.0); // unseen, unchanged

        let mut s = Sampler::new(SamplingParams {
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
            ..SamplingParams::default()
        });
        s.start(&[1]); // the prompt does not count for these two
        s.record(0);
        s.record(0);
        let d = s.distribution(&LOGITS).to_vec();
        assert_eq!(d[0].logit, 2.0 - 0.5 * 2.0 - 0.25);
        assert_eq!(d[1].logit, 1.0);
    }

    #[test]
    fn draw_walks_the_cumulative_distribution() {
        let c = [
            Candidate {
                token: 10,
                logit: 0.0,
                p: 0.5,
            },
            Candidate {
                token: 20,
                logit: 0.0,
                p: 0.3,
            },
            Candidate {
                token: 30,
                logit: 0.0,
                p: 0.2,
            },
        ];
        assert_eq!(draw(&c, 0.0), 10);
        assert_eq!(draw(&c, 0.49), 10);
        assert_eq!(draw(&c, 0.51), 20);
        assert_eq!(draw(&c, 0.79), 20);
        assert_eq!(draw(&c, 0.81), 30);
        assert_eq!(draw(&c, 0.999_999), 30);
    }

    #[test]
    fn logprobs_are_log_softmax() {
        let p = probs(&LOGITS);
        for t in 0..6 {
            assert!((f64::from(logprob(&LOGITS, t)) - p[t as usize].ln()).abs() < 1e-5);
        }
        let top = top_logprobs(&LOGITS, 2);
        assert_eq!(top.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1]);
        assert!(top[0].1 > top[1].1);
    }

    mod generation {
        use super::super::*;
        use ch13_transformer::{Config, Weights};
        use ch14_kv_cache::DenseF32;

        fn setup() -> (Model<DenseF32>, SpinPool, KvCache, Scratch) {
            let c = Config::tiny();
            let model = Model::from_reference(&Weights::random(&c, 1));
            let (cache, scratch) = (KvCache::new(&c, 64), Scratch::new(&c, 16, 64));
            (model, SpinPool::new(2), cache, scratch)
        }

        fn run(
            params: SamplingParams,
            stop_tokens: &[u32],
            max: usize,
        ) -> (Vec<u32>, FinishReason) {
            let (model, mut pool, mut cache, mut scratch) = setup();
            let mut sampler = Sampler::new(params);
            let mut out = Vec::new();
            let reason = generate(
                &model,
                &mut pool,
                &[1, 2, 3],
                &mut sampler,
                &mut cache,
                &mut scratch,
                max,
                stop_tokens,
                |t| {
                    out.push(t);
                    ControlFlow::Continue(())
                },
            );
            (out, reason)
        }

        #[test]
        fn greedy_generation_matches_chapter_14() {
            let (model, mut pool, mut cache, mut scratch) = setup();
            let want = ch14_kv_cache::generate_greedy(
                &model,
                &mut pool,
                &[1, 2, 3],
                10,
                &mut cache,
                &mut scratch,
                |_, _| {},
            );
            let (got, reason) = run(SamplingParams::greedy(), &[], 10);
            assert_eq!(got, want[3..]);
            assert_eq!(reason, FinishReason::Length);
        }

        #[test]
        fn sampling_is_reproducible_with_a_seed() {
            let params = SamplingParams {
                temperature: 1.5,
                seed: 42,
                ..SamplingParams::default()
            };
            assert_eq!(run(params.clone(), &[], 20), run(params, &[], 20));
        }

        #[test]
        fn a_stop_token_ends_generation_and_is_not_emitted() {
            let (all, _) = run(SamplingParams::greedy(), &[], 10);
            let stop = all[4];
            let (out, reason) = run(SamplingParams::greedy(), &[stop], 10);
            assert_eq!(reason, FinishReason::StopToken);
            assert_eq!(out, all[..all.iter().position(|&t| t == stop).unwrap()]);
        }

        #[test]
        fn the_callback_can_stop_generation() {
            let (model, mut pool, mut cache, mut scratch) = setup();
            let mut sampler = Sampler::new(SamplingParams::greedy());
            let mut seen = 0;
            let reason = generate(
                &model,
                &mut pool,
                &[1, 2, 3],
                &mut sampler,
                &mut cache,
                &mut scratch,
                10,
                &[],
                |_| {
                    seen += 1;
                    if seen == 3 {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                },
            );
            assert_eq!((seen, reason), (3, FinishReason::Stopped));
            assert_eq!(cache.len(), 3 + 2); // the third token was never fed back
        }
    }

    #[test]
    fn stop_sequences_are_found_across_pieces() {
        let mut m = StopMatcher::new(&["\nUser:"]);
        assert_eq!(m.push("Hello"), Matched::Continue("Hello".into()));
        assert_eq!(m.push(" there\n"), Matched::Continue(" there".into()));
        assert_eq!(m.push("Us"), Matched::Continue(String::new()));
        assert_eq!(m.push("er: hi"), Matched::Stop(String::new()));
    }

    #[test]
    fn a_false_start_is_released() {
        let mut m = StopMatcher::new(&["\nUser:"]);
        assert_eq!(m.push("a\nU"), Matched::Continue("a".into()));
        assert_eq!(m.push("nder"), Matched::Continue("\nUnder".into()));
        assert_eq!(m.push("\n"), Matched::Continue(String::new()));
        assert_eq!(m.finish(), "\n");
    }

    #[test]
    fn the_earliest_stop_sequence_wins_and_text_before_it_is_kept() {
        let mut m = StopMatcher::new(&["END", "STOP"]);
        assert_eq!(m.push("abc STOP def END"), Matched::Stop("abc ".into()));
    }

    #[test]
    fn holding_back_respects_character_boundaries() {
        let mut m = StopMatcher::new(&["éclair"]);
        assert_eq!(m.push("un é"), Matched::Continue("un ".into()));
        assert_eq!(m.push("t"), Matched::Continue("ét".into()));
    }
}
