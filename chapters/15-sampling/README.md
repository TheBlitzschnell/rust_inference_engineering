# Chapter 15: Sampling

> **In one sentence:** the model only scores every possible next token; **sampling** is the separate, cheap, surprisingly consequential step that turns those scores into one chosen token, by adjusting them (penalties, temperature), cutting off the unlikely tail (top-k, top-p, min-p) and drawing at random with a per-request seed, and around it sit the rules for stopping (stop tokens, stop strings, length).

**Where this fits:** chapter 14's engine returns logits and picks the largest. Every real deployment offers more choices than that, and every API request carries them (`temperature`, `top_p`, `seed`, `stop`...). This chapter builds the sampler that chapter 16 uses to talk to SmolLM2, chapter 22 exposes over HTTP, chapter 26 extends for speculative decoding and chapter 27 constrains to valid JSON.

**You need:** chapter 8 (softmax), chapter 14 (the engine).

**You will build:** a `Sampler` with temperature, top-k, top-p, min-p, three kinds of penalty and a seeded generator; log-probabilities; a stop-sequence matcher for streamed text; and a generation loop with finish reasons. The demo measures what each setting costs and shows what each does to text.

---

## 1. The intuition

Think of a raffle. For every candidate next token, the model hands out tickets in proportion to how likely it thinks that token is. The sampler draws one ticket.

- **Greedy** decoding skips the raffle and always takes the token with the most tickets.
- **Temperature** redistributes tickets before the draw: a low temperature takes tickets from the long shots and gives them to the favourites; a high one does the opposite.
- **Top-k, top-p and min-p** throw away the tickets of the long shots entirely, in three different ways.
- **Penalties** confiscate tickets from tokens that were already used.
- **The seed** fixes how the drum spins, so the same raffle can be replayed.

**Where the analogy breaks:** raffles are independent, and generation is not. The token drawn now becomes part of the input for every later step, so one unlucky draw of a long shot (a nonsense word in the middle of a sentence) sends the rest of the text somewhere the model would never have gone. That is why nearly every setup cuts the tail rather than sampling from the model's full distribution.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Logits** | The model's raw scores, one per vocabulary token. |
| **Distribution** | Probabilities over tokens, from `softmax(logits)`. |
| **Greedy decoding** | Always choosing the most likely token. Deterministic. |
| **Temperature (T)** | Logits are divided by T before the softmax. T < 1 sharpens, T > 1 flattens, T → 0 is greedy. |
| **Top-k** | Keep only the k most likely tokens. |
| **Top-p (nucleus)** | Keep the smallest set of most likely tokens whose probabilities add up to at least p. |
| **Min-p** | Keep tokens whose probability is at least `min_p` times that of the most likely token. |
| **Repetition penalty** | Hugging Face style: shrink the logits of tokens already seen, by division or multiplication. |
| **Frequency / presence penalty** | OpenAI style: subtract from the logits of generated tokens, per occurrence / once. |
| **Seed** | The starting state of the random generator; same seed, same draws. |
| **Logprob** | The natural log of a token's probability, reported by APIs to show the model's confidence. |
| **Stop token** | A token (like `<|im_end|>`) that ends generation when produced. |
| **Stop sequence** | A string (like `"\nUser:"`) that ends generation when it appears in the text. |
| **Finish reason** | Why generation ended: `length`, `stop` (token or string) or cancellation. |

## 3. The concepts in depth

### 3.1 From logits to a distribution

Chapter 8's softmax turns logits into probabilities: `p_i = exp(z_i) / Σ exp(z_j)`. Two properties matter here:

- **Only differences between logits matter.** Adding the same constant to every logit changes nothing, because it multiplies numerator and denominator by the same factor. Every sampling setting should respect that; section 3.5 shows one that does not.
- **Order is preserved**, so the most likely token is the one with the largest logit, which is why greedy needs no softmax at all.

Part 1 of the demo takes six logits for the continuation of "The capital of France is" and shows the distribution each setting produces:

```text
== 1. "The capital of France is" -> next-token probabilities
   setting                   Paris     Lyon   France      the        a   banana
   temperature 1             0.679    0.151    0.092    0.056    0.020    0.002
   temperature 0.5           0.930    0.046    0.017    0.006    0.001    0.000
   temperature 2             0.425    0.201    0.157    0.122    0.074    0.021
   greedy (temperature 0)    1.000        -        -        -        -        -
   top-k 4                   0.694    0.155    0.094    0.057        -        -
   top-p 0.8                 0.818    0.182        -        -        -        -
   min-p 0.1                 0.736    0.164    0.100        -        -        -
```

A dash means the token can no longer be chosen. Each row is what the sampler draws from.

### 3.2 Greedy, and why it is not enough

Greedy decoding is deterministic, cheap and often right for short factual answers, classification, extraction and code, where there is one best answer and variety is a defect. Evaluation benchmarks usually run greedy so that results are reproducible.

Its known failure is **repetition**. Part 4 of the demo uses a word-pair (bigram) model built from the first six chapters of *Pride and Prejudice*: its "logits" for the next word are the log-frequencies of the words that followed the current one in the book. Greedy decoding from "It":

```text
It was a very much as he had been of the two or views of the two or views of the two or views of
```

Once it reaches "of", the most likely continuation leads back to "of", forever. A bigram model remembers only one word, so it falls into the loop immediately, but real language models do the same thing on a longer scale, especially small ones: always taking the locally most likely word leads to bland, looping text. Holtzman et al. (2020) called this "neural text degeneration" and showed that human text is regularly *not* the most likely continuation.

### 3.3 Temperature

Temperature divides every logit by T before the softmax: `p_i ∝ exp(z_i / T)`.

- **T = 1** is the model's own distribution.
- **T < 1** stretches the differences between logits, so the favourite gains: " Paris" goes from 0.679 to 0.930 at T = 0.5.
- **T > 1** shrinks them, so the tail gains: " banana" goes from 0.002 to 0.021 at T = 2, ten times likelier.
- **T → 0** is greedy. Dividing by a tiny T can overflow `f32`: at T = 0, or at T = 1e-39, `1 / T` is infinite, the logits become ±infinity (and 0 × infinity is NaN), and infinity minus infinity in the softmax is NaN. So the sampler treats any T below `GREEDY_BELOW = 1e-5` as greedy, as vLLM does.

### 3.4 Cutting the tail: top-k, top-p, min-p

All three remove unlikely tokens and **renormalize** the rest so they add up to 1 again. They differ in how they decide what "unlikely" means.

- **Top-k** keeps a fixed number. It ignores the shape of the distribution: with k = 40, a confident model keeps 39 tokens it would never say, and an uncertain one (the first word of a story) loses reasonable options.
- **Top-p** keeps the most likely tokens until their probabilities add up to at least p. It adapts: one token when the model is sure, hundreds when it is not. In part 1, p = 0.8 keeps " Paris" (0.679) and " Lyon" (0.679 + 0.151 = 0.830 ≥ 0.8). It needs the tokens in order of probability, which means sorting (section 3.8).
- **Min-p** keeps tokens with at least `min_p` times the probability of the best one. With min-p 0.1, the threshold is 0.0679, so " France" (0.092) stays and " the" (0.056) goes. Like top-p it adapts to the model's confidence, but it needs no sort, only the maximum.

### 3.5 Penalties

Penalties change the logits of tokens that already appeared, to discourage repetition.

- **Repetition penalty** (from the CTRL paper, the Hugging Face `repetition_penalty`): for every token in the prompt or the output, divide its logit by the penalty if it is positive, multiply it if negative. Both make the token less likely. In part 4, a penalty of 1.5 lets greedy decoding escape the loop once ("...of the two or views on the first entering a good fortune..."), but it cannot remove repetition completely.
- **Frequency penalty** (OpenAI): subtract `frequency_penalty × count` from the logit of every generated token. The more often a token appeared, the stronger the push.
- **Presence penalty** (OpenAI): subtract a constant once from every generated token, however often it appeared. It pushes towards new topics.

The repetition penalty has a flaw worth understanding: **it depends on the sign of the logit, and the sign is meaningless.** Section 3.1 showed that adding a constant to every logit leaves the model's probabilities unchanged. It does not leave the repetition penalty unchanged: a logit of 0.5 divided by 2 becomes 0.25 (0.25 lower), but after adding 10 to all logits, 10.5 divided by 2 becomes 5.25 (5.25 lower). The same model, the same probabilities, a very different penalty. The OpenAI penalties subtract, so they do not have this problem. The repetition penalty stays in use because models have been tuned with it.

Penalties are also blunt. Code, JSON and ordinary prose need to repeat tokens (`}`, `"`, "the"), and a penalty cannot tell necessary repetition from degenerate repetition. Most production defaults leave them off.

### 3.6 The order of operations

The sampler applies, in order:

```text
penalties → temperature (or greedy) → top-k → softmax → top-p → min-p → draw
```

This follows Hugging Face's order. It is a convention, not a law: other engines differ in places (some apply min-p before top-k and top-p), and llama.cpp lets the user arrange its "sampler chain" freely. The order changes the result. At T = 1, top-p 0.8 keeps 2 of part 1's tokens; applied after T = 2, when the distribution is flatter, it keeps 4 (exercise 1). When you compare your engine's output with another's at the same settings, check the order first.

### 3.7 Drawing, and seeds

With the final candidates and their probabilities, the draw is **inverse CDF sampling**: pick a uniform random number `u` in [0, 1), then walk the candidates subtracting probabilities from `u` until it drops below zero. A token with probability 0.3 owns a slice of width 0.3 of the interval, so it is chosen 30% of the time. Part 2 of the demo checks it:

```text
== 2. 100000 draws at temperature 1
    Paris   drawn  0.680 of the time, probability 0.679
    Lyon    drawn  0.151 of the time, probability 0.151
    France  drawn  0.090 of the time, probability 0.092
    the     drawn  0.055 of the time, probability 0.056
    a       drawn  0.021 of the time, probability 0.020
    banana  drawn  0.002 of the time, probability 0.002
```

The random numbers come from a **seeded generator owned by the request.** Two consequences:

- **Reproducibility.** The same prompt, settings and seed give the same tokens. APIs expose this as `seed`. `Sampler::start` reseeds, so rerunning a request reproduces it.
- **Independence.** If all requests shared one generator, the numbers request A received would depend on how many draws request B made before it, and A's output would change depending on who else was using the server. Chapter 23 runs many requests in one batch; each keeps its own sampler.

Reproducibility has a second requirement: the logits themselves must be identical from run to run. This engine's are, because every thread always computes the same outputs in the same order. On GPUs, the order of floating-point additions can depend on the batch size, so the same request can produce slightly different logits, and eventually different tokens, depending on what else is in the batch. Making kernels "batch-invariant" is an active topic in serving engines.

### 3.8 What sampling costs

Part 3 times one sampling step on 49,152 logits (SmolLM2's vocabulary, random values):

```text
== 3. time per sampled token, 49152 logits
   greedy                         66.8 µs
   temperature 1, no filter      317.1 µs
   top-k 50                      176.4 µs
   top-p 0.9                    1323.4 µs
   top-k 50 + top-p 0.9          131.7 µs
   min-p 0.05                    211.7 µs
   frequency penalty 0.5         217.7 µs
   top 5 logprobs                221.3 µs
```

Across four runs the numbers moved by up to 45% (greedy 61-67 µs, no filter 225-317 µs, top-k 50 121-176 µs, top-p 0.9 1.30-1.39 ms), but the ranking never changed:

- **Greedy** is one pass over the logits.
- **No filter** builds the candidate list and computes 49,152 exponentials.
- **Top-k 50** is cheaper than no filter: `select_nth_unstable_by` finds the 50 best in one O(n) pass, and the softmax then runs on 50 numbers.
- **Top-p alone is four to ten times more expensive** than any other setting, because it sorts all 49,152 candidates: 1.3 ms, 8% of a 17 ms decode step from chapter 14. **Top-k first** shrinks the sort to 50 elements. This is why most default configurations combine top-p with a top-k (llama.cpp defaults to top-k 40 with top-p 0.95).

For one sequence on a CPU, sampling is a few percent of a decode step. Its importance grows with batching: a GPU server decoding 256 sequences at once samples 256 times per step, from vocabularies of 128k-256k tokens. Engines move sampling onto the GPU and use algorithms that avoid the full sort (section 8).

### 3.9 Stopping

Generation ends for one of three reasons, which APIs report as `finish_reason`:

- **A stop token.** Chat models end their turn with a special token (SmolLM2: `<|im_end|>`, id 2). It is not part of the answer, so it is not shown.
- **The length limit** (`max_tokens`). Every request needs one; a model that never produces its stop token would otherwise run until the cache is full.
- **A stop sequence** chosen by the user, or the client going away.

Stop sequences are harder than they look, because they are *strings* and the model produces *tokens*. `"\nUser:"` may arrive as `"\n"`, `"User"`, `":"`, or as `".\nUs"`, `"er:"`, and the text before it must reach the user as it is generated. `StopMatcher` keeps the text that could still be the start of a stop sequence **held back**, releasing it only when the next piece shows it is not:

```text
pushed        released     held      why
"Hello"       "Hello"      ""        cannot start "\nUser:"
" there\n"    " there"     "\n"      "\n" could be the start
"Us"          ""           "\nUs"    still could be
"er: hi"      stop!                  "\nUser:" complete; " hi" is discarded
```

Releasing text before checking is the classic bug: the user sees `"\nUs"` and then the stream stops.

### 3.10 Log-probabilities

APIs can return, for every generated token, its log-probability and the most likely alternatives (`logprobs`, `top_logprobs`). They are used to measure confidence, to score answers, and (chapter 19) to compute perplexity. This chapter computes them from the model's raw logits, before temperature and filters: they describe the model, not the sampler. Engines differ on this point, so check which one yours reports.

`log_sum_exp` computes `ln Σ exp(z)` as `max + ln Σ exp(z − max)`, the same trick as the stable softmax, so a logprob is `z_i − log_sum_exp(z)`.

## 4. The code

The sampler, the stop matcher and the generation loop are in [`src/lib.rs`](src/lib.rs); the demo, including the bigram model, is in [`src/main.rs`](src/main.rs); the text is in [`data/`](data/pride-and-prejudice-1-6.txt) (public domain).

### 4.1 The parameters

<!-- file: src/lib.rs -->
```rust
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
```

Every field's default is its "off" value, so a request changes only what it names: `SamplingParams { top_p: 0.9, ..SamplingParams::default() }`. Chapter 22 maps the JSON fields of an API request onto this struct the same way.

### 4.2 The pipeline

`Sampler` owns the parameters, the generator, the history the penalties need, and a reusable `Vec<Candidate>`. `distribution` fills that vector with every token and its logit, then:

<!-- file: src/lib.rs -->
```rust
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
```

The penalties loop over the tokens seen so far (a few hundred), not over the vocabulary. The repetition penalty counts the prompt, the other two only the output, matching vLLM.

<!-- file: src/lib.rs -->
```rust
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
```

- `select_nth_unstable_by(k - 1, ...)` rearranges the vector so the k largest come first, in no particular order, in linear time. Top-k needs exactly that and no more.
- The comparisons use `total_cmp`, which orders every `f32`, NaN included (section 6).
- Top-p stops at the first token that brings the running sum to at least p, then renormalizes.

The greedy path returns early, before temperature, and when no penalty is active it skips the candidate list entirely and runs an argmax over the logits.

### 4.3 The draw

<!-- file: src/lib.rs -->
```rust
pub fn draw(candidates: &[Candidate], u: f64) -> u32 {
    let total: f64 = candidates.iter().map(|x| f64::from(x.p)).sum();
    let mut remaining = u * total;
    for cand in candidates {
        remaining -= f64::from(cand.p);
        if remaining < 0.0 {
            return cand.token;
        }
    }
```

The sum runs in `f64` and `u` is scaled by the actual total, so a distribution that adds up to 0.9999999 after rounding still samples correctly. If rounding leaves a tiny remainder after the last candidate, the function returns the last token with a nonzero probability.

The generator is SplitMix64: a 64-bit counter passed through a mixing function. It is small, fast, statistically strong for this purpose, and trivially seedable. `next_f64` takes the top 53 bits, exactly the precision of an `f64` in [0, 1).

### 4.4 Stop sequences

<!-- file: src/lib.rs -->
```rust
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
```

`longest_partial_match` tries the suffixes of the pending text from the longest down, starting only at character boundaries, and returns the first that is the beginning of some stop sequence. Starting at a byte inside a multi-byte character would make `&text[i..]` panic.

### 4.5 The generation loop

<!-- file: src/lib.rs -->
```rust
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
```

The callback returns `ControlFlow<()>`: `Continue(())` to go on, `Break(())` to stop. Chapter 16's command line uses it to stop on a stop sequence; chapter 21's engine uses it to stop when the client disconnects. A token is fed back into the model only if generation continues, so a stopped sequence leaves exactly the positions it used in the cache (a test checks this).

## 5. Run it

```bash
cargo test -p ch15-sampling
cargo run --release -p ch15-sampling
```

The output appears in sections 3.1 to 3.8, plus the text of part 4:

```text
== 4. a bigram model of Pride and Prejudice, chapters 1-6 (7694 words, 2186 distinct)
   greedy:
     It was a very much as he had been of the two or views of the two or views of the two or views of
   greedy, repetition penalty 1.5:
     It was a very much as he had been of the two or views on the first entering a good fortune must be so well
   temperature 1:
     It was struck with my dear. If my old friends. I do not believe he married a charming mother-in-law, indeed, nobody can, you will be
   temperature 1, top-p 0.5:
     It was highly favourable. Sir William at it, he had heard you may so very few of being gone to think of the last night
   temperature 0.7, top-p 0.9:
     It is considered as a man ought likewise a woman had a way to her twice. To be proud, to Sir William Lucas had been
```

A bigram model knows only which word follows which, so none of these are sensible sentences; the point is the difference between the settings. Greedy loops. Sampling at temperature 1 wanders (a model with one word of memory has nothing to keep it on topic). The sampled outputs are reproducible: the seed is fixed, so you will see the same words.

## 6. The Rust behind it

**`total_cmp` for sorting floats.** `f32` is only `PartialOrd`, because NaN is not less than, equal to or greater than anything. `sort_by(|a, b| a.partial_cmp(b).unwrap())` therefore panics the day a NaN logit appears (from an overflow somewhere upstream), taking the server down. `f32::total_cmp` defines a total order over every bit pattern, NaN included, so sorting never panics; a NaN simply sorts to one end.

**`select_nth_unstable_by`** is the standard library's introselect: expected linear time, rearranges in place, and tells you nothing about the order within each side. It is the right tool whenever you need "the k best" but not "the k best, sorted".

**Struct update syntax and `Default`.** `SamplingParams { top_p: 0.9, ..SamplingParams::default() }` fills the unnamed fields from another value. Implementing `Default` with the "off" values makes this the natural way to write a request.

**`std::ops::ControlFlow`** is the standard type for "continue or stop early". It says what it means, unlike a `bool` whose meaning (`true` = stop? continue?) the reader has to look up.

**The `entry` API.** `*self.output_counts.entry(token).or_insert(0) += 1` looks the key up once, inserts 0 if missing, and returns a mutable reference to the count.

**Strings are UTF-8, and slicing is by byte.** `&s[i..]` panics if `i` falls inside a multi-byte character. `char_indices` yields only valid boundaries; the `holding_back_respects_character_boundaries` test feeds an "é" (2 bytes) through the stop matcher to prove it.

**`include_str!`** embeds the text file into the binary at compile time, so the demo has no file paths to get wrong at run time.

## 7. Mistakes you will make

- **Dividing by a temperature of 0**, or of 1e-39. You get infinities, then NaN, then a token chosen by whatever your argmax does with NaN. Treat tiny temperatures as greedy.
- **Sorting logits with `partial_cmp().unwrap()`.** Fine until the first NaN.
- **Sorting the whole vocabulary for top-p on every token** without a top-k first. Measure it: 1.3 ms here.
- **One random generator for all requests.** Results then depend on the other traffic and cannot be reproduced.
- **Checking stop sequences token by token**, which misses sequences split across tokens, or **releasing text before checking**, which shows the user the beginning of the stop sequence.
- **Showing the stop token** (`<|im_end|>`) to the user.
- **Forgetting `max_tokens`.** A model that never emits its stop token will fill the cache and then crash or hang.
- **Comparing outputs with another engine without matching the order of operations**, the penalty definitions, and whether logprobs are raw or processed.

## 8. How the professionals do it

- **Hugging Face `generate`** builds a list of "logits processors" (`RepetitionPenaltyLogitsProcessor`, `TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper`, `MinPLogitsWarper` and many more) and applies them in a fixed order.
- **llama.cpp** has a configurable "sampler chain" (`llama_sampler_chain_add`) with many more samplers (typical, XTC, DRY, Mirostat...), and defaults of top-k 40, top-p 0.95, min-p 0.05, temperature 0.8.
- **vLLM's `SamplingParams`** has all of this chapter's fields, plus `stop`, `stop_token_ids`, `logprobs` and `n` (several samples per prompt, sharing the prompt's cache). Sampling runs on the GPU for the whole batch, and top-k/top-p can use FlashInfer's kernels, which sample without sorting the vocabulary.
- **The OpenAI API** defines `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `logprobs` and `top_logprobs`, the names most open-source servers copy (chapter 22).
- **Beam search**, which keeps several candidate sequences and returns the most likely one, is standard in machine translation but rarely used for chat models: it favours short, generic answers and multiplies the cost.

## 9. Exercises

1. **Order matters.** Using part 1's logits, which tokens does top-p 0.8 keep at temperature 1, and at temperature 2?
2. **Shift invariance.** Add 10 to every logit. Which of the chapter's settings still produce exactly the same distribution? Which one does not, and why?
3. **A faster greedy.** The greedy path spends about 62 µs on an argmax over 49,152 logits. Write a version that first finds the maximum value with `fold(f32::NEG_INFINITY, f32::max)` and then its position, and one that keeps 8 running maxima (`as_chunks::<8>`, like chapter 8's `softmax_fast`). Measure both.
4. **How much to hold back.** A simpler stop matcher always holds back the last `longest_stop_len − 1` bytes. Is it correct? What does the chapter's version gain?
5. **A shared generator.** Explain, with an example of two requests, why a server-wide random generator makes seeds useless.
6. **Top-p versus min-p.** For the distributions `[0.90, 0.06, 0.02, 0.02]` and `[0.30, 0.25, 0.25, 0.20]`, which tokens do top-p 0.95 and min-p 0.1 keep? What does the comparison show?

## 10. Check yourself

1. What does temperature do to the logits, and what do T = 0.5 and T = 2 do to the distribution?
2. Why does the sampler treat very small temperatures as greedy?
3. How do top-k, top-p and min-p decide what to cut?
4. Why is top-p much more expensive than top-k, and how do engines avoid the cost?
5. Why does each request need its own random generator?
6. What is wrong with the Hugging Face repetition penalty from a mathematical point of view?
7. Why must a stop-sequence matcher hold text back, and when may it release it?

## 11. Recap

- The model scores; the sampler chooses. The sampler's settings are part of every request.
- Pipeline: penalties → temperature → top-k → softmax → top-p → min-p → inverse-CDF draw, with a per-request seeded generator.
- Greedy is deterministic and prone to loops; temperature reshapes the distribution; top-k, top-p and min-p cut the tail, which matters because one bad draw derails everything after it.
- Top-p over a full vocabulary costs a sort (1.3 ms for 49,152 tokens here); a top-k first makes it cheap.
- Stop sequences span tokens: hold back text that could start one. Report why generation ended.
- Use `total_cmp` for floats, `select_nth_unstable_by` for "the k best", `ControlFlow` for early exit.

## Answers

**Exercises**

1. At T = 1 the probabilities are 0.679, 0.151, ...: the first two add up to 0.830 ≥ 0.8, so top-p keeps " Paris" and " Lyon". At T = 2 they are 0.425, 0.201, 0.157, 0.122, ...: the running sum is 0.425, 0.626, 0.783, then 0.905, so it keeps four tokens. Temperature applied first changes what top-p keeps.
2. Temperature, top-k, top-p, min-p, the frequency and presence penalties and greedy are all unchanged: each depends only on differences between logits (the penalties subtract the same amount from a token's logit whatever its value). The repetition penalty changes: it divides positive logits and multiplies negative ones, so after the shift a logit of −1 (multiplied, −2) becomes 9 (divided, 4.5). With a penalty of 2, the token's logit drops by 1 before the shift and by 4.5 after it, relative to unpenalized tokens.
3. Measured on the reference machine in a separate benchmark: the `enumerate().reduce(...)` version 62.5 µs, the two-pass version (vectorizable `fold` for the maximum, then `position`) 33.1 µs, the 8-lane version 25.0 µs. The original compares and tracks an index in one loop-carried chain; the faster versions let the compiler keep several comparisons in flight.
4. It is correct: any stop sequence that could still complete must start within the last `longest_stop_len − 1` bytes. But it delays all text by that many bytes, even text that cannot begin a stop sequence. With the stop `"\nUser:"` (6 bytes) it always withholds the last 5 bytes; the chapter's version withholds nothing unless the text actually ends in `"\n"`, `"\nU"` and so on. The stream reaches the user sooner and in steadier pieces.
5. With one generator, request A's k-th random number is whatever the generator produces after all the draws other requests made in between. Run request A (seed ignored) alone and it gets numbers 1, 2, 3...; run it next to request B and it may get 2, 4, 6... Its tokens differ even though nothing about A changed. With a generator per request, seeded from the request, A's numbers depend only on A.
6. First distribution: top-p 0.95 keeps the first two tokens (0.90, then 0.96 ≥ 0.95); min-p 0.1 keeps only the first (threshold 0.09). Second distribution: top-p 0.95 keeps all four (0.30, 0.55, 0.80, 1.00); min-p 0.1 also keeps all four (threshold 0.03). Min-p cuts more aggressively when the model is confident and stays permissive when it is not, and it expresses that with a threshold relative to the best token rather than a cumulative sum.

**Check yourself**

1. It divides them by T before the softmax. T = 0.5 doubles the differences between logits, concentrating probability on the most likely tokens; T = 2 halves them, spreading probability towards the tail.
2. Dividing by a tiny T can produce infinite values, and the softmax then computes infinity minus infinity, which is NaN. Below 1e-5, sampling is practically indistinguishable from greedy anyway.
3. Top-k keeps a fixed number of the most likely tokens; top-p keeps the smallest set of most likely tokens reaching a total probability p; min-p keeps tokens with at least `min_p` times the best token's probability.
4. Top-p needs the tokens sorted by probability, a full sort of the vocabulary; top-k needs only a linear-time partial selection. Engines apply a top-k first (so the sort is small) or use algorithms that find the top-p set without sorting.
5. So that its random numbers, and therefore its tokens, depend only on its own seed and not on other traffic; otherwise seeds cannot reproduce anything.
6. It divides positive logits and multiplies negative ones, so its effect depends on the sign of the logit, which is arbitrary: adding a constant to all logits does not change the model's distribution but does change the penalty's effect.
7. Because a stop sequence can arrive split across several tokens; releasing its beginning before knowing would show part of it to the user. Text can be released as soon as it cannot be the beginning of any stop sequence.

## Further reading

- Holtzman et al., "The Curious Case of Neural Text Degeneration", 2020: why greedy and beam search degenerate, and nucleus (top-p) sampling.
- Fan, Lewis and Dauphin, "Hierarchical Neural Story Generation", 2018: top-k sampling.
- Keskar et al., "CTRL: A Conditional Transformer Language Model for Controllable Generation", 2019: the repetition penalty.
- Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", 2024.
- Next: [Chapter 16: A real model](../16-real-model/README.md). Load SmolLM2's weights and tokenizer, and chat with it.
