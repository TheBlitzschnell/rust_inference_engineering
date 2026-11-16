# Chapter 26: Speculative decoding

> **In one sentence:** a decode step costs about as much for a few tokens as for one, so let something cheap guess the next few tokens and have the large model check all of them in one pass; keeping the guesses it agrees with gives the large model's exact output in fewer passes, and how much faster that is depends on how often the guesses are right, how cheap they are, and how cheap checking several tokens really is.

**Where this fits:** chapters 23 to 25 made a server fast for many users at once. This chapter makes a single request faster, by attacking the sequential nature of decoding itself. Chapter 27 then constrains what the model may produce.

**You need:** chapter 14 (the KV cache, `forward_all`), chapter 15 (sampling, the seeded generator), chapters 18 and 19 (int8 and 4-bit weights), chapter 23 (packed weights; batching is almost free when memory-bound).

**You will build:** the two acceptance rules (greedy agreement, and rejection sampling, which keeps the target's distribution exactly, proved and tested); two drafters (a small model with its own KV cache, and n-gram lookup in the text so far); the draft-verify-rollback loop; and measurements with SmolLM2-360M as the target: what checking several tokens costs, the ceiling with a perfect drafter, SmolLM2-135M drafting in three precisions, n-gram lookup on editing tasks and on chat, and sampling.

---

## 1. The intuition

A senior editor and a junior writer. The writer drafts the next sentence quickly; the editor reads it and either approves it word by word up to the first word they would have written differently, or crosses out from there and writes that one word themselves. Reading a sentence takes the editor barely longer than writing a single word, so whenever the junior guesses well, the page fills several words per round instead of one. And the page ends up exactly as the editor would have written it alone: nothing the editor disagrees with survives.

**Where the analogy breaks:** an editor can tell at a glance whether a word is "the one they would have chosen". When the large model samples at a temperature, there is no single right word, only a distribution; the check then becomes a coin toss with carefully chosen odds (section 3.4), and the output is guaranteed to have the large model's distribution rather than to be identical to any particular answer.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Target model** | The model whose output we want (here SmolLM2-360M). |
| **Draft model, drafter** | Anything that cheaply proposes the next tokens: a smaller model, extra heads on the target, or a lookup in the text. |
| **Draft length `k`** | How many tokens the drafter proposes per round. |
| **Verification** | One forward pass of the target over the last token and the `k` draft tokens, giving `k + 1` rows of logits. |
| **Acceptance rate** | The share of draft tokens the target keeps. |
| **Rejection sampling** | Accepting a sample from one distribution with a probability chosen so that the accepted results follow another. |
| **Total variation distance** | Half the sum of absolute differences between two distributions; 0 when equal, 1 when they share nothing. |
| **Prompt lookup (n-gram) decoding** | Drafting by finding the last few tokens earlier in the text and proposing what followed them there. |
| **Rollback** | Discarding the KV cache entries of rejected draft tokens. |

## 3. The concepts in depth

### 3.1 Why checking several tokens is cheap

Decoding one token reads every weight of the model to do two FLOPs per weight (chapter 23): it is memory-bound. A forward pass over `n` tokens reads the same weights once and uses each `n` times. Until the arithmetic catches up with the memory (the ridge point, about 13 rows on this machine), the extra tokens are nearly free. What is not free: everything per token that is not a shared matrix product: attention over the cache, norms, RoPE, and the LM head's output for each row.

Part 1 measures SmolLM2-360M (packed `bf16` weights from chapter 23, 4 threads, the machine of chapter 17) processing `n` new tokens after a 35-token prompt:

```text
== 1. one forward pass of SmolLM2-360M over n new tokens (after a 35-token prompt)
   n                          1      2      3      4      5      6      8     16
   time (ms)               37.8   42.2   43.8   43.2   45.2   48.4   53.6   91.3
   relative to n = 1      1.00x  1.12x  1.16x  1.14x  1.20x  1.28x  1.42x  2.42x
   one token of the draft models:
      135M, packed bf16   11.5 ms, 0.31 of a target token
      135M, int8           7.9 ms, 0.21 of a target token
      135M, 4-bit          5.2 ms, 0.14 of a target token
```

Four tokens cost 1.1 to 1.4 times one token (across four runs; this machine is noisy), eight tokens 1.4 to 1.9 times, sixteen 2.1 to 2.8 times. On a GPU at batch size 1, the ridge point is near 300 rows and the first column's cost holds almost unchanged up to 16 tokens or more; here it does not, and that will limit everything below.

The draft models cost a fraction of a target token: SmolLM2-135M about a third in `bf16` (0.31-0.39 across runs), a fifth in int8 (0.21-0.25), an eighth in 4 bits (0.11-0.14), with chapters 18 and 19's weights.

### 3.2 Draft, verify, keep

One round:

```text
context:        ... the cat sat
draft (k = 4):                  on  the  red  mat          (the drafter's guesses)
target input:          sat  on  the  red  mat              (the last token, then the draft)
target's choices:      on   the  big  ...  ...             (row i: its prediction after input i)
kept:                       on  the                        (agrees up to the first disagreement)
added:                                big                  (the target's own choice there)
```

The target's row `i` is its prediction for the position of draft token `i`, given everything before it, including draft tokens `0..i`. If draft token `i` equals the target's choice, everything up to it is exactly what the target would have produced, so the next row is valid too. At the first disagreement the target's own choice is taken, and the rest of the draft (and the target's rows after it, which were computed on a wrong token) are thrown away. If all `k` draft tokens agree, row `k` gives one more token for free.

Every round therefore produces between 1 and `k + 1` tokens, never fewer than plain decoding, and the output is exactly the target's greedy output. Not approximately: the same tokens. In every measurement below, all answers were identical to plain decoding, token for token. (That relies on the target computing the same logits in a `k + 1`-token pass as in a one-token pass. Chapter 23's packed weights are batch-invariant; a kernel that is not could, very rarely, break a near-tie differently. The tests therefore check every output token against the target's maximum logit within rounding rather than comparing whole answers.)

The KV cache must follow. The target's verification wrote `k + 1` positions into its cache; those after the last kept token are wrong and are dropped by moving the cache's length back (`truncate`, chapter 14). The draft model has its own cache, which contains its guesses: it keeps the positions that still match the accepted text and recomputes from there.

### 3.3 How fast can it be?

Suppose each draft token is accepted with probability α, independently. A round produces `1 + α + α² + ... + α^k` tokens on average:

```text
E[tokens per round] = (1 − α^(k+1)) / (1 − α)
```

A round costs `k` draft tokens and one verification of `k + 1` tokens. With `c` the cost of a draft token and `v(k)` the cost of verifying, both relative to one plain decode step of the target, the speedup is

```text
speedup = E[tokens per round] / (k·c + v(k))
```

Three things limit it: how often the drafter is right (α), how cheap it is (`c`), and how cheap verification is (`v(k)` close to 1). Growing `k` adds tokens per round only while α^k is not small, but adds `c` and some `v` every time, so there is a best `k`, usually small.

Part 2 removes the first two limits. The drafter is an oracle: it replays the target's own answer, so it is always right and costs nothing:

```text
== 2. the ceiling: a drafter that knows the answer (three chat answers)
   plain decoding: 436 tokens, 35.2 ms per token
   drafter             k   accepted   tokens/pass   per token   speedup   same answers
   oracle              1       100%          2.00     19.4 ms     1.82x   3 of 3
   oracle              2       100%          2.98     15.5 ms     2.28x   3 of 3
   oracle              4       100%          4.92     10.5 ms     3.36x   3 of 3
   oracle              8       100%          8.76      7.8 ms     4.52x   3 of 3
   oracle             16       100%         16.22      6.4 ms     5.54x   3 of 3
```

Tokens per pass is `k + 1` (a little less, because answers end), and the speedup is pure `v(k)`: 3.0-3.4 times at `k = 4`, 4.4-5.5 times at 16 (three runs). That is the ceiling on this machine. On a GPU, where `v(k)` stays near 1, an oracle with `k = 4` would come close to 5 times.

### 3.4 Real drafters: a smaller model

The obvious drafter is a smaller model of the same family: it must use the same tokenizer, so that its tokens are the target's tokens. SmolLM2-135M drafts for SmolLM2-360M (part 3, three chat answers, greedy):

```text
== 3. SmolLM2-135M drafts for SmolLM2-360M, greedy (three chat answers)
   plain decoding: 436 tokens, 35.2 ms per token
   drafter             k   accepted   tokens/pass   per token   speedup   same answers
   135M bf16           1        70%          1.70     30.3 ms     1.16x   3 of 3
   135M bf16           2        60%          2.19     29.8 ms     1.18x   3 of 3
   135M bf16           3        50%          2.49     33.6 ms     1.05x   3 of 3
   135M bf16           4        44%          2.76     36.7 ms     0.96x   3 of 3
   135M int8           2        60%          2.21     27.6 ms     1.28x   3 of 3
   135M int8           3        50%          2.50     28.3 ms     1.25x   3 of 3
   135M 4-bit          2        49%          1.97     27.1 ms     1.30x   3 of 3
   135M 4-bit          3        40%          2.19     29.9 ms     1.18x   3 of 3
```

The drafts are right often (70% of single-token drafts), and still the gain is small: with `bf16` drafts, between 0.98 and 1.18 times at best across three runs, which on this noisy machine is hard to tell from no gain at all. The formula says why. At `k = 2`, a round makes 2.19 tokens, but costs 2 × 0.31 (drafting) + 1.16 (verifying) = 1.78 target steps, which predicts 1.23 times: the draft model is only about 3 times cheaper than the target. Quantizing the drafter (int8, 4 bits) cuts `c` to a fifth or an eighth; it also makes its guesses a little worse (4 bits: 49% accepted at `k = 2` instead of 60%), but cheaper drafting wins: 1.1 to 1.3 times. Larger `k` stops paying from `k = 3` or 4: the extra guesses are mostly wrong and each costs drafting and verification.

The lesson generalizes: speculative decoding needs a drafter one or two orders of magnitude cheaper than the target. Production pairs look like a 1B drafter for a 70B target (`c` ≈ 0.02), on GPUs where `v(4)` ≈ 1.05; with α = 0.8, that is 3.4 tokens per round at a cost of about 1.1, roughly 3 times faster. A 135M drafter for a 360M target, on a CPU, cannot do that.

### 3.5 A drafter for free: n-gram lookup

Many answers copy from their prompt: a corrected text, an edited document, code with a small change, an answer quoting retrieved passages. *Prompt lookup decoding* exploits this with no model at all: find the most recent earlier occurrence of the last few tokens, and propose whatever followed them there. It costs microseconds, so `c` ≈ 0, and it is right exactly when the answer is copying.

Part 4 runs it on three editing tasks (fix spelling, replace names, comment a function) and on the chat answers. `n-gram, 2 to 4` tries to match the last 4 tokens, then 3, then 2, and proposes nothing when not even 2 match:

```text
== 4. n-gram lookup, greedy: three editing tasks
   plain decoding: 99 tokens, 41.9 ms per token
   drafter             k   accepted   tokens/pass   per token   speedup   same answers
   n-gram, 1 to 4      4        49%          2.60     28.3 ms     1.48x   3 of 3
   n-gram, 2 to 4      4        68%          2.36     26.6 ms     1.58x   3 of 3
   n-gram, 2 to 4      8        57%          2.86     28.3 ms     1.48x   3 of 3

== 4. n-gram lookup, greedy: three chat answers
   plain decoding: 436 tokens, 35.2 ms per token
   drafter             k   accepted   tokens/pass   per token   speedup   same answers
   n-gram, 1 to 4      4         9%          1.17     33.9 ms     1.04x   3 of 3
   n-gram, 2 to 4      4        20%          1.10     33.4 ms     1.05x   3 of 3
   n-gram, 2 to 4      8        11%          1.11     35.7 ms     0.99x   3 of 3

   the second editing answer: "Alex went to the market with John. Alex bought apples, and John bought bread. Later, Alex and John cooked dinner together for their friends."
```

On the editing tasks: 1.6 to 1.8 times faster across three runs, with nothing to load or run. Requiring a match of at least 2 tokens makes the proposals right more often (68% against 49% at `k = 4`): single-token matches ("the", a comma) are everywhere, and what followed them once is often wrong now. But they also propose more often, and the two settings came out about equally fast. On chat answers, which copy little, almost nothing is proposed and the speed is unchanged; the drafter costs nothing, so it also loses almost nothing. Production servers often enable it by default for exactly this reason.

### 3.6 Sampling: rejection sampling

With sampling, "the target's choice" is a random draw. Comparing the draft token with one draw of the target would accept too rarely and, worse, would change the output's distribution. The rule that keeps the target's distribution exactly (Leviathan, Kalman and Matias; Chen et al.; both 2023):

- Draft token `x` was drawn from the drafter's distribution `q`. The target's distribution at that position is `p`.
- Accept `x` with probability `min(1, p(x) / q(x))`.
- If rejected, draw the replacement from the leftover distribution `max(0, p − q)`, normalized; stop the round there.
- If all `k` are accepted, draw one more token from the target's last row.

Why the output follows `p`: a token `x` comes out either by being drafted and accepted, with probability `q(x) · min(1, p(x)/q(x)) = min(p(x), q(x))`, or by a rejection followed by drawing `x` from the leftover. A rejection happens with probability `1 − Σ min(p, q)`, which equals `Σ max(0, p − q)`, the leftover's total; so the second path has probability `max(0, p(x) − q(x))`. Together: `min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x)`.

The same computation gives the acceptance probability: `Σ min(p, q) = 1 − TV(p, q)`, one minus the total variation distance between the two distributions. For a deterministic drafter (n-gram lookup), `q` puts all its probability on its proposal, and the rule becomes: accept with probability `p(x)`; on rejection, draw from `p` with `x` removed.

Part 5 compares greedy and sampled speculation with the same drafter and `k = 2`:

```text
== 5. 135M bf16 drafts, k = 2, greedy and sampled (three chat answers)
   decoding           accepted   tokens/pass   plain   speculative   speedup
   greedy                  60%          2.19   35.2 ms       32.5 ms     1.08x
   temperature 0.5         72%          2.43   36.4 ms       30.2 ms     1.20x
   temperature 1.0         68%          2.36   35.4 ms       30.8 ms     1.15x
```

More draft tokens are accepted when sampling than when decoding greedily, and the speedups were a little larger too (across two runs: 1.20-1.26 times at temperature 0.5, 1.09-1.15 at 1.0, 0.98-1.08 greedy). Greedy acceptance needs the two models' *most likely* tokens to coincide; sampled acceptance only needs their distributions to overlap, and two related models' distributions overlap a lot even when their top choices differ. The speedups stay small for the same reason as in part 3: the `bf16` drafter is too expensive.

### 3.7 Speculation and batching

Speculative decoding converts spare arithmetic into fewer sequential steps. At batch size 1, decode is memory-bound and there is plenty of spare arithmetic. In a server running many requests per step (chapter 23), the step is already closer to compute-bound; verification tokens then cost real compute, and every rejected draft token is compute that produced nothing. So the benefit shrinks as load grows, and can turn into a loss. Serving engines therefore use speculation mostly at low and moderate load, limit `k`, or adjust it to the measured acceptance rate.

## 4. The code

The acceptance rules are in [`src/verify.rs`](src/verify.rs), the drafters in [`src/draft.rs`](src/draft.rs), the loop in [`src/speculative.rs`](src/speculative.rs), the demo (including the oracle drafter) in [`src/main.rs`](src/main.rs).

### 4.1 Greedy verification

<!-- file: src/verify.rs -->
```rust
pub fn verify_greedy(draft: &[u32], logits: &[f32], vocab: usize) -> (usize, u32) {
    let rows: Vec<&[f32]> = logits.chunks_exact(vocab).collect();
    // ...
    for (i, &d) in draft.iter().enumerate() {
        let best = argmax(rows[i]);
        if best != d {
            return (i, best);
        }
    }
```

It returns how many draft tokens to keep and the token that follows them: the target's choice at the first disagreement, or, after the loop, its choice from the last row. `argmax` takes the first maximum on ties, as chapter 15's sampler does, so a tie is broken the same way with or without speculation.

### 4.2 Rejection sampling

<!-- file: src/verify.rs -->
```rust
        let (pd, qd) = (p[i][d as usize], q[i].q(d, d));
        // Accept with probability min(1, p/q).
        if (rng.next_f64() as f32) * qd < pd {
            continue;
        }
        // Rejected: draw from what p has and q lacks.
        let mut residual: Vec<f32> = p[i]
            .iter()
            .enumerate()
            .map(|(t, &pt)| (pt - q[i].q(d, t as u32)).max(0.0))
            .collect();
```

`u < p/q` is written `u · q < p`, which needs no division and is right when `p ≥ q` (always accepted, since `u < 1`). `Proposal::q(d, t)` is the drafter's probability of token `t` for a proposal of `d`: a value from its distribution for a sampled draft, or 1 for `t = d` and 0 otherwise for a deterministic one, so one formula serves both drafters. If the leftover sums to zero (the two distributions agree to rounding, and the rejection was a rounding accident), the replacement is drawn from `p` itself.

The tests check the rule on small distributions with 200,000 trials each (drafts from a uniform `q`, from a reversed one, from `q = p`, from a point mass), and end to end with tiny random models: speculative sampling's first token, over 10,000 seeds, lands as close to the target's distribution as 10,000 direct draws do, while the draft model's own distribution is far outside that margin.

### 4.3 The loop, and the rollback

<!-- file: src/speculative.rs -->
```rust
            let mut input = vec![*tokens.last().expect("a prompt")];
            input.extend_from_slice(&draft.tokens);
            let logits = self
                .model
                .forward_all(pool, &input, &mut self.cache, &mut self.scratch);
            // ...
            // The cache now holds the input's positions; forget the ones
            // after the last kept draft token.
            let rejected = draft.tokens.len() - kept;
            self.cache.truncate(self.cache.len() - rejected);
```

The target's cache always holds every accepted token but the last; the last is the first input of the next verification, whose output row predicts the first draft token. `forward_all` (chapter 14, written for this chapter) returns logits for every input position instead of only the last. Rolling back is moving the cache's length: the rejected positions' keys and values are simply overwritten later. With chapter 24's paged cache, it would be the same move of `table.len`, plus releasing blocks past the new end.

### 4.4 Keeping the draft model in step

<!-- file: src/draft.rs -->
```rust
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
```

The drafter does not need to be told what was accepted: it compares the tokens in its cache with the current context, keeps the common prefix (its accepted guesses), and feeds the rest (the target's correction, plus the last draft token if all were accepted). At least one token is always fed, because its logits give the first guess. Then it generates `k` tokens with its own sampler state.

### 4.5 N-gram lookup

<!-- file: src/draft.rs -->
```rust
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
```

Longest match first, most recent occurrence first. This scans the whole context for each `n`, which is fine for a few thousand tokens (microseconds); production implementations keep a hash table or suffix structure updated as tokens arrive.

## 5. Run it

```bash
./tools/download_model.sh 360M                                   # the target, once
cargo test -p ch26-speculative-decoding
cargo run --release -p ch26-speculative-decoding                 # all five parts, about 7 minutes
cargo run --release -p ch26-speculative-decoding -- ngram        # or: cost, ceiling, draft, sampling
```

The tests check greedy verification on hand-made logits; that rejection sampling reproduces the target distribution for four kinds of draft distribution and for deterministic proposals; that a draft drawn from the target's own distribution is always accepted; that greedy speculation with a perfect drafter, a random other model and n-gram lookup, for `k` = 1, 3 and 5, produces the target's greedy tokens (each checked against the target's maximum logit), with every draft accepted for the perfect drafter; that sampled speculation's first token follows the target's distribution (section 4.2); that a perfect drafter is always accepted when sampling (which checks that the rows of `p` line up with the draft tokens); that generation stops before a stop token even inside an accepted draft; and the n-gram lookup.

## 6. The Rust behind it

**Trait objects for interchangeable drafters.** `generate` takes `&mut dyn Drafter`, and the tests keep a `Vec<(&str, Box<dyn Drafter>)>` of three different drafters. A generic parameter would compile a separate loop per drafter; dynamic dispatch costs one indirect call per round, nothing next to a forward pass, and lets the demo choose drafters at run time.

**One enum for two kinds of proposal.** `Proposal::Sampled(Vec<f32>)` carries a distribution, `Proposal::Exact` carries nothing; `q(proposed, token)` answers for both. The verification code never asks which drafter it is talking to.

**Iterators for prefix logic.** `zip(...).take_while(|(a, b)| a == b).count()` is the length of the common prefix of two slices; `chunks_exact(vocab)` views the flat logits as rows without copying; `iter().chain([&next])` appends one item to an iterator (arrays are iterable by value since the 2021 edition).

**`bool` to `f32` without `as`.** `f32::from(u8::from(token == proposed))` converts a comparison to 0.0 or 1.0 through the lossless `From` conversions; Clippy's pedantic lints prefer that to `as` casts, whose meaning depends on the types.

**`let` chains.** The demo's `if chat && let Some(run) = &self.plain_chat` combines a boolean and a pattern match in one condition, which the 2024 edition allows.

## 7. Mistakes you will make

- **Forgetting the rollback.** The target's cache keeps the rejected positions' keys and values, and the next round attends to tokens that are not in the text. The output stays fluent and becomes subtly wrong.
- **Off by one in the rows.** Row `i` of the verification is the prediction *for* draft token `i`, made after the input token before it. Feeding only the draft tokens, without the last accepted one, shifts everything by one.
- **Accepting a sampled draft by comparing it with one sample of the target.** It changes the output distribution. Use the ratio test.
- **Using a `p` or `q` other than the distributions actually sampled from.** With temperature, top-k, top-p or penalties, both must be the processed distributions; the proof needs the real ones.
- **Draft and target with different tokenizers.** The tokens mean different things; nothing is ever accepted (or worse, something is, by coincidence).
- **Judging a drafter by its acceptance rate.** The 4-bit drafter is accepted less often than the `bf16` one and is faster overall. Measure time per token.
- **A large `k` by default.** Each extra guess costs drafting and verification and is accepted with probability α^i.
- **Speculating at high load.** When the batch is compute-bound, rejected tokens are wasted compute.

## 8. How the professionals do it

- **vLLM, SGLang, TensorRT-LLM, llama.cpp** all support draft-model speculation and n-gram (prompt lookup) drafting; llama.cpp's `llama-speculative` and its server's `--model-draft` pair a small GGUF model with a large one.
- **Medusa** (Cai et al., 2024) adds several extra heads to the target that predict tokens 2, 3, 4 ahead from the same hidden state, and verifies a *tree* of candidates in one pass with a tree-shaped attention mask.
- **EAGLE** (Li et al., 2024; EAGLE-2 and EAGLE-3) trains a one-layer draft network on the target's hidden features rather than on tokens; it reaches acceptance rates far above a separate small model's, at a tiny cost. vLLM, SGLang and TensorRT-LLM support it.
- **Multi-token prediction.** DeepSeek-V3 was trained with an extra module predicting the next-but-one token; at inference that module drafts, with acceptance rates reported around 85-90% for the second token.
- **Lookahead decoding** (Fu et al., 2024) drafts with Jacobi iteration and n-grams gathered from the model's own earlier guesses, with no draft model at all.
- **Batch-aware speculation**: engines reduce or disable speculation when the batch is large, and some adapt `k` per request from its recent acceptance rate.
- Inference APIs expose prompt lookup as "predicted outputs": the client sends the text it expects (the file being edited), and the server uses it as the draft.

## 9. Exercises

1. **Adaptive `k`.** Let the draft model stop drafting when its most likely token's probability falls below a threshold (0.5, say), up to `k = 6`. Compare time per token with the best fixed `k` in part 3.
2. **Top-p with rejection sampling.** Make `probabilities` apply top-p (chapter 15) for both models. Which distributions must the ratio test use, and does the test in section 4.2 still pass?
3. **Self-speculation with quantization.** Use the 4-bit SmolLM2-135M to draft for the `bf16` SmolLM2-135M. Predict the speedup from part 1's costs and the acceptance you measure, then measure it.
4. **Two candidates.** Let the drafter propose its two most likely first tokens, each followed by its greedy continuation, and verify both branches in one pass. What must change in the attention mask and in the KV cache, compared with a single draft?
5. **Speculation in chapter 24's engine.** A decoding request contributes its last token plus `k` draft tokens to the step, like a small prompt chunk. Write down what changes in the plan, the logits, the sampling and the block table, and predict, from chapter 23's step-time table, at what batch size it stops paying on this machine.
6. Compute the acceptance probability of a draft from `q = (0.5, 0.3, 0.2)` for a target `p = (0.2, 0.3, 0.5)`, and the distribution of the replacement token after a rejection.

## 10. Check yourself

1. Why does processing `k + 1` tokens in one pass cost much less than `k + 1` passes?
2. Why does greedy speculative decoding produce exactly the target's greedy output?
3. What does a round produce if every draft token is wrong? If every one is right?
4. Write the expected tokens per round and the speedup in terms of α, `k`, `c` and `v(k)`.
5. With a free, always-right drafter, what limits the speedup on this machine, and why is that limit lower than on a GPU?
6. Why did quantized drafters, whose guesses were accepted less often, beat the `bf16` drafter?
7. State the rejection-sampling rule, and show that the probability of accepting a draft is `1 − TV(p, q)`.
8. When does n-gram lookup help, and why does it cost almost nothing when it does not?
9. Why does speculative decoding help less in a heavily loaded batching server?

## 11. Recap

- Verification is cheap because decoding is memory-bound: on this CPU, 4 tokens in one pass cost 1.1-1.4 times one token, 8 tokens 1.4-1.9 times (on a GPU, close to 1).
- Each round drafts `k` tokens, verifies them in one target pass, keeps the agreeing prefix plus one target token, and rolls the KV caches back. Greedy output is exactly the target's.
- Speedup = `(1 − α^(k+1)) / (1 − α)` divided by `k·c + v(k)`: it needs a drafter that is often right *and* much cheaper than the target, and cheap verification.
- A perfect free drafter reached 3.0-3.4 times at `k = 4` and 4.4-5.5 times at `k = 16` here: the ceiling set by verification cost.
- SmolLM2-135M drafting for SmolLM2-360M gave 0.98-1.18 times at best in `bf16` (it costs a third of the target) and 1.1-1.3 times in int8 or 4 bits, with fewer accepted tokens but cheaper drafts.
- N-gram lookup, free, made editing tasks 1.6-1.8 times faster and left chat unchanged.
- Rejection sampling (accept with `min(1, p/q)`, else draw from `max(0, p − q)`) keeps the target's distribution exactly; it accepts with probability `1 − TV(p, q)`, which was higher than greedy agreement.

## Answers

**Exercises**

1. With a confidence threshold, the drafter spends drafting time where it tends to be right (common phrases, the ends of words) and stops early on uncertain tokens, where guesses are mostly rejected. Expect slightly more tokens per round than `k = 2` at a lower average cost than `k = 4`, so a small gain over the best fixed `k`; the threshold itself needs tuning, and on this machine the gain is bounded by part 2's ceiling.
2. The ratio test must use the distributions the tokens are actually drawn from: `q` after top-p for the drafter, `p` after top-p for the target (and the replacement after a rejection is drawn from `max(0, p − q)` with those filtered `p` and `q`). Then the proof is unchanged and the test passes. Using the unfiltered `p` would make the output follow the unfiltered distribution, not the one the user asked for.
3. With the 4-bit 135M costing about 0.35 of the `bf16` 135M (part 1: 4 ms against 11), and acceptance probably near or above that of 135M drafting for 360M (a quantized copy of the same model agrees with it more than a different model does), expect about `E / (k·0.35 + v(k))` with `v(k)` for the 135M (which, being smaller, gains less from extra rows: closer to its own ridge point). Something like 1.3-1.5 times at `k = 2` or 3 is plausible; measure.
4. The input holds the last token, branch A's tokens and branch B's tokens; each branch's tokens must attend to the context and to their own branch only, not to the other branch, so the causal mask becomes a tree mask. Positions (for RoPE) follow each branch, so both branches have tokens at the same positions. After verification, the kept branch's KV entries must end up contiguous in the cache: copy them (or, with a paged cache, arrange the block table), and drop the other branch.
5. The plan gives a decoding request `k + 1` rows instead of 1; `forward_paged` must return logits for every row of such a request (as `forward_all` does), not only its last; sampling becomes verification; the block table must have room for `k + 1` positions, and after verification `table.len` moves back by the number rejected (releasing any block left empty). Speculation pays while a step with `B·(k + 1)` rows costs much less than `k + 1` steps with `B` rows: chapter 23's table shows step time growing roughly linearly beyond 8-16 rows on this machine, so with `k = 2` the gain should vanish around a batch of 4 to 8.
6. Acceptance = Σ min(p, q) = 0.2 + 0.3 + 0.2 = 0.7. The leftover `max(0, p − q)` = (0, 0, 0.3), normalized (0, 0, 1): after a rejection the replacement is always the third token.

**Check yourself**

1. Because the pass is memory-bound: its cost is dominated by reading the weights, which happens once whatever the number of tokens; each extra token adds only arithmetic (cheap until the ridge point) and its own per-token work such as attention.
2. A draft token is kept only if it equals the target's argmax at that position given everything before it, which is exactly the token the target would have produced; the first disagreement is replaced by the target's own choice and everything after it is discarded. By induction, the kept text is the target's greedy text.
3. All wrong: one token, the target's choice at the first position, the same as plain decoding (but the round cost drafting and a longer verification). All right: `k + 1` tokens, the `k` drafts and one more from the last row.
4. `E = (1 − α^(k+1)) / (1 − α)`; speedup = `E / (k·c + v(k))`, with `c` and `v(k)` relative to one plain decode step.
5. The cost of verifying `k + 1` tokens, `v(k)`: here it grows from 1 to about 1.4 at 4 tokens and over 2 at 16, because the CPU's ridge point is about 13 rows and per-token work (attention, norms, the LM head's rows) is not shared. A GPU's ridge point is around 300 rows, so `v(k)` stays near 1 for the `k` values used.
6. Because cost matters as much as accuracy: the 4-bit drafter costs 0.11-0.14 of a target step against 0.31-0.39 for `bf16`; its lower acceptance (49% against 60% at `k = 2`) loses fewer tokens per round than its cheaper drafting saves.
7. Accept draft `x ~ q` with probability `min(1, p(x)/q(x))`; on rejection draw from `max(0, p − q)` normalized; if all `k` are accepted, draw one more from the target's next row. The acceptance probability is `Σₓ q(x)·min(1, p(x)/q(x)) = Σₓ min(q(x), p(x))`, and since `Σ min(p, q) = 1 − ½ Σ |p − q|`, that is `1 − TV(p, q)`.
8. When the output copies stretches of earlier text (edits, corrections, code changes, quotes from retrieved documents). When it does not, the lookup usually finds no match of the minimum length and proposes nothing, so the round is a plain decode step plus a few microseconds of searching.
9. In a large batch the step is closer to compute-bound, so the extra verification tokens are no longer nearly free, and every rejected token is arithmetic that produced nothing; the spare capacity speculation relies on has already been used by batching.

## Further reading

- Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
- Chen, Borgeaud, Irving, Lespiau, Sifre and Jumper, "Accelerating Large Language Model Decoding with Speculative Sampling", 2023.
- Saxena, "Prompt Lookup Decoding", 2023 (the n-gram drafter).
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024.
- Li, Wei, Zhang and Zhang, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024.
- Fu, Bailis, Stoica and Zhang, "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", ICML 2024.
- DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024 (multi-token prediction used for speculative decoding).
- Next: [Chapter 27: Structured output](../27-structured-output/README.md). Constraining every step so that the answer is always valid JSON.
