# Chapter 12: Attention

> **In one sentence:** attention lets every token build its new representation as a weighted average of earlier tokens' information, with weights computed from how well its query matches their keys, and its memory and compute costs, which grow with the length of the text, are the central problem of LLM serving.

**Where this fits:** chapter 11 turned text into token IDs. An embedding lookup turns each ID into a vector. Attention is the one operation in a transformer where those vectors interact; everything else (norms, matmuls, activations) treats each token on its own. Chapter 13 wraps attention into a full transformer layer, chapter 14 caches its keys and values, and chapter 20 rewrites it so it never stores its biggest intermediate result.

**You need:** chapters 5 (matmul, dot products), 8 (softmax) and 11 (tokens).

**You will build:** causal scaled dot-product attention with multi-head and grouped-query support; rotary position embeddings in both layouts real checkpoints use; property tests for masking, grouped heads and RoPE's relative-position behaviour; and measurements of attention's quadratic cost and of KV cache sizes.

---

## 1. The intuition

Picture a meeting in which people speak one after another. Each new speaker, before talking, looks back over everyone who spoke earlier. They have a **question** in mind (their query). Every earlier speaker wears a **badge** describing what they talked about (their key). The new speaker pays most attention to the people whose badges match their question, and builds their own remarks mostly from what those people **said** (their values).

That is attention, for one token:

1. Compare my query with every earlier token's key (a dot product: how well do they match?).
2. Turn the match scores into weights that sum to 1 (softmax).
3. Take the weighted average of those tokens' values.

"Multi-head" means doing this several times in parallel with different questions: one head might look for the subject of the sentence, another for the previous word, another for an opening bracket that needs closing.

**Where the analogy breaks:** a person picks a few speakers to listen to. Attention mixes *everyone*, just with different weights, and nobody chooses what the questions and badges are: they come from learned weight matrices (`W_q`, `W_k`, `W_v`) applied to each token's vector. And in a causal language model, a speaker may only look back, never ahead: the text after them does not exist yet.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Hidden state** | The vector representing one token inside the model (576 numbers for SmolLM2). |
| **Query, key, value (Q, K, V)** | Three vectors computed from each token's hidden state by learned linear layers. |
| **Score** | The dot product of a query and a key, scaled by 1/√d. |
| **Attention weights** | The softmax of the scores: how much each earlier token contributes. |
| **Causal mask** | The rule that a token may only attend to itself and earlier tokens. |
| **Head** | One independent attention computation on a slice of the vector (`head_dim` numbers). |
| **MHA / GQA / MQA** | Multi-head / grouped-query / multi-query attention: every query head has its own K/V head, groups share one, or all share one. |
| **RoPE** | Rotary position embedding: positions encoded by rotating pairs of Q and K dimensions. |
| **Context length** | How many tokens a query can attend to. |
| **Prefill / decode** | Processing the whole prompt at once / generating one token at a time (chapter 1). |
| **KV cache** | Stored keys and values of earlier tokens, so they are not recomputed (chapter 14). |

## 3. The concepts in depth

### 3.1 From token IDs to vectors

The first layer of a language model is an **embedding table**: one learned row of `hidden_size` numbers per vocabulary entry. For SmolLM2 that is 49,152 rows of 576. Token ID 42 becomes row 42 (chapter 8's embedding lookup, a borrowed slice). From here on each token is a vector, called its **hidden state**, and the model's layers repeatedly refine those vectors.

### 3.2 The computation

For a sequence of `n` tokens with hidden states `X` (`n × hidden`):

```text
Q = X · W_qᵀ          (n × n_heads·d)       one query per token and head
K = X · W_kᵀ          (n × n_kv_heads·d)    one key per token and KV head
V = X · W_vᵀ          (n × n_kv_heads·d)    one value per token and KV head

for each head h, query token t:
    scores[j]  = (q[t,h] · k[j,h']) / √d          for every visible j
    weights    = softmax(scores)
    out[t,h]   = Σ_j weights[j] · v[j,h']

output = concat over heads(out) · W_oᵀ           (n × hidden)
```

where `d` is the head dimension and `h'` is the key/value head that query head `h` uses. The four projections are ordinary linear layers (chapters 5-7). The part in the middle is what this chapter implements.

**Why divide by √d?** If the entries of `q` and `k` are roughly independent with variance 1, their dot product has variance `d`, so scores grow with the head size. Large scores push softmax towards putting all weight on one token (its outputs saturate at 0 or 1), which makes learning and behaviour brittle. Dividing by √d keeps the scores' scale independent of `d`.

### 3.3 The causal mask, and why it enables caching

A language model is trained to predict each next token from the ones before it, so during training every position is only allowed to see earlier positions. The same rule must hold at inference: query `t` sees keys `0..=t`, never later ones. In the demo's 6-token example, row `t` has non-zero weights only in columns `0..=t`:

```text
   token 0: 1.00   .    .    .    .    .
   token 1: 0.65 0.35   .    .    .    .
   token 2: 0.20 0.28 0.52   .    .    .
   token 3: 0.10 0.31 0.18 0.41   .    .
   token 4: 0.35 0.23 0.19 0.12 0.12   .
   token 5: 0.33 0.15 0.19 0.14 0.09 0.10
```

Token 0 can only see itself, so its weight on itself is 1. Every row sums to 1.

The mask has a consequence that the whole of chapter 14 rests on: **appending a token never changes the outputs for earlier tokens.** Earlier queries cannot see the new token's key, so nothing about their computation changes. The test `the_future_cannot_change_the_past` checks exactly this, and `query_offset_matches_the_full_computation` checks the practical form: computing attention for only the last queries, against all the keys, gives the same result as computing everything and keeping the last rows. So when generating token 1,001, the model only needs attention for the *new* query, against keys and values it can keep from before.

### 3.4 Heads, and sharing keys and values

With `n_heads` heads, each head works on its own `d`-sized slice (SmolLM2: 9 heads × 64 = 576). Each head can learn to look for something different.

In the original transformer, every query head had its own key and value head (**MHA**). That makes the keys and values, which must be stored for every past token (chapter 14), as large as the queries. **Grouped-query attention (GQA)** lets several query heads share one key/value head: SmolLM2 has 9 query heads and only 3 KV heads, so each KV head serves a group of 3. **Multi-query attention (MQA)** is the extreme: one KV head for all query heads.

Quality drops very little with GQA (it is used by Llama 2 70B, Llama 3, Mistral, Qwen and SmolLM2), while memory drops a lot. Part 4 of the demo:

```text
   model                              | heads (q/kv) | per token | 8,192 tokens
   SmolLM2-135M as trained (GQA), f32 |     9/3      |   45.0 KB |     0.38 GB
   SmolLM2-135M if it were MHA, f32   |     9/9      |  135.0 KB |     1.13 GB
   8B, 32 layers, MHA, bf16           |    32/32     |  512.0 KB |     4.29 GB
   8B, 32 layers, GQA (8 kv), bf16    |    32/8      |  128.0 KB |     1.07 GB
   8B, 32 layers, MQA (1 kv), bf16    |    32/1      |   16.0 KB |     0.13 GB
```

For an 8B model, one 8,192-token conversation needs 4.3 GB of cache with MHA and 1.1 GB with GQA. On an 80 GB GPU holding 16 GB of weights, that is the difference between 15 and 60 concurrent conversations. GQA is primarily an *inference* optimization baked into the architecture.

In code, GQA is one line: query head `h` reads KV head `h / group_size`. The test `gqa_equals_mha_with_repeated_kv_heads` checks that it gives exactly what MHA would give if each KV head were copied once per query head in its group.

### 3.5 Positions: rotary embeddings

Without extra information, attention cannot tell word order: the weighted average is the same whichever order the keys come in. Early transformers added a position vector to each embedding. Most current models use **rotary position embeddings (RoPE)** instead:

- Split each query and key head vector into pairs of numbers, and treat each pair as a point in a plane.
- Rotate pair `i` by the angle `position × θ^(−2i/d)`. Early pairs rotate quickly with position, later pairs slowly, like the hands of a clock running at many speeds. θ (the "base") is 10,000 in the original Llama and 100,000 in SmolLM2.
- Apply this to queries and keys (not values), after the projections, before the dot products.

The useful property: rotating a query by angle `a` and a key by angle `b` changes their dot product only through `a − b`. So the attention score between two tokens depends on their *distance*, not their absolute positions. Part 2 of the demo checks it:

```text
   query at    3, key at    0 (distance   3): score -5.28930
   query at  103, key at  100 (distance   3): score -5.28930
   query at 5003, key at 5000 (distance   3): score -5.28927
```

Identical up to rounding. (The last digit differs because at position 5,003 the angle `5003 × frequency` is rounded to `f32` before taking the cosine; the Hugging Face reference does the same, and section 4.4 explains why we copy its rounding on purpose.)

RoPE also tends to make scores between similar vectors shrink as distance grows, a mild bias towards nearby tokens. The demo shows the same vector used as query and key at growing distances:

```text
   d=0: +18.99  d=1: +18.01  d=4: +14.05  d=16: +10.75  d=64: +10.96  d=256: +7.13  d=1024: +6.23  d=4096: +1.57
```

### 3.6 The layout trap: interleaved versus half-split

"Pairs" can be laid out two ways in a head vector:

- **Interleaved**: pairs are neighbours, `(x0, x1), (x2, x3), ...`. The original Llama code and llama.cpp's GGUF files use this.
- **Half-split**: pair `i` is `(x_i, x_{i+d/2})`. Hugging Face's Llama implementation uses this (its `rotate_half` function), and so do checkpoints saved from it, including SmolLM2 (`"rope_interleaved": false` in its config).

Both are the same mathematics on differently ordered dimensions, and converting a checkpoint from one to the other means permuting the rows of `W_q` and `W_k`. Using the wrong layout for a checkpoint is one of the most common bugs in hand-written inference code: the model runs, produces fluent-looking nonsense or slightly worse text, and nothing crashes. The test `the_two_layouts_agree_after_permuting` checks the relationship between them.

### 3.7 The cost: quadratic in the sequence length

For causal prefill of `n` tokens, query `t` compares against `t + 1` keys, so the total is about n²/2 (query, key) pairs per head, each costing about 2d FLOPs for the score and 2d for the weighted sum. Doubling the prompt quadruples the attention work. Part 3 of the demo measures one SmolLM2 layer on one core:

```text
   tokens |      time | x previous | score matrix if stored
      256 |     9.1ms |          - |      2.4 MB
      512 |    36.5ms |        4.0 |      9.4 MB
     1024 |   147.4ms |        4.0 |     37.7 MB
     2048 |   778.1ms |        5.3 |    151.0 MB
     4096 |      4.2s |        5.5 |    604.0 MB
```

Exactly 4x per doubling up to 1,024 tokens. Beyond that the factor rises to 5.3-5.5x because the keys and values (1.5-3 MB per layer at 2,048-4,096 tokens) no longer stay in L2, so the loops start waiting on memory as well. A 4,096-token prompt costs 4.2 seconds *per layer* with this simple single-threaded kernel; SmolLM2 has 30 layers.

Two further costs:

- **Decode is linear.** Generating one more token at position `n` needs one query against `n` keys: O(n·d) per head. At long contexts, attention becomes a large share of every decode step, and it reads the entire KV cache from memory each time (chapter 14).
- **The score matrix.** A naive implementation that stores all scores before the softmax needs n² × heads × 4 bytes: 604 MB per layer at 4,096 tokens. Our implementation never stores more than one row (`scores` holds `n` floats), and chapter 20's FlashAttention goes further, computing the output tile by tile without ever holding a full row.

## 4. The code

All of it is in [`src/lib.rs`](src/lib.rs); the demo is [`src/main.rs`](src/main.rs).

### 4.1 Head bookkeeping

<!-- file: src/lib.rs -->
```rust
    /// Which key/value head query head `h` reads.
    pub fn kv_head(&self, h: usize) -> usize {
        h / self.group_size()
    }
```

With 9 query heads and 3 KV heads, the group size is 3: query heads 0-2 read KV head 0, 3-5 read KV head 1, 6-8 read KV head 2. MHA is the special case group size 1, MQA the case of a single KV head. This one integer division is all GQA changes in the attention loop.

### 4.2 The attention loop

<!-- file: src/lib.rs -->
```rust
    for t in 0..n_q {
        let pos = q_start + t;
        // Causal: keys 0..=pos. Otherwise: every key.
        let visible = if causal { (pos + 1).min(n_kv) } else { n_kv };
        for h in 0..n_heads {
            let kvh = heads.kv_head(h);
            let query = &q[t * q_row + h * d..t * q_row + (h + 1) * d];
            let s = &mut scores[..visible];
            for (j, score) in s.iter_mut().enumerate() {
                let key = &k[j * kv_row + kvh * d..j * kv_row + (kvh + 1) * d];
                *score = dot(query, key) * scale;
            }
            softmax(s);
            let o = &mut out[t * q_row + h * d..t * q_row + (h + 1) * d];
            o.fill(0.0);
            for (j, &p) in s.iter().enumerate() {
                let value = &v[j * kv_row + kvh * d..j * kv_row + (kvh + 1) * d];
                for (oi, &vi) in o.iter_mut().zip(value) {
                    *oi += p * vi;
                }
            }
        }
    }
```

Line by line:

- `pos = q_start + t`: the absolute position of this query. `q_start` is 0 when processing a whole sequence, and the number of earlier tokens when processing only new tokens (chapter 14 uses exactly this).
- `visible`: the causal mask as a *range* instead of a mask matrix. Keys past `pos` are never touched at all, which is both correct and saves half the work compared with computing all scores and setting some to −∞.
- `query` and `key` are slices of the flat `[tokens × heads × d]` buffers (chapter 3's layout: element `[t, h, i]` is at `t·(heads·d) + h·d + i`).
- `dot` is chapter 6's SIMD dot product, dispatched to AVX-512, AVX2 or NEON.
- `softmax(s)` is chapter 8's stable softmax, applied in place to the visible scores.
- The last loop accumulates `Σ p_j · v_j` into the output slice: an `axpy` (exercise 1 of chapter 6) per key.
- `scores` is passed in by the caller: the function does not allocate, and needs at most one row of scores at a time.

The loop order (query, then head, then keys) is the simplest correct one, not the fastest. The key and value reads for one head are strided through memory (each token's keys for all KV heads sit together), which is chapter 3's access-pattern problem; exercise 6 measures what a head-major layout gains.

### 4.3 Rotary embeddings: precomputing the table

<!-- file: src/lib.rs -->
```rust
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(max_positions * half);
        let mut sin = Vec::with_capacity(max_positions * half);
        for pos in 0..max_positions {
            for &f in &inv_freq {
                let angle = f64::from(pos as f32 * f);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
```

Cosines and sines depend only on position and pair index, so they are computed once for every position up to the maximum and looked up afterwards: `max_positions × d/2` of each (for SmolLM2's 8,192 positions and d = 64, 1 MB each).

The rounding here is deliberate. The mathematically nicer choice would be to compute everything in `f64`. But the reference implementation (Hugging Face transformers) computes the inverse frequencies and the angles `pos × inv_freq` in `f32`, and only the cosine of that already-rounded angle is exact. At position 5,000 the `f32` angle can differ from the exact one by about 3 × 10⁻⁴ radians. Copying the reference's rounding keeps our numbers matching it (chapter 16 compares logits against PyTorch); being "more accurate" than the reference would show up as an unexplained difference. When your job is to reproduce a model, the reference's numerics are the specification.

### 4.4 Rotary embeddings: applying them

<!-- file: src/lib.rs -->
```rust
            RopeLayout::HalfSplit => {
                let (lo, hi) = x.split_at_mut(half);
                for ((a, b), (&c, &s)) in lo.iter_mut().zip(hi.iter_mut()).zip(cos.iter().zip(sin))
                {
                    let (x0, x1) = (*a, *b);
                    *a = x0 * c - x1 * s;
                    *b = x0 * s + x1 * c;
                }
            }
```

A 2-D rotation by angle θ maps (x0, x1) to (x0·cos θ − x1·sin θ, x0·sin θ + x1·cos θ). For the half-split layout the two members of each pair are `half` elements apart, so the head vector is split with `split_at_mut` into two non-overlapping mutable halves, and zipping them yields each pair. `split_at_mut` is the safe way to hold two `&mut` borrows into one slice at the same time (chapter 3); indexing `x[i]` and `x[i + half]` in the same statement would need two mutable borrows the compiler cannot prove disjoint.

The interleaved layout uses `chunks_exact_mut(2)` instead: each chunk is one pair.

### 4.5 KV cache size

<!-- file: src/lib.rs -->
```rust
pub fn kv_bytes_per_token(layers: usize, heads: Heads, bytes_per_value: usize) -> usize {
    2 * layers * heads.n_kv_heads * heads.head_dim * bytes_per_value
}
```

Keys and values (the 2), for every layer, every KV head, every dimension. This formula is worth memorizing: it decides how many users fit on a GPU (chapter 30).

## 5. Run it

```bash
cargo test -p ch12-attention
cargo run --release -p ch12-attention
```

The demo's output on the reference machine appears in full, piece by piece, in section 3.

## 6. The Rust behind it

**Flat buffers plus index arithmetic, wrapped in slices.** Q, K and V are single `&[f32]` buffers with a documented layout; the loop takes sub-slices for each head. Every slice carries its length, so a wrong index is a panic with a clear message, not a silent read of a neighbouring head's data.

**`split_at_mut` for two mutable halves.** The RoPE rotation needs to modify `x[i]` and `x[i + d/2]` together. `split_at_mut` turns one `&mut [f32]` into two disjoint `&mut [f32]`, which the borrow checker accepts because they provably do not overlap.

**Const items as test fixtures.** `const HEADS: Heads = Heads { ... }` works because `Heads` is a plain struct of integers; the tests share one configuration without a function call.

**Struct update syntax.** `Heads { n_kv_heads: 4, ..HEADS }` builds a copy with one field changed: MHA from the GQA test configuration.

**Scratch parameters instead of allocation.** `attention(..., scores: &mut [f32])` follows chapter 8's rule: the caller owns the memory, so the function can run inside the per-token loop without touching the allocator.

## 7. Mistakes you will make

- **The wrong RoPE layout** for the checkpoint: fluent nonsense, no error.
- **Applying RoPE to values**, or after the attention instead of before. Only queries and keys are rotated.
- **Off-by-one in the causal mask** (`< pos` instead of `<= pos`): a token cannot see itself. The model runs and is worse.
- **Forgetting the 1/√d scale**, or applying it twice.
- **Using the wrong KV head** in GQA (for example `h % n_kv_heads` instead of `h / group_size`). Both compile; only one matches how the model was trained.
- **Positions that restart at 0** for each new chunk of tokens. After the prompt, generated tokens must continue at `prompt_length`, `prompt_length + 1`, ...

## 8. How the professionals do it

- **FlashAttention** (Dao et al., 2022-2024) is the standard GPU attention kernel: tiled, never storing the score matrix, fused with softmax. Chapter 20 builds its CPU equivalent.
- **Separate prefill and decode kernels.** Prefill attention is a batch of matrix products (compute-bound); decode attention is one query against a long cache (memory-bound, and needs splitting across the sequence to use a whole GPU: "flash-decoding"). Engines such as vLLM, TensorRT-LLM and llama.cpp ship both.
- **RoPE scaling for long contexts.** Models are often extended beyond their training length by adjusting RoPE frequencies (position interpolation, NTK-aware scaling, YaRN, Llama 3's `rope_scaling`). These appear as extra fields in the model config, and missing them produces a model that works at short lengths and falls apart at long ones.
- **Other attention variants you will meet:** sliding-window attention (Mistral, Gemma: each token sees only the last W tokens, bounding the KV cache), multi-head latent attention (DeepSeek: compresses K and V into a small latent vector per token), and attention sinks (keeping the first few tokens' keys forever when evicting old ones).

## 9. Exercises

1. **Bidirectional attention.** Run the tests with `causal = false`. Which ones fail, and why? Which kinds of models use non-causal attention?
2. **Count the FLOPs.** Derive the FLOP count of causal prefill attention for `n` tokens, `h` query heads and head size `d`. Using part 3's time for 1,024 tokens, what GFLOP/s did this kernel reach?
3. **Sliding window.** Add an optional `window: Option<usize>` so each query sees only the last `window` keys. Write a test. What does a window of 4,096 do to the KV cache of a 32k-token conversation?
4. **The layout bug.** Take random Q and K for one head, apply RoPE with the *wrong* layout to the keys only, and compare attention weights with the correct version. How different are they?
5. **Position IDs after the prompt.** A prompt has 12 tokens. What positions do the first three generated tokens get? What happens to the attention scores if the code mistakenly gives them positions 0, 1, 2?
6. **Parallel heads.** Run the heads on chapter 7's `SpinPool` for the 2,048-token case. What speed-up do you get on 4 cores, and what limits it?

## 10. Check yourself

1. What are Q, K and V, and which of them does RoPE modify?
2. Why are attention scores divided by √d?
3. Why does the causal mask make it possible to cache keys and values?
4. How much KV cache does GQA with 8 KV heads save compared with MHA with 32 heads?
5. Why does a RoPE score depend only on the distance between two tokens?
6. Why does doubling the prompt length roughly quadruple prefill attention time?

## 11. Recap

- Attention: scores = q·k/√d over visible tokens, softmax, weighted sum of values. Q, K, V and the output come from learned linear layers.
- The causal mask restricts each query to earlier tokens, so appending tokens never changes earlier outputs: the basis of the KV cache.
- Heads attend independently; GQA shares KV heads among groups of query heads, cutting the KV cache 3x for SmolLM2 and 4x for typical 8B models at little quality cost.
- RoPE rotates pairs of Q and K dimensions by position-dependent angles, making scores depend on relative position. Two memory layouts exist, and mixing them up is a silent bug.
- Prefill attention costs O(n²) (4x per doubling, measured), decode O(n) per token; a naive score matrix would need n² memory per head.

## Answers

**Exercises**

1. `the_future_cannot_change_the_past` and `weights_are_probabilities_and_respect_the_mask` fail (future tokens now influence earlier ones, and the upper triangle of the weights is no longer zero), as does `query_offset_matches_the_full_computation`: without a mask, earlier queries' outputs depend on later keys. Encoder models such as BERT and embedding models use bidirectional attention: they see a whole text at once and never generate token by token.
2. For each head, query `t` attends to `t + 1` keys; the score costs 2d FLOPs and the weighted sum another 2d, so 4d(t + 1). Summed over t = 0..n−1: 4d · n(n+1)/2 = 2d·n(n+1) per head, 2·h·d·n(n+1) in total. For n = 1,024, h = 9, d = 64: 1.21 GFLOP. In 147 ms that is 8.2 GFLOP/s, a quarter of what the single-core dot-product kernels of chapter 6 reach, because of the strided key/value access, the scalar `axpy` for the values, and the per-row softmax.
3. Replace `0..visible` with `visible.saturating_sub(window)..visible` (and shift the score indices). With a 4,096-token window, a 32k-token conversation only needs the last 4,096 tokens' keys and values: the cache stops growing at an eighth of the size, and it can be stored as a ring buffer.
4. Measured with random 64-dimensional queries and keys over 64 positions (20 trials): on average 10.5% of each row's attention weight moved to different tokens (total variation distance 0.105; worst row 0.18). Keys are rotated by angles belonging to a different pairing of their dimensions, so scores no longer depend only on distance. Random vectors give fairly flat attention, so this understates the damage for a trained model, whose attention is sharper; and every layer is affected, so the hidden states drift further from anything the model saw in training at each layer. The output is often still grammatical text, which is what makes this bug easy to miss.
5. 12, 13 and 14. Giving them 0, 1, 2 rotates their queries and keys as if they were at the start of the text; distances to the prompt tokens come out wrong (even negative), so the model sees a scrambled order and its output degrades or becomes incoherent.
6. Measured on the reference machine at 2,048 tokens: single-threaded 790-930 ms. Just copying each head's keys and values into contiguous per-head buffers made the one-thread time 489 ms (1.9x faster), because the inner loops no longer stride through memory. Running the 9 heads on a 4-thread spin pool then brought it to 215-240 ms, about 2x more. The thread gain is limited because 9 heads split into chunks of 3 keep only 3 of the 4 threads busy; splitting the work by (head, block of queries) instead would use all 4. Layout and parallelism both matter, and the layout change came first.

**Check yourself**

1. Query, key and value vectors, computed from each token's hidden state by learned linear layers; the query is compared with keys, and the values are averaged. RoPE rotates queries and keys only.
2. So that the scores' scale does not grow with the head size; otherwise softmax saturates for large d.
3. Because a query only sees earlier positions, adding a new token never changes earlier tokens' keys, values or outputs. Their keys and values can be computed once and reused for every later token.
4. KV cache size is proportional to the number of KV heads: 8 instead of 32 is 4 times smaller.
5. Rotating the query by angle a and the key by angle b changes their dot product through the rotation by a − b only, and the angles are proportional to positions, so only the difference of positions matters.
6. Each of the n queries attends to up to n keys, so the work grows as n²/2; doubling n multiplies it by 4.

## Further reading

- Vaswani et al., "Attention Is All You Need", 2017.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023; Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA), 2019.
- Next: [Chapter 13: The transformer](../13-transformer/README.md). Attention, norms, a gated MLP and residual connections, stacked 30 times.
