# Chapter 14: The KV cache

> **In one sentence:** the keys and values of earlier tokens never change during generation, so the engine computes them once, keeps them in a cache and afterwards runs the model on the newest token only, which turns generation from quadratic into linear work and splits inference into two phases with opposite bottlenecks: **prefill** (many tokens at once, limited by arithmetic) and **decode** (one token at a time, limited by memory bandwidth).

**Where this fits:** chapter 13 built a slow reference model that recomputes everything at every step. This chapter builds the real engine, tested against that reference. Everything after it extends this crate: chapter 15 adds sampling, chapter 16 loads SmolLM2's real `bf16` weights through the `Matrix` trait, chapters 18-19 add quantized matrices, chapter 20 rewrites attention, chapter 23 batches many caches, chapter 24 pages them, and chapter 26 uses `forward_all` and `truncate` for speculative decoding.

**You need:** chapter 7 (the spin pool), chapter 12 (attention, GQA, RoPE) and chapter 13 (the model).

**You will build:** a KV cache, a model generic over its weight format, a forward pass with no per-token buffers, chunked prefill, greedy generation, tests that compare every step against chapter 13, and a test that counts heap allocations.

---

## 1. The intuition

Picture someone taking minutes in a long meeting. Each time a person speaks, the minute-taker must decide what the remark means in light of everything said so far.

- **Without notes**, they would replay the whole recording from the start before every new remark. The meeting gets slower with every sentence.
- **With notes**, they keep one card per earlier remark: on the front a short label of what it was about (the **key**), on the back what it said (the **value**). For a new remark they only compare it against the labels and read the backs of the matching cards. Old cards are never rewritten.

The KV cache is that box of cards, one set per layer.

**Where the analogy breaks:** real cards are small. Here every card is a vector of numbers for every layer and every KV head: 22.5 KiB per token for SmolLM2-135M in `bf16`, and 128 KiB per token for Llama-3.1-8B. At long contexts the box of cards outweighs the model itself. And the cards are only valid for the exact conversation that produced them: change one earlier token and every later card is wrong.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **KV cache** | The keys and values of every processed position, for every layer, kept between forward passes. |
| **Prefill** | The first forward pass over the whole prompt. Fills the cache and produces the first new token. |
| **Decode** | Every later forward pass: one new token, attending to everything in the cache. |
| **TTFT / TPOT** | Time to first token (mostly prefill) and time per output token (one decode step), from chapter 1. |
| **Context length** | Number of positions in the cache: prompt plus generated tokens. |
| **Chunked prefill** | Processing a long prompt in pieces of at most `max_chunk` tokens, each continuing the cache. |
| **Memory-bound / compute-bound** | Limited by bytes moved from memory, or by arithmetic (chapter 4's roofline). |
| **Arithmetic intensity** | FLOPs per byte moved. Decode's is tiny; prefill's grows with the number of tokens. |
| **Rollback** | Shrinking the cache to an earlier length, to discard tokens (used by speculative decoding). |
| **Scratch space** | Temporary buffers allocated once and reused by every forward pass. |
| **Monomorphization** | The compiler generating a separate copy of generic code for each concrete type it is used with. |

## 3. The concepts in depth

### 3.1 Why keys and values can be cached

Look again at one layer of chapter 13's model. For position `i`, the key and value are

```text
k_i = RoPE(RMSNorm(x_i) · W_kᵀ, i)      v_i = RMSNorm(x_i) · W_vᵀ
```

and `x_i`, the hidden state at layer `l`, depends only on tokens `0..=i` because of the causal mask. Appending token `n` to the sequence changes nothing at positions `0..n`: not their hidden states, not their keys, not their values. So when the model processes token `n`, the keys and values of positions `0..n` are exactly the ones computed in earlier passes. Keep them, and the only new work is position `n` itself:

```text
without a cache, step n:  run all n + 1 positions through 30 layers, keep 1 row of logits
with a cache,    step n:  run 1 position through 30 layers,
                          attending to n cached keys/values per layer
```

What gets cached is only `k` and `v`. The query of an old position is never needed again (queries are used once, by their own position), and neither are old hidden states.

**Keys are cached after RoPE.** Each key is rotated for its own absolute position and never rotated again. Chapter 12 showed that the dot product of a rotated query and a rotated key depends only on their distance, so a cached, rotated key stays correct forever.

Chapter 13's demo generated 32 tokens from a 32-token prompt by processing 1,520 positions. With the cache, it processes 63 (the prompt once, then one position per new token except the last), and part 2 of the demo shows both give the same tokens:

```text
== 2. 32-token prompt, 32 new tokens, greedy
   without a cache (chapter 13):      7.3s
   with the KV cache:              646.2ms  (11.2x faster)
     first token (prefill of 32):  118.5ms   later tokens (decode): median 15.8ms
   same tokens generated: true
```

The speedup is less than 1,520 / 63 = 24x because the positions are not equally expensive: processing 32 prompt positions in one pass costs far less than 32 separate passes (section 3.3).

### 3.2 What the cache costs

Per token, the cache holds one key and one value vector of `head_dim` numbers, for every KV head, in every layer:

```text
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × bytes per number
```

Part 1 of the demo applies it to some real models, with `bf16` numbers:

```text
== 1. KV cache size in bf16 (2 bytes per value)
   model                      layers kv heads head dim    per token    4k tokens  128k tokens
   SmolLM2-135M                   30        3       64     22.5 KiB     90.0 MiB      2.8 GiB
   SmolLM2-360M                   32        5       64     40.0 KiB    160.0 MiB      5.0 GiB
   Llama-3.1-8B                   32        8      128    128.0 KiB    512.0 MiB     16.0 GiB
   Llama-3.1-8B without GQA       32       32      128    512.0 KiB      2.0 GiB     64.0 GiB
   Llama-3.1-70B                  80        8      128    320.0 KiB      1.2 GiB     40.0 GiB
```

Three things to take from this table:

- **The cache is per sequence.** Llama-3.1-8B's weights are 16 GB in `bf16`. One sequence at its full 128k context needs another 16 GiB. A server running 32 conversations of 4k tokens each needs 16 GiB of cache. On a GPU, the cache (not the weights) decides how many requests fit at once, which is why chapters 23 and 24 are about managing it.
- **GQA is a 4x memory saving.** The "without GQA" row is the same model with one KV head per query head (plain multi-head attention). Grouped-query attention (chapter 12) exists mostly because of this table.
- **Small models are not exempt.** SmolLM2-135M's weights are 270 MB in `bf16`; its cache at 128k tokens would be more than ten times that. (SmolLM2 was trained for 8k positions, so it never gets there.)

This chapter's cache stores `f32` (4 bytes per number), so its numbers are twice the table's. Storing it in `bf16` or 8 bits is an exercise here and a technique in chapter 18.

### 3.3 Prefill and decode are different workloads

Part 3 of the demo times one prefill pass at three prompt lengths, and one decode step:

```text
== 3. prefill versus decode
   prefill  32 tokens:   91.1ms     351 tokens/s   94.4 GFLOP/s
   prefill 128 tokens:  315.3ms     406 tokens/s  109.2 GFLOP/s
   prefill 512 tokens:     2.0s     261 tokens/s   70.1 GFLOP/s
   decode, 1 token:       18.8ms      53 tokens/s   14.3 GFLOP/s  28.6 GB/s of weights
   (each token needs 0.27 GFLOP and reads 513.0 MiB of weights)
```

The same weights, the same code, and a factor of 7 between the arithmetic rates. Chapter 4's roofline explains it:

- **Decode** reads every weight once (513 MiB in `f32`) to do 0.27 GFLOP: each 4-byte weight is used for one multiply-add, 2 FLOPs. That is 0.5 FLOP per byte, far below this machine's ridge point of about 4 FLOP per byte (chapter 4). The step is **memory-bound**: its time is the weights divided by the bandwidth. It reached 28.6 GB/s; chapter 4 measured 30-45 GB/s for this machine. The arithmetic units sit mostly idle.
- **Prefill** of `m` tokens also reads every weight once, but uses each one `m` times. Its intensity is `m × 0.5` FLOP per byte, past the ridge from about 8 tokens on. It is **compute-bound**, running at 94-109 GFLOP/s, about the speed chapter 7's parallel matmul reached on this machine (104 GFLOP/s).

This split runs through the rest of the course:

- **TTFT is a compute problem, TPOT is a bandwidth problem.** Making the matmul faster (chapter 17) mostly helps prefill. Making the weights smaller (`bf16` in chapter 16, 8 and 4 bits in chapters 18-19) mostly helps decode, almost in proportion to the bytes saved.
- **One decode step leaves the arithmetic idle.** Decoding for many users at once (a batch of sequences, chapter 23) turns `m` single-token decode steps into one `m`-token pass over the weights: nearly the same time as one step, `m` times the tokens. That is how servers get their throughput.
- **Prefill slows down on long prompts** (261 tokens/s at 512 tokens). Attention's work grows with the square of the prompt length, and this chapter's attention is simple (section 3.5).

### 3.4 Chunked prefill

The engine's scratch space holds activations for up to `max_chunk` tokens. A longer prompt goes through the layers in chunks, each one continuing the cache like a small prefill. Part 4 of the demo prefills the same 256-token prompt with different chunk sizes:

```text
== 4. prefill of 256 tokens with different chunk sizes
   chunk   1:     3.5s      73 tokens/s  scratch 234.0 KiB
   chunk   4:     1.4s     188 tokens/s  scratch 933.0 KiB
   chunk  16:     1.1s     239 tokens/s  scratch 3.6 MiB
   chunk  64:  806.8ms     317 tokens/s  scratch 14.4 MiB
   chunk 256:  755.0ms     339 tokens/s  scratch 57.8 MiB
```

This is section 3.3's roofline, measured. A chunk of 1 token is a sequence of decode steps. Going from 1 to 4 tokens per chunk multiplies the throughput by 2.6, because the weights are read once per chunk instead of once per token. From 4 to 16 it gains another 1.3x. Past that the pass is compute-bound and bigger chunks buy little: across several runs, chunks of 16 to 256 tokens measured between 240 and 340 tokens/s, with no consistent order. The scratch memory, meanwhile, grows in proportion to the chunk.

So real engines prefill long prompts in chunks of a few hundred to a few thousand tokens. It bounds activation memory whatever the prompt length, and (chapter 25) it lets a scheduler slip decode steps for other users between the chunks of a long prompt, instead of making them wait for all of it.

Most of the scratch memory is the logits buffer: `chunk × 49,152 × 4` bytes, 48 MiB of the 57.8 MiB at chunk 256. Only `forward_all` needs logits for every row (exercise 4).

### 3.5 Decode gets slower as the context grows

Each decode step still attends to every cached position. Part 5 of the demo measures one step at growing context lengths:

```text
== 5. one decode step at different context lengths
   context   16:  16.6ms per token (1.00x)  cache holds 720.0 KiB (f32)
   context  256:  17.2ms per token (1.03x)  cache holds 11.2 MiB (f32)
   context 1024:  19.5ms per token (1.17x)  cache holds 45.0 MiB (f32)
   context 2048:  28.5ms per token (1.71x)  cache holds 90.0 MiB (f32)
```

At 2,048 tokens the step reads 90 MiB of cache on top of 513 MiB of weights (18% more bytes), and attention's arithmetic, 4 × 2,048 × 576 × 30 = 141 MFLOP, is more than half of the weights' 269 MFLOP (chapter 13's exercise 2 found the two are equal at about 3,900 tokens). The measured cost, 71% more time, is larger than either suggests, because this chapter's attention is written for clarity, not speed:

- It splits work by query head: 9 heads on 4 threads means one thread gets 3 heads while the others get 2, and everyone waits for it.
- Each of the 3 query heads that share a KV head reads that head's keys and values separately.
- It calls the dispatched `dot` once per 64-number key, and does the weighted sum of values with a plain loop.

Chapter 20 rewrites attention (tiling, online softmax, splitting long contexts across threads). Until then, remember that decode cost has two parts: a fixed part (the weights) and a part proportional to the context (the cache).

### 3.6 The engine's design

**The weight format is a type parameter.** The model is `Model<W: Matrix>`, where `Matrix` is a trait with one real job: multiply a batch of activations by the matrix. This chapter implements it for `f32` weights (`DenseF32`); chapter 16 adds `bf16`, chapters 18-19 add quantized formats. The forward pass never changes. Only the storage and the dot-product kernel do.

**All temporary memory lives in a `Scratch`.** Every activation buffer, the per-head attention buffers and the matmul's temporary are allocated once, sized for `max_chunk` tokens. The forward pass borrows slices of them. A decode step allocates no activation memory at all; a test counts the few small allocations that remain (section 4.6).

**The cache layout is `[layer][kv_head][position][head_dim]`.** For attention, one head's keys at positions `0..n` are then one contiguous run of `n × 64` numbers, the best access pattern chapter 4 found. The cost is that storing a new position writes 3 separate pieces (one per KV head) instead of one, which is cheap, and that the cache must be sized for its maximum length up front, which is not: a cache of capacity 8,192 holds 8,192 positions of memory even for a 10-token conversation. Chapter 24 fixes that with pages.

**Rolling back is free.** `truncate(len)` only changes the length. The numbers beyond it stay in memory, but attention reads only positions `0..len`, and the next forward pass overwrites them before anything reads them. Speculative decoding (chapter 26) relies on this to discard rejected guesses.

## 4. The code

The engine is in [`src/lib.rs`](src/lib.rs), its matrix multiplication in [`src/matmul.rs`](src/matmul.rs), the demo in [`src/main.rs`](src/main.rs) and the allocation test in [`tests/allocations.rs`](tests/allocations.rs).

### 4.1 The `Matrix` trait

<!-- file: src/lib.rs -->
```rust
pub trait Matrix: Send + Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Bytes of weight data: what one full pass over the matrix reads.
    fn bytes(&self) -> usize;
    /// Row `r` converted to `f32` (used for embedding lookups).
    fn row_to_f32(&self, r: usize, out: &mut [f32]);
    /// `y[m × rows] = x[m × cols] · selfᵀ`, split across the pool's threads.
    /// `scratch` is reusable temporary space.
    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    );

    /// Multiplies the same `x` by several matrices with the same number of
    /// columns: `outputs[i] = x · matrices[i]ᵀ`. The default simply calls
    /// `matmul` for each; chapter 17 overrides it to do all of them in one
    /// parallel pass, which matters when the matrices are small.
    fn matmul_many(
        pool: &mut SpinPool,
        matrices: &[&Self],
        x: &[f32],
        outputs: &mut [&mut [f32]],
        m: usize,
        scratch: &mut Vec<f32>,
    ) where
        Self: Sized,
    {
        for (w, y) in matrices.iter().zip(outputs.iter_mut()) {
            w.matmul(pool, x, y, m, scratch);
        }
    }
}
```

- **`Send + Sync`**: the weights are read by every pool thread at once, so they must be safe to share.
- **`row_to_f32`** serves the embedding lookup, which reads one row per token. With tied embeddings, the same matrix is the LM head, so it needs both operations.
- **`matmul`** takes activations in `f32` whatever the weight format. Every format in this course keeps activations in `f32` and changes only how weights are stored.
- **`matmul_many`** is a *provided* method: it has a default body, so implementations get it for free and may override it. The forward pass uses it where several matrices read the same input (Q, K and V; gate and up). Its `where Self: Sized` keeps the trait usable as `dyn Matrix`, since a method taking `&[&Self]` could not be called through a trait object.

The `f32` implementation forwards to the shared kernel, with chapter 6's `dot` as the inner loop:

<!-- file: src/lib.rs -->
```rust
    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        matmul_pooled(
            pool, x, &self.data, y, m, self.cols, self.rows, scratch, dot,
        );
    }
```

### 4.2 The kernel: chapter 7's matmul, without the allocation

<!-- file: src/matmul.rs -->
```rust
pub fn matmul_pooled<W, D>(
    pool: &mut SpinPool,
    x: &[f32],
    w: &[W],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Vec<f32>,
    dot: D,
) where
    W: Sync,
    D: Fn(&[W], &[f32]) -> f32 + Sync,
{
    assert_eq!(x.len(), m * k, "x must be m×k");
    assert_eq!(w.len(), n * k, "w must be n×k");
    assert_eq!(y.len(), m * n, "y must be m×n");
    if m == 1 {
        pool.for_each_chunk_mut(y, 1, |start, y_part| {
            let rows = &w[start * k..(start + y_part.len()) * k];
            for (out, row) in y_part.iter_mut().zip(rows.chunks_exact(k)) {
                *out = dot(row, x);
            }
        });
        return;
    }
    scratch.clear();
    scratch.resize(n * m, 0.0);
    pool.for_each_chunk_mut(scratch, m, |start, yt_part| {
```

This is chapter 7's `matmul_nt_pool_with`: threads split the weight rows, each thread writes whole rows of a transposed output, and small groups of weight rows stay in cache while every input row passes through them. Two changes:

- **The weight element type `W` and the kernel `dot` are generic.** `matmul_pooled::<f32, _>` with chapter 6's `dot` and `matmul_pooled::<Bf16, _>` with `dot_bf16` are two separate compiled functions, each with its kernel inlined. Chapters 16-19 reuse this function for every format.
- **The transposed temporary is `scratch`**, owned by the caller. `clear` plus `resize` reuse its capacity, so after the first call at a given size nothing is allocated.

For decode (`m == 1`) there is no transpose: each thread computes a range of outputs directly.

### 4.3 The cache

<!-- file: src/lib.rs -->
```rust
    fn offset(&self, layer: usize, head: usize, pos: usize) -> usize {
        ((layer * self.kv_heads + head) * self.capacity + pos) * self.head_dim
    }

    /// Stores one position's keys and values (all KV heads) for a layer.
    pub fn store(&mut self, layer: usize, pos: usize, k_row: &[f32], v_row: &[f32]) {
        assert!(pos < self.capacity, "position {pos} beyond the cache");
        let d = self.head_dim;
        for head in 0..self.kv_heads {
            let at = self.offset(layer, head, pos);
            self.k[at..at + d].copy_from_slice(&k_row[head * d..(head + 1) * d]);
            self.v[at..at + d].copy_from_slice(&v_row[head * d..(head + 1) * d]);
        }
    }

    // ...

    /// Keys of one head for positions `0..upto`, as `[upto × head_dim]`.
    pub fn keys(&self, layer: usize, head: usize, upto: usize) -> &[f32] {
        let at = self.offset(layer, head, 0);
        &self.k[at..at + upto * self.head_dim]
    }
```

`offset` is the whole layout decision in one line: layer, then KV head, then position, then the number within the head. `keys` returns a borrowed slice straight into the cache, so attention reads the cached keys in place: no copy, and the borrow checker guarantees nobody writes to the cache while attention holds that slice.

### 4.4 The forward pass

`forward_last` returns the logits of the last token; `forward_all` returns them for every token. Both call `forward`, which checks capacity and splits the input into chunks:

<!-- file: src/lib.rs -->
```rust
        assert!(!tokens.is_empty(), "nothing to process");
        assert!(
            cache.len() + tokens.len() <= cache.capacity(),
            "KV cache full: {} + {} > {}",
            cache.len(),
            tokens.len(),
            cache.capacity()
        );
        // Long inputs go through the layers in chunks of at most `max_chunk`
        // tokens. Only the last chunk needs logits.
        let chunks = tokens.len().div_ceil(s.max_chunk);
        for (i, chunk) in tokens.chunks(s.max_chunk).enumerate() {
            let last_chunk = i + 1 == chunks;
            self.forward_chunk(pool, chunk, cache, s, all && last_chunk, last_chunk);
        }
```

The capacity check comes first, so a full cache is an error before any state changes, not a corrupted cache. Each chunk then runs through every layer:

<!-- file: src/lib.rs -->
```rust
        let c = &self.config;
        let m = tokens.len();
        let (h, q_dim, kv_dim, inter) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size);
        let start = cache.len();

        for (t, &token) in tokens.iter().enumerate() {
            self.embed
                .row_to_f32(token as usize, &mut s.x[t * h..(t + 1) * h]);
        }
        // Each `s.field[..]` below borrows one field of the scratch; the
        // borrow checker sees that different fields never overlap.
        for (l, layer) in self.layers.iter().enumerate() {
            // Attention block.
            norm_rows(
                &s.x[..m * h],
                &layer.attn_norm,
                c.rms_norm_eps,
                &mut s.normed[..m * h],
            );
            W::matmul_many(
                pool,
                &[&layer.wq, &layer.wk, &layer.wv],
                &s.normed[..m * h],
                &mut [
                    &mut s.q[..m * q_dim],
                    &mut s.k[..m * kv_dim],
                    &mut s.v[..m * kv_dim],
                ],
                m,
                &mut s.matmul,
            );
            for t in 0..m {
                let pos = start + t;
                let (qt, kt) = (t * q_dim..(t + 1) * q_dim, t * kv_dim..(t + 1) * kv_dim);
                self.rope.apply_heads(&mut s.q[qt], pos);
                self.rope.apply_heads(&mut s.k[kt.clone()], pos);
                cache.store(l, pos, &s.k[kt.clone()], &s.v[kt]);
            }
            self.attention(pool, l, start, m, cache, s);
```

Compared with chapter 13's `forward`:

- **Positions are absolute.** Token `t` of this chunk is at position `start + t`, where `start` is how many positions the cache already holds. RoPE and the cache both use that position. Using `t` instead is the classic KV cache bug: the first chunk works, every later one is wrong, and the `chunked_prefill_matches_one_big_prefill` test exists to catch it.
- **Keys and values go into the cache right after RoPE**, before attention, so each new token attends to itself as well as to the past.
- **Every buffer is a slice of the scratch**, cut to the chunk's `m` rows.
- **Q, K and V come from one `matmul_many` call**, because all three multiply the same normalized input (the MLP's gate and up projections do the same). With this chapter's matrices it is three ordinary `matmul`s; chapter 17 measures why doing them as one pass is faster.

After the layers, the cache length advances once, and the LM head runs on only the rows that need logits:

<!-- file: src/lib.rs -->
```rust
        cache.len = start + m;

        if !want_logits {
            return;
        }
        // Final norm and LM head: for every row, or only the last one.
        let rows = if all_logits { 0..m } else { m - 1..m };
```

With SmolLM2's 49,152-token vocabulary and tied embeddings, the LM head is 108 MiB of the 513 MiB read per token in `f32`. Skipping it for every prompt position but the last is a large part of why prefill is cheap.

### 4.5 Attention against the cache

<!-- file: src/lib.rs -->
```rust
        let q = &s.q;
        pool.for_each_chunk_mut(&mut s.heads, 1, |first, my_heads| {
            for (i, hs) in my_heads.iter_mut().enumerate() {
                let head = first + i;
                let kvh = heads.kv_head(head);
                for t in 0..m {
                    let visible = start + t + 1;
                    let query = &q[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                    let keys = cache.keys(layer, kvh, visible);
                    let scores = &mut hs.scores[..visible];
                    for (score, key) in scores.iter_mut().zip(keys.chunks_exact(d)) {
                        *score = dot(query, key) * scale;
                    }
                    softmax(scores);
```

Each pool thread takes some query heads. For every token `t` of the chunk, it scores the query against the `start + t + 1` positions that token may see (the causal mask is simply "stop at `visible`"), applies softmax, and sums the values with those weights into the head's own output buffer. Afterwards, the results are gathered from head-major order into the `[token × head × dim]` layout the output projection expects.

`q` is a shared borrow of one scratch field and `&mut s.heads` an exclusive borrow of another. Rust allows both at once because they are different fields of the same struct (section 6).

The function starts with a check you can ignore for now: a model can carry a replacement attention (`with_attention`), used in chapter 20 to plug in flash attention without changing anything else.

### 4.6 Generation, and counting allocations

<!-- file: src/lib.rs -->
```rust
    cache.clear();
    let mut tokens = prompt.to_vec();
    for step in 0..new_tokens {
        let start = Instant::now();
        // First step: the whole prompt. Afterwards: only the newest token.
        let input = if step == 0 {
            prompt
        } else {
            &tokens[tokens.len() - 1..]
        };
        let next = argmax(model.forward_last(pool, input, cache, scratch));
        on_token(next, start.elapsed());
        tokens.push(next);
    }
    tokens
```

The first step is the prefill (its time is the TTFT), every later one a decode step (its time is the TPOT). `input` borrows either the caller's prompt or the last element of `tokens`; the borrow ends before `tokens.push`, so the push is allowed.

To check the claim that a decode step allocates no activation memory, `tests/allocations.rs` replaces the global allocator with one that counts:

<!-- file: tests/allocations.rs -->
```rust
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller upholds `alloc`'s contract, which we pass on.
        unsafe { System.alloc(layout) }
    }
```

It runs 50 decode steps and finds 17 allocations, 1,176 bytes in total, per step, the same at every step. 17 is exactly the number of parallel calls in a step of the 2-layer test model (8 per layer plus the LM head): chapter 7's `for_each_chunk_mut` builds a small list of chunk locks each time. Building and freeing one such list takes about 15 ns on the reference machine (measured in isolation), so SmolLM2's 241 per token cost about 4 µs of a 17 ms step. They stay. The test pins the number down, so a new allocation cannot sneak in unnoticed.

### 4.7 The tests

<!-- file: src/lib.rs -->
```rust
        // Prefill the prompt, then decode 6 tokens one at a time.
        let mut logits = model
            .forward_last(&mut pool, &tokens, &mut cache, &mut scratch)
            .to_vec();
        for _ in 0..6 {
            let reference = w.forward(&mut pool, &tokens);
            let want = &reference[(tokens.len() - 1) * vocab..];
            assert!(close(&logits, want), "diverged at length {}", tokens.len());
```

Every test builds the engine's model from chapter 13's random weights and compares:

- **cached decoding** against the reference at every step,
- **chunked prefill** (chunks of 4) against one prefill of the whole prompt,
- **`forward_all`** against every row of the reference,
- **`truncate`**: after a wrong turn and a rollback, the logits match the ones before the wrong turn,
- **greedy generation** with and without the cache, token for token,
- **overflowing the cache** panics with a clear message.

## 5. Run it

```bash
cargo test -p ch14-kv-cache
cargo run --release -p ch14-kv-cache
```

The output is shown in sections 3.1 to 3.5. The run takes about 45 seconds on the reference machine, most of it chapter 13's uncached generation and the chunk-size sweep. Decode times vary by 10-20% between runs on the reference machine; the ratios are stable.

## 6. The Rust behind it

**Returning a borrow of the scratch.** `forward_last<'s>(..., scratch: &'s mut Scratch) -> &'s [f32]` returns the logits in place, without copying 49,152 numbers. The signature ties the result to the scratch: while you hold the logits, the scratch stays borrowed, so you cannot run another forward pass that would overwrite them. The tests call `.to_vec()` exactly when they need logits to survive the next pass.

**Disjoint field borrows.** In `forward_chunk`, `&s.normed[..]` and `&mut s.q[..]` coexist because the compiler tracks borrows per field. It stops working at a function boundary: an earlier version of this code held `let x = &mut s.x[..]` across the call `self.attention(..., s)`, and the compiler refused, because `attention` takes the whole `Scratch` and might touch `x`. The fix is to borrow fields at the point of use (as the code does now) or to pass the fields a function needs as separate arguments.

**Generics or trait objects?** `Model<W: Matrix>` is generic, so each weight format gets its own compiled forward pass. A `Vec<Box<dyn Matrix>>` would also work here without a measurable cost: the dynamic call happens once per matrix per forward pass (211 times per SmolLM2 token, a few nanoseconds each, against a step of 17 ms), not once per multiply. What must not be dynamic is the inner kernel, and `matmul_pooled`'s generic `dot: D` keeps that monomorphized either way. Generics were chosen because all matrices in a model share one format; a model that mixes formats (keeping some layers at higher precision, as chapter 19 discusses) would use trait objects.

**A global allocator in a test.** `#[global_allocator]` replaces the allocator for a whole binary. Each file in `tests/` is its own binary, so the counting allocator affects only that test, and no other test runs in the same process to disturb the counts. Implementing `GlobalAlloc` is `unsafe` because the allocator must uphold the contract every other piece of code relies on; forwarding to `System` keeps it.

**`impl FnMut(u32, Duration)`** lets the caller watch each token as it is produced: the demo records times, chapter 16's CLI prints text, chapter 21's server sends it to a client. The loop does not know or care.

## 7. Mistakes you will make

- **Using the chunk index instead of the absolute position** for RoPE or for the cache slot. Single-chunk prefill works, everything after it silently degrades.
- **Advancing the cache length inside the layer loop.** Layer 2 then writes its keys one position later than layer 1. The length must advance once per chunk, after all layers.
- **Caching keys before RoPE** and then forgetting to rotate them at read time (or rotating them with the wrong position). Cache them rotated.
- **Reusing a cache for a new conversation without `clear()`**: the new prompt attends to the old one.
- **Indexing the cache by query head instead of KV head.** With GQA, query head 7 of SmolLM2 reads KV head 2; a cache indexed by query head is 3 times too big, or reads garbage.
- **Allocating per token.** A `Vec` for the logits, a `Vec` for each matmul output: each is cheap, together they add latency and memory churn. Measure with a counting allocator.
- **Assuming decode is slow because of arithmetic.** It is waiting for memory. Faster math kernels do little for it; fewer bytes do.

## 8. How the professionals do it

- **Every production engine is built around the KV cache**: Hugging Face `transformers` (a `DynamicCache` that grows, or a `StaticCache` preallocated like ours), llama.cpp (a preallocated cache per layer with bookkeeping for several sequences), vLLM and SGLang (paged caches, chapter 24).
- **The cache is often stored in fewer bits**: vLLM's `--kv-cache-dtype fp8` and llama.cpp's `--cache-type-k q8_0` / `--cache-type-v q8_0` halve it relative to 16 bits, which doubles the number of sequences that fit.
- **Architectures shrink the cache itself**: multi-query attention (one KV head, Shazeer 2019), grouped-query attention (Llama 2 70B onward), sliding-window attention (Mistral 7B keeps only the last 4,096 positions), and DeepSeek-V2's multi-head latent attention, which caches one compressed vector per token per layer instead of full keys and values.
- **Chunked prefill** is standard in vLLM and SGLang, mainly so a long prompt does not stall other users' decode steps (the Sarathi-Serve paper; chapter 25).
- **Prefill and decode are sometimes run on different machines** ("disaggregated serving", as in DistServe and in NVIDIA Dynamo), because one wants compute and the other bandwidth. The cache is then sent from the prefill machine to the decode machine.

## 9. Exercises

1. **Another model.** Qwen2.5-7B has 28 layers, 4 KV heads and head dimension 128. How big is its cache per token and at 32k tokens, in `bf16`?
2. **How many users fit?** An 80 GB GPU holds Llama-3.1-8B in `bf16` (16 GB). Ignoring activations, how many 4k-token sequences fit in the remaining memory? Two such GPUs hold Llama-3.1-70B (140 GB of weights): how many 4k-token sequences fit?
3. **One token, two speeds.** Part 4 of the demo prefills at 73 tokens/s with chunks of 1 token, while decode at short context runs at about 60 tokens/s (16.6 ms per step). Both push one token through the model per pass. Why is the chunk-1 prefill faster? Check your answer with numbers.
4. **Logits only where needed.** `Scratch` sizes its logits for `max_chunk` rows, but `forward_last` needs one. Add a separate `max_logit_rows` to `Scratch::new` (make `forward_all` check it) and recompute part 4's scratch column.
5. **A `bf16` cache.** Store the cache in `Bf16` (chapter 2) and convert keys and values to `f32` as attention reads them. Do the tests still pass with the same tolerance? What happens to decode at 2,048 tokens of context?
6. **Stale numbers.** `truncate` leaves old keys and values in memory. Write a test that fills the cache, truncates it, continues with different tokens and compares against a fresh cache, and explain why it passes.

## 10. Check yourself

1. Why do the keys and values of earlier positions not change when a token is appended?
2. Why are queries not cached?
3. Write the formula for KV cache bytes per token.
4. Why is decode memory-bound and prefill compute-bound, with the same weights and code?
5. What does chunked prefill trade, and why does throughput stop improving past about 16 tokens per chunk here?
6. Which part of a decode step grows with context length?
7. Why is `truncate` enough to roll back speculative tokens?

## 11. Recap

- Causal attention makes past keys and values immutable, so the engine caches them and processes only new tokens: 63 positions instead of 1,520 for chapter 13's example, 11.2x faster in this run.
- The cache costs `2 × layers × kv_heads × head_dim × bytes` per token, per sequence: 128 KiB for Llama-3.1-8B in `bf16`, 16 GiB at 128k tokens. It, not the weights, limits how many sequences a server holds.
- Prefill is compute-bound (94-109 GFLOP/s here); decode is memory-bound (28.6 GB/s of weights, 14 GFLOP/s). TTFT improves with faster math, TPOT with fewer bytes and with batching.
- Chunked prefill bounds memory at almost no cost once chunks pass the roofline's ridge (about 16 tokens here).
- Decode cost = a fixed part (weights) + a part proportional to context (cache and attention arithmetic).
- The engine is generic over its weight format, allocates its buffers once and is tested against the reference at every step.

## Answers

**Exercises**

1. 2 × 28 × 4 × 128 × 2 = 57,344 bytes = 56 KiB per token. At 32,768 tokens: 1.75 GiB.
2. Llama-3.1-8B at 4k tokens: 4,096 × 128 KiB = 512 MiB = 0.537 GB per sequence. (80 − 16) / 0.537 ≈ 119 sequences. Llama-3.1-70B: 4,096 × 320 KiB = 1.25 GiB = 1.34 GB per sequence; (160 − 140) / 1.34 ≈ 14 sequences. The 70B model serves far fewer users per GPU, which is part of why it costs more per token.
3. Prefill chunks other than the last skip the final norm and the LM head. With tied embeddings, the LM head is the 49,152 × 576 embedding matrix: 108 MiB of the 513 MiB a decode step reads in `f32`. Without it, a pass reads 405 MiB, 79% of the bytes. A memory-bound decode step of 16.6 ms (the context-16 measurement) predicts 0.79 × 16.6 ≈ 13.1 ms per chunk-1 token, 76 tokens/s. The measurement was 73 tokens/s.
4. The logits at chunk 256 take 256 × 49,152 × 4 bytes = 48 MiB. With one logits row, the scratch at chunk 256 drops from 57.8 MiB to 9.9 MiB. What remains (activations, per-head buffers and the matmul temporary) is about 40 KB per token of chunk.
5. Measured: with a `bf16` cache, the largest relative difference from the reference in the tests is 4.6e-4, so the two tests that compare logits at 1e-4 fail, and they pass at 1e-3. Greedy generation still produces exactly the same tokens. Loosening the tolerance is legitimate here: the engine now computes something slightly different on purpose (rounded keys and values), and 1e-3 still catches real bugs, which produce errors of order 1. Decode at 2,048 tokens: 24.6-29.7 ms with the `bf16` cache against 26.2-33.2 ms with `f32` over three runs each, no difference beyond the noise. The cache is only 90 MiB against 513 MiB of weights, and this chapter's attention is limited by its own inefficiency (section 3.5), not by the bytes it reads. A smaller cache matters where the cache is large: long contexts, many sequences, and chapter 24's server, where it decides how many sequences fit at all.
6. The test passes because attention reads positions `0..visible` only, and `visible` never exceeds the current length plus the chunk being written; every position in the chunk is written by `store` before attention reads it, in the same layer.

**Check yourself**

1. A position's hidden state at every layer depends only on the tokens at or before it (the causal mask), and its key and value are computed from that hidden state and its position. Nothing after it can change them.
2. A query is used only by its own position, once, to attend to the past. Later positions use their own queries.
3. 2 × layers × KV heads × head dimension × bytes per number.
4. Decode uses each weight once per pass (0.5 FLOP per byte in `f32`), below the machine's ridge point; prefill uses each weight once per token in the pass, so its intensity grows with the number of tokens and passes the ridge.
5. It trades a little speed (more passes over the weights) for bounded memory. Past about 16 tokens the pass is already compute-bound, so reading the weights fewer times no longer helps.
6. Reading the cached keys and values and the attention arithmetic over them, both proportional to the number of cached positions.
7. Attention only reads positions below the cache length, and later writes overwrite the discarded positions before they are read.

## Further reading

- Pope et al., "Efficiently Scaling Transformer Inference", 2022: KV cache memory, prefill versus decode, and the arithmetic behind both, at scale.
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019: multi-query attention, motivated by cache bandwidth.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023.
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", 2024: chunked prefill.
- Next: [Chapter 15: Sampling](../15-sampling/README.md). Choosing the next token when greedy is not what you want.
