# Chapter 20: FlashAttention

> **In one sentence:** attention can be computed in one pass over the keys and values, keeping a running maximum and sum instead of a full row of scores, and that one fact lets you cut the work into tiles and pieces, run the pieces on different threads, and merge the results exactly; on a CPU, most of the speed then comes from dividing the work evenly and vectorizing it, not from the memory the one pass saves.

**Where this fits:** chapter 14 made each decode step cost more as the context grows, and chapter 17's profile showed that at long contexts most of a step is attention, not matrix products. This chapter rewrites attention. The same "partial results that merge exactly" idea returns in chapter 24 (paged attention) and chapter 29 (GPUs, where FlashAttention was born).

**You need:** chapter 8 (softmax, `exp_fast`), chapter 12 (attention, grouped-query attention), chapter 14 (the KV cache and its layout), chapter 17 (interleaved measurement), chapter 6 (`#[target_feature]`).

**You will build:** the online softmax and its merge rule, a tiled attention that handles all query heads of a KV head together, split-KV for decoding (several threads on one head's context, merged afterwards), AVX-512 kernels through multiversioning, and measurements on SmolLM2-135M, including the first version, which was slower than what it replaced.

---

## 1. The intuition

You are averaging exam results, but each result is weighted by `e^score`, and the scores arrive one by one. The textbook method reads the list three times: once to find the highest score (to keep `e^score` from overflowing), once to compute the weights and their total, once to form the weighted average.

The one-pass method keeps three running numbers: the highest score so far, the total weight so far, and the weighted sum so far, with every weight measured relative to that highest score. When a higher score arrives, everything you have accumulated is measured against the old maximum, so you multiply it once by `e^(old max − new max)` (a number below 1) and carry on. It is like keeping a running total in one currency: when you switch currency, you convert the total once, not every past purchase.

Two people can each process half of the list this way and combine their three numbers at the end with the same conversion. That is what lets several threads share one long context.

**Where the analogy breaks:** currency conversion is exact; floating-point rescaling is not. The one-pass result equals the three-pass one up to rounding in the last bits, not bit for bit, because the additions happen in a different order. The tests compare within a tolerance for that reason.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Online softmax** | Softmax (and the weighted sum after it) in one pass, keeping a running maximum and sum and rescaling when the maximum grows (Milakov and Gimelshein, 2018). |
| **Tile** | A block of consecutive keys (and their values) processed together; here 64. |
| **FlashAttention** | Attention computed tile by tile with the online softmax, never storing a full row of scores (Dao et al., 2022). |
| **Split-KV / flash-decoding** | Splitting one head's keys into parts processed in parallel, each producing a partial state, merged at the end. |
| **Task** | Here, one KV head × one block of query tokens × one part of the keys. The unit of work handed to a thread. |
| **Load balance** | Whether every thread gets the same amount of work. The slowest thread decides when a parallel step ends. |
| **Multiversioning** | Compiling the same source code several times, for different instruction sets, and choosing at run time. |

## 3. The concepts in depth

### 3.1 Where the time goes

Chapter 17 measured a decode step at a 4,096-token context; part 1 of this chapter's demo repeats that measurement with both attentions (4 vCPUs of an Intel Xeon with AVX-512, see chapter 17):

```text
== 1. a decode step at 4,096 tokens of context, per part
   built-in  step  39.60 ms: matrix products  13.58 ms, the rest (mostly attention)  26.02 ms (66%)
   flash     step  29.89 ms: matrix products  13.89 ms, the rest (mostly attention)  16.00 ms (54%)
```

The matrix products cost the same at any context length; attention grows with it, and at 4,096 tokens it is two thirds of the step. Flash attention takes 10 ms out of the 26.

### 3.2 The online softmax

For one query, attention is `out = Σ_j p_j v_j` with `p_j = e^(s_j) / Σ_k e^(s_k)` and `s_j = q · k_j / √d`. The one-pass state after some of the keys is:

```text
m   = the largest score seen
l   = Σ e^(s_j − m)          over the keys seen
acc = Σ e^(s_j − m) · v_j    over the keys seen (a vector of d numbers)
```

and the answer is `acc / l` when all keys have been seen. A new tile of scores with maximum `m_t` updates it:

```text
m'   = max(m, m_t)
r    = e^(m − m')                      (≤ 1; 0 on the very first tile, where m = −∞)
l'   = l · r   + Σ_tile e^(s_j − m')
acc' = acc · r + Σ_tile e^(s_j − m') · v_j
```

Merging two states `(m₁, l₁, acc₁)` and `(m₂, l₂, acc₂)` over different keys is the same rule with the second state in place of the tile: rescale each by `e^(mᵢ − m')` and add.

Every exponent is `≤ 0`, so nothing overflows, which is the reason chapter 8 subtracted the maximum in the first place. Rescaling once per tile rather than once per key keeps the extra work small: one `exp` and `d` multiplications per 64 keys.

### 3.3 What changes on a CPU

FlashAttention was invented for GPUs, for prefill. There, the scores of a 4,096-token prompt form a 4,096 × 4,096 matrix per head, 64 MB in `f32`, which the standard method writes to and reads back from the GPU's main memory. Keeping tiles in the GPU's small on-chip memory removes that traffic, and that was the whole speedup.

A CPU decoding one token has one row of scores per head: 4,096 × 4 bytes = 16 KB, which fits in the L1 or L2 cache anyway. The memory argument mostly disappears. What remains useful:

- **Reading each key once for a whole GQA group.** SmolLM2 has 9 query heads and 3 KV heads: each KV head serves 3 query heads. Chapter 14's attention runs query heads independently, reading every key and value 3 times. Here a task takes a tile of keys and runs all 3 query heads over it while it is in L1.
- **Splitting the work any way you like.** Because partial states merge exactly, the keys of one head can be divided among threads.
- **No separate softmax pass**, and chapter 8's `exp_fast` instead of the standard library's `exp`.

### 3.4 Splitting the work, and the first version

Decode has one query token and 3 KV heads, so there are 3 natural tasks for 4 threads. Chapter 14's attention has 9 tasks (one per query head), which is no better: chapter 7's `for_each_chunk_mut` gives each thread `ceil(9 / 4) = 3` consecutive tasks, so 3 threads work and the fourth has nothing.

The first version of this chapter split each head's keys into 3 parts, making 9 tasks, "at least two per thread". It was **slower** than chapter 14's attention at every context length (0.91-0.99x in decode, 0.91-0.94x in prefill). Two reasons, found by measuring rather than guessing:

1. **9 tasks still means 3 busy threads**, by the same `ceil(9 / 4) = 3` arithmetic. The split added merging work and gained no parallelism.
2. **The innermost loops were chapter 14's.** For every key, both versions call chapter 6's `dot` (AVX-512 inside, but reached through a function call and a check of which instruction set to use), then run a plain Rust loop over the 64 values, which the compiler, targeting baseline x86-64, can only vectorize with SSE2, 4 floats per instruction. Those two steps run 4,096 × 9 × 30 times per decode step at this context. The new structure removed work around them (a separate softmax pass, re-reading keys that were in cache anyway) and left them alone.

The fixes: choose the number of parts so that the task count is a multiple of the thread count (3 KV heads × 4 parts = 12 tasks, 3 per thread), and write the two inner steps for AVX-512, compiled into the task's loop with no calls (section 4.4).

### 3.5 Which fix mattered

Part 3 runs every combination at a 4,096-token context, each against chapter 14's attention:

```text
== 3. decode at 4,096 tokens: built-in -> flash variants (4 threads)
   3 tasks, portable :   36.08ms ->   33.90ms, speedup 1.05x (80% of pairs: 1.00-1.20x)
   9 tasks, portable :   37.14ms ->   36.98ms, speedup 0.99x (80% of pairs: 0.92-1.20x)
   12 tasks, portable:   36.28ms ->   31.08ms, speedup 1.20x (80% of pairs: 1.04-1.34x)
   3 tasks, AVX-512  :   39.81ms ->   28.93ms, speedup 1.36x (80% of pairs: 1.19-1.48x)
   9 tasks, AVX-512  :   37.01ms ->   28.69ms, speedup 1.23x (80% of pairs: 1.20-1.44x)
   12 tasks, AVX-512 :   37.11ms ->   25.60ms, speedup 1.45x (80% of pairs: 1.26-1.53x)
```

Read it in pairs:

- "9 tasks, portable" is the first version: no faster than the built-in.
- Same kernels, 3 → 12 tasks: 1.05x → 1.20x (portable), 1.36x → 1.45x (AVX-512). Four busy threads instead of three should cut attention's time by a quarter. For AVX-512: 28.93 ms minus about 13.9 ms of matrix products leaves 15.0 ms, a quarter less is 11.3 ms, predicting a 25.2 ms step; measured 25.60 ms.
- Same tasks, portable → AVX-512: 1.05x → 1.36x (3 tasks), 1.20x → 1.45x (12 tasks). The kernels were the larger fix.
- 9 tasks is no better than 3: the same three threads do the same work, plus a merge.

### 3.6 Decode and prefill

Part 2 measures the default configuration at several context lengths:

```text
== 2. one decode step, built-in -> flash attention (4 threads)
   context  256:   12.99ms ->   13.72ms, speedup 0.95x (80% of pairs: 0.78-1.03x)
   context 1024:   18.18ms ->   16.37ms, speedup 1.10x (80% of pairs: 1.00-1.25x)
   context 2048:   23.42ms ->   19.43ms, speedup 1.26x (80% of pairs: 1.03-1.37x)
   context 4096:   36.27ms ->   24.81ms, speedup 1.45x (80% of pairs: 1.31-1.58x)
```

At 256 tokens attention is a small part of the step and the two are within noise of each other (the range spans 1.0). The gain grows with the context, as attention's share does.

Prefill is where FlashAttention was meant to shine, and part 4 shows the largest gains:

```text
== 4. prefill, built-in -> flash attention (4 threads)
    256 tokens:  748.60ms ->  685.13ms, speedup 1.08x (80% of pairs: 0.98-1.14x)
      = 342 -> 374 tokens/s
   1024 tokens:     4.40s ->     3.28s, speedup 1.33x (80% of pairs: 1.22-1.42x)
      = 233 -> 312 tokens/s
   2048 tokens:    12.51s ->     8.24s, speedup 1.52x (80% of pairs: 1.38-1.68x)
      = 164 -> 248 tokens/s
```

In prefill there are many query tokens, so no splitting is needed (a 512-token chunk makes 3 KV heads × 32 blocks = 96 tasks, 24 per thread). Each tile of 64 keys is used by 16 tokens × 3 heads while it sits in L1, where chapter 14's attention reads every key again for every token and every head. Note that the tokens per second still fall as the prompt grows: attention's work grows with the square of the prompt length, and flash attention does the same arithmetic, only faster.

## 4. The code

The online softmax, on its own and with tests, is in [`src/online.rs`](src/online.rs); the attention in [`src/attention.rs`](src/attention.rs); plugging it into the model in [`src/lib.rs`](src/lib.rs); the measurements in [`src/main.rs`](src/main.rs).

### 4.1 A hook in chapter 14's model

Chapter 14's `Model` now has an optional replacement for its attention:

<!-- file: ../14-kv-cache/src/lib.rs -->
```rust
pub type AttentionFn = dyn Fn(&mut SpinPool, &AttentionInput<'_>, &mut [f32]) + Send + Sync;
```

`AttentionInput` carries the layer, the chunk's position and length, its queries and a shared reference to the cache. `with_flash` in this chapter is one line: `model.with_attention(move |pool, input, out| flash_attention(pool, input, out, &options))`. Everything else in the model (weights, norms, the cache, sampling) is unchanged, so the tests can compare the two attentions on the same model.

### 4.2 The online update

<!-- file: src/attention.rs -->
```rust
fn online_update<K: Kernels>(state: &mut [f32], scores: &mut [f32], values: &[f32]) {
    let tile_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let new_max = state[0].max(tile_max);
    let rescale = exp_fast(state[0] - new_max);
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = exp_fast(*s - new_max);
        sum += *s;
    }
    let (head, acc) = state.split_at_mut(2);
    head[1] = head[1] * rescale + sum;
    K::accumulate(acc, rescale, scores, values);
    head[0] = new_max;
}
```

Section 3.2, line by line. A state is `[m, l, acc...]` in one slice of `d + 2` floats. `split_at_mut(2)` borrows the two numbers and the vector separately, so `acc` can be handed to the kernel while `l` is updated. The scores are overwritten by their weights: the tile's buffer is reused rather than allocating another.

### 4.3 Tasks

<!-- file: src/attention.rs -->
```rust
    let splits = if opts.key_splits > 0 {
        opts.key_splits
    } else {
        // `for_each_chunk_mut` gives every thread the same number of tasks
        // only if the task count is a multiple of the thread count. Split
        // the keys just enough for that (4 parts for decode's 3 KV heads
        // on 4 threads), but never into parts shorter than a tile.
        let base = c.num_kv_heads * blocks;
        let threads = pool.threads();
        let wanted = threads / gcd(base, threads);
        wanted.min((start + m).div_ceil(opts.key_block)).max(1)
    };
```

`base` is the number of tasks without splitting (KV heads × query blocks). The smallest `s` that makes `base × s` a multiple of `threads` is `threads / gcd(base, threads)`: for 3 and 4 that is 4; for a prefill of 256 tokens in blocks of 16, `base = 48`, already a multiple of 4, so `s = 1` and nothing is split.

Each task gets its own `d + 2`-float state per (token, query head), all in one `Vec`. (The query block is capped at the chunk's length, so a decode step allocates states for one token, not sixteen. Chapter 23 adds `flash_decode_many`, which runs the decode tasks of many sequences in one parallel pass with the same code and reuses the state buffer from call to call.) `for_each_chunk_mut` hands every thread a disjoint `&mut` slice of it, so threads write without locks; afterwards, one loop merges each head's parts with the rule of section 3.2 and writes the output.

The loop inside a task:

<!-- file: src/attention.rs -->
```rust
    while k < task.k1 {
        let tile_end = (k + key_block).min(task.k1);
        for t in task.t0..task.t1 {
            // Causal: token t sees positions up to start + t.
            let end = tile_end.min(start + t + 1);
            if end <= k {
                continue;
            }
            let n = end - k;
            for g in 0..group {
                let head = task.kv_head * group + g;
                let q = &input.q[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                let tile = &mut scores[..n];
                for (sc, key) in tile.iter_mut().zip(keys[k * d..end * d].chunks_exact(d)) {
                    *sc = K::dot(q, key) * scale;
                }
                let st = &mut state[((t - task.t0) * group + g) * stride..][..stride];
                online_update::<K>(st, tile, &values[k * d..end * d]);
            }
        }
        k = tile_end;
    }
```

Tile outermost, then query tokens, then the query heads of the group: one tile of keys and values (64 × 64 floats each, 16 KB each) is loaded once and used by every (token, head) pair of the task. The causal mask is per token: during prefill, token `t` of the chunk sees positions up to `start + t`, so later tokens use more of the tile. `scores` is a stack array of `MAX_TILE` floats: no allocation per task.

### 4.4 Multiversioning

The loop above is generic over `K: Kernels`, two functions: `dot` and `accumulate` (rescale `acc`, then add the tile's weighted value rows). `Portable` implements them in plain Rust. `Avx512D64` implements them with AVX-512 intrinsics for heads of exactly 64 dimensions, where a head is four 16-float registers:

<!-- file: src/attention.rs -->
```rust
                let r = _mm512_set1_ps(rescale);
                let mut a: [__m512; 4] = std::array::from_fn(|i| {
                    _mm512_mul_ps(_mm512_loadu_ps(acc.as_ptr().add(16 * i)), r)
                });
                for (j, &pj) in p.iter().enumerate() {
                    let w = _mm512_set1_ps(pj);
                    let v = values.as_ptr().add(64 * j);
                    for (i, ai) in a.iter_mut().enumerate() {
                        *ai = _mm512_fmadd_ps(w, _mm512_loadu_ps(v.add(16 * i)), *ai);
                    }
                }
```

The whole accumulator stays in four registers for the tile: 4 fused multiply-adds per key, where the portable loop, compiled for SSE2, needs 16 multiplies and 16 adds.

The intrinsics need the function they are compiled into to have AVX-512 enabled. They are inlined (`#[inline(always)]`) into `run_task_with::<Avx512D64>`, which is itself inlined into:

<!-- file: src/attention.rs -->
```rust
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn run_task_avx512(
        input: &AttentionInput<'_>,
        task: Task,
        group: usize,
        state: &mut [f32],
        key_block: usize,
    ) {
        run_task_with::<Avx512D64>(input, task, group, state, key_block);
    }
```

so the whole task, loops, `exp_fast` and all, is compiled once for AVX-512. `run_task` checks the CPU at run time (`is_x86_feature_detected!`, which caches its answer) and picks this version or `run_task_with::<Portable>`. Without `#[inline(always)]`, the compiler may keep `run_task_with` or the kernels as separate functions compiled for the baseline target; the intrinsics, themselves `#[target_feature]` functions, could then not be inlined into them and would each become a call.

## 5. Run it

```bash
cargo test -p ch20-flash-attention
cargo run --release -p ch20-flash-attention                 # all parts: about 10 minutes
cargo run --release -p ch20-flash-attention -- decode       # or: profile, ablation, prefill
```

The tests compare flash attention with chapter 14's on a small random model, for several tile sizes, block sizes and split counts, with prefill in chunks of 8 and 64 and then decode, including a configuration with 64-dimension heads that takes the AVX-512 path. The timings vary from run to run; the ratios are more stable than the times.

## 6. The Rust behind it

**`Box<dyn Fn + Send + Sync>` as a hook.** The model stores `Option<Box<AttentionFn>>`. A trait object keeps `Model<W>` a single type whichever attention it uses, so chapter 21 can move either kind to its engine thread. The `Send + Sync` bounds are part of the type: a `Model` is `Send` or `Sync` only if all its fields are, and a box of any closure would make every model neither, hook or not. With the bounds, only closures that are safe to move and share can be stored.

**Generic code, compiled per instruction set.** Rust compiles a generic function once per type it is used with. `run_task_with::<Portable>` and `run_task_with::<Avx512D64>` are two separate machine-code functions from one source. `#[target_feature]` applies to the function it is written on and to what is inlined into it, which is why the kernels and the task body are `#[inline(always)]`. Clippy's pedantic set warns about `inline(always)`; the module states why it is needed with `#![expect(clippy::inline_always, reason = ...)]`.

**`split_at_mut` for two mutable views.** The borrow checker does not let you hold `&mut state[1]` and `&mut state[2..]` from two separate index expressions at once. `split_at_mut(2)` returns two non-overlapping `&mut` slices, proven disjoint by the function itself.

**Disjoint `&mut` chunks instead of locks.** All task states live in one `Vec<f32>`; `for_each_chunk_mut` splits it with `chunks_mut` and moves one `&mut` chunk to each worker. The borrow checker guarantees no two threads write the same state, and there is no locking in the hot loop.

## 7. Mistakes you will make

- **Merging an empty part.** A part whose keys are all masked has `m = −∞`, and `e^(m − m')` with both `−∞` is `e^NaN`. Skip parts with `m = −∞` when merging (the code does), and make sure every query sees at least one key.
- **Masking per block instead of per token.** In prefill, the tokens of a block see different numbers of keys. Applying the last token's limit to all of them lets early tokens see the future; the tests catch it because the logits change.
- **Assuming more tasks means more parallelism.** It does only if the scheduler spreads them. Count what each thread actually gets.
- **Restructuring around an unchanged inner loop.** If the loop that runs for every key is the same, reorganizing everything around it gains little. Look first at what the innermost loop compiles to: plain Rust loops are compiled for the build's target, which for x86-64 means SSE2 unless you say otherwise.
- **Rescaling on every key.** Correct, but it does an `exp` and `d` multiplications per key instead of per tile.

## 8. How the professionals do it

- **FlashAttention 1, 2 and 3** (Dao et al.) are CUDA kernels. Version 2 parallelizes over query blocks as well as heads and reduces non-matrix work; version 3 uses Hopper's asynchronous copies and FP8. They matter most for prefill and training, where the score matrix is large.
- **Flash-decoding** (Dao, Haziza, Massa, Sizov, 2023) is this chapter's split-KV: for long contexts in decode, split the keys, compute partial states in parallel, merge. FlashInfer and vLLM use the same idea, and paged attention (chapter 24) combines it with a block-structured cache.
- **llama.cpp** has a CPU flash attention (`-fa`) and stores its KV cache in `f16` by default, halving the bytes read per step, something this chapter does not do.
- On CPUs, production engines also vectorize across queries in prefill (several query tokens against one key at a time, which needs no horizontal sums) and use `bf16` matrix instructions such as Intel AMX for the `QKᵀ` and `PV` products.

## 9. Exercises

1. Run the ablation with `key_splits: 2` on 4 threads. How many tasks, how many busy threads? Predict the result, then measure.
2. Add an AVX2 version of the kernels (8 floats per register, a head is 8 registers) and a `run_task_avx2` with `#[target_feature(enable = "avx2,fma")]`. Measure it against the AVX-512 version on the same machine.
3. Store the KV cache in `bf16` (chapter 2) and convert tiles to `f32` as they are loaded. What does it do to decode speed at 4,096 tokens, and to the model's output?
4. The merge after the parallel part runs on one thread. At what number of splits would it start to matter? Estimate its cost from the sizes.

## 10. Check yourself

1. Write the three numbers of an online-softmax state and what the final output is in terms of them.
2. Why is the rescale factor always at most 1?
3. The original FlashAttention saves memory traffic. Why does that argument mostly not apply to a CPU decoding one token?
4. With 3 KV heads, 4 threads and chapter 7's `for_each_chunk_mut`, how many threads work with 3 tasks, 9 tasks and 12 tasks?
5. Why did the first version gain nothing, and what does `#[inline(always)]` have to do with fixing it?
6. What happens to the result if the same keys are split into 4 parts instead of 1?

## 11. Recap

- Online softmax keeps `(max, sum, acc)` and rescales once per tile; two states over different keys merge exactly. That is all of FlashAttention's math.
- On a CPU in decode, the gains come from reading keys once per GQA group, spreading the work evenly over all threads, and vectorizing the inner loops.
- The first version was no faster than chapter 14's attention. An ablation showed why: its 9 tasks kept only 3 of 4 threads busy, and its innermost loops were chapter 14's. With 12 tasks and AVX-512 kernels, decode at 4,096 tokens is 1.45x faster.
- Prefill gains more (1.52x at 2,048 tokens), because each tile of keys is reused by a whole block of query tokens.

## Answers

**Exercises**

1. 3 KV heads × 2 parts = 6 tasks; `ceil(6 / 4) = 2` tasks per chunk gives chunks of 2, 2 and 2: three busy threads again, no better than 3 tasks, plus merging. The automatic choice avoids this by requiring a multiple of the thread count.
2. On this machine expect it between the portable and the AVX-512 versions: half the lanes, and the reduction in `dot` costs relatively more. On a CPU without AVX-512 it is the version that would run.
3. The keys and values read per step halve, which helps when attention is limited by memory bandwidth, as it is at long contexts; the conversion costs a shift per value. Rounding keys and values to `bf16` changes the logits slightly; compare with chapter 18's KL measurement.
4. Per (head, token) the merge touches `splits × (d + 2)` floats and computes `2 × splits` exponentials. For decode: 9 heads × 4 parts × 66 floats, about 2,400 floats per layer, around a microsecond; 30 layers make tens of microseconds against a 25 ms step. It would matter only with hundreds of splits, or with long prefill chunks that are also split.

**Check yourself**

1. The largest score seen `m`, the sum `l = Σ e^(s_j − m)`, the weighted sum `acc = Σ e^(s_j − m) v_j`. Output: `acc / l`.
2. It is `e^(m_old − m_new)`, and the maximum can only grow, so the exponent is `≤ 0`.
3. The scores of one query are one row of `context` floats (16 KB at 4,096 tokens), which fits in the cache; there is no large score matrix whose traffic to main memory could be saved.
4. 3 tasks: 3 threads (one each). 9 tasks: chunks of 3, so 3 threads. 12 tasks: chunks of 3, so 4 threads.
5. It kept chapter 14's innermost steps (a dispatched `dot` call per key, and a value loop compiled for SSE2), and those are most of attention's time. `#[target_feature]` only applies to the function it is on and to what is inlined into it; `#[inline(always)]` puts the generic task body and the AVX-512 kernels inside `run_task_avx512`, so the whole loop is compiled for AVX-512, with no calls per key.
6. Mathematically nothing; numerically, the last bits may differ because additions happen in a different order. The tests allow `1e-4` relative difference.

## Further reading

- Milakov, Gimelshein, "Online normalizer calculation for softmax", 2018.
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022; Dao, "FlashAttention-2", 2023; Shah et al., "FlashAttention-3", 2024.
- Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference", 2023 (PyTorch blog).
- Rabe, Staats, "Self-attention Does Not Need O(n²) Memory", 2021.
- Next: [Chapter 21: The engine thread](../21-engine-thread/README.md). The model gets its own thread, and requests start arriving from outside.
