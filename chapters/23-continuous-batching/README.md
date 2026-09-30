# Chapter 23: Continuous batching

> **In one sentence:** a decode step reads every weight of the model to produce one token, so a step that produces one token for each of many requests costs little more than a step for one; an engine that rebuilds its batch at every step, letting requests join and leave between steps, turns that into several times the throughput, up to the limits set by how fast its kernels compute and by each request's own KV cache, which no batching can share.

**Where this fits:** chapter 21's engine serves one request at a time, and chapter 22 showed what that costs the clients who wait. This chapter makes the engine serve all of them together. Chapter 24 then fixes the memory this chapter wastes (each request reserves a full-length KV cache), and chapter 25 tunes the policy deciding what goes into each step.

**You need:** chapter 4 (the roofline: bytes against FLOPs), chapter 14 (the forward pass and KV cache), chapter 17 (the tiled kernel, `instrument`), chapter 20 (flash attention), chapter 21 (the engine thread and its handle).

**You will build:** `forward_batch`, one forward pass over several sequences with their own caches; a `bf16` weight layout and kernel designed for small batches; a batched decode attention (added to chapter 20); and the batching engine, which admits requests into KV cache slots, plans each step, samples, and retires finished requests. Then measurements: step cost against batch size, eight clients at once, clients arriving one after another, and whether batching changes the model's answers.

---

## 1. The intuition

A bus between two towns. Every trip costs the same fuel whether it carries one passenger or forty; carrying one passenger per trip, as chapter 21's engine does, wastes almost all of it. So fill the bus.

There are two ways. A **static** bus waits at the terminal until it is full, drives the whole route, and only then takes new passengers: people wait at the terminal, and passengers going one stop ride along with those going to the end. A **continuous** bus stops at every station: people get off when they arrive and new people get on, so the bus is full most of the time and nobody waits for the whole group.

**Where the analogy breaks:** each passenger also brings luggage that only they can carry (their KV cache): the bus trip is shared, the luggage is not, and with enough passengers the luggage costs more than the trip. And the bus has a top speed (the processor's arithmetic): past some number of passengers, each extra one slows the trip down.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Batch** | The sequences processed in one forward pass. |
| **Step** | One forward pass of the engine: one new token for every decoding request, plus prompt chunks. |
| **Static batching** | Forming a batch, running it to completion, then forming the next. |
| **Continuous (iteration-level) batching** | Rebuilding the batch at every step: requests join and leave between steps (Yu et al., "Orca", 2022). |
| **Slot** | A KV cache reserved for one request while it runs. |
| **Throughput** | Tokens produced per second by the whole engine. |
| **TPOT** | Time per output token, as one client sees it: the time between its tokens. |
| **Packing** | Rearranging a weight matrix once, at load time, into the order a kernel reads it. |
| **Batch invariance** | Getting bit-for-bit the same result for a request whatever else is in its batch. |

## 3. The concepts in depth

### 3.1 Why a batch is almost free

A decode step multiplies one activation row by every weight matrix. Each `bf16` weight (2 bytes) is read from memory and used for one multiply-add (2 FLOPs): 1 FLOP per byte. Chapter 4 measured this machine at about 40 GB/s from memory with 4 threads, and computed its AVX-512 peak at about 540 GFLOP/s. At 1 FLOP per byte, memory allows 40 GFLOP/s, and the arithmetic units sit idle more than 90% of the time.

With `B` sequences in the step, each weight read is used `B` times: `B` FLOPs per byte, for the same bytes. Until `B` reaches the machine's ridge point (here about 540 / 40 ≈ 13), the extra sequences should cost almost nothing, and throughput should grow almost linearly with `B`.

A GPU has far more arithmetic per byte: an H100 has about 990 TFLOP/s of dense `bf16` and 3.35 TB/s of memory bandwidth, a ridge near 300. That is why GPU servers batch hundreds of requests, and why batching is the single most important optimization in LLM serving.

### 3.2 What is shared and what is not

`forward_batch` stacks the tokens of every sequence in the step into one `[M × hidden]` matrix, `M` being the total number of tokens: one per decoding sequence, a whole chunk for a sequence that is prefilling. Most of a transformer layer does not care where a row came from:

- **Shared:** the Q, K, V, output and MLP projections (one matrix product each, over all `M` rows), the norms, the residual additions, the LM head (for the last token of each sequence that needs logits).
- **Per sequence:** the position of each token (for RoPE), where its keys and values are stored (its own cache), and attention (each sequence attends only to its own past).

The per-sequence part is small in code but not in cost. Each sequence's attention reads its entire KV cache, and that cannot be shared. For SmolLM2 one position costs 30 layers × 3 KV heads × 64 dimensions × 2 (keys and values) × 4 bytes = 46 KB, so 256 tokens of context are 11.8 MB per sequence per step. At `B = 16`, that is 189 MB of cache against 269 MB of weights; with longer contexts, the cache dominates.

### 3.3 Measured: the first kernel stops at 3.8x

Part 1 times one decode step for `B` sequences, each with 256 tokens of context, and splits it into matrix products and the rest (4 threads, the reference machine of chapter 17):

```text
== 1. one decode step for B sequences (256 tokens of context each, 4 threads)
   chapter 17's tiled bf16 kernel:
         B   step time   tokens/s   vs B = 1   matrix products   the rest
         1     14.2 ms         71       1.0x           13.9 ms     0.3 ms
         2     20.6 ms         97       1.4x           17.4 ms     3.2 ms
         4     25.3 ms        158       2.2x           19.5 ms     5.7 ms
         8     35.3 ms        227       3.2x           25.6 ms     9.7 ms
        16     60.6 ms        264       3.7x           40.8 ms    19.8 ms
        32    120.0 ms        267       3.8x           82.8 ms    37.2 ms
        64    253.1 ms        253       3.6x          168.4 ms    84.7 ms
```

Throughput grows, but far from linearly: 2 sequences cost 1.45 times one, and beyond 16 nothing more is gained. Two things are wrong, one per column.

**The matrix products.** From `B = 2`, they take chapter 17's prefill path, a 4 × 4 tile kernel. It was written for long prompts, not for 2 to 16 rows: rows that do not fill a group of four (`B = 2`, or the last rows of any batch) fall back to one dot product per weight row and batch row, and every output ends with a horizontal sum across a register. At `B = 16` the products run at 16 × 270 MFLOP / 40.8 ms ≈ 106 GFLOP/s, a fifth of the peak.

**The rest,** which grows by about 1.2 ms per sequence. In the first version of this chapter it was about 1.9 ms per sequence: each sequence's attention was a separate call to chapter 20's `flash_attention`, with its own dispatch to the thread pool and a freshly zeroed buffer of partial states sized for 16 query tokens (about 150 KB). Chapter 20 now has `flash_decode_many`, which puts every decoding sequence's tasks into one parallel pass and reuses its buffers. What remains is mostly each sequence reading its own cache (section 3.2), which is real work.

### 3.4 A kernel for small batches

The fix for the first column is to change the layout of the weights. Chapter 17's kernel computes each output as a dot product: a weight row times an activation row, summed across the 16 lanes of a register at the end. The packed layout instead stores, for each block of 16 output rows, the 16 weights of column `c` next to each other. One load then gives column `c` of 16 different outputs, and

```text
acc[r] += broadcast(x[r][c]) × w[c][16 outputs]
```

adds the contribution of column `c` to 16 outputs of batch row `r` in one fused multiply-add. Up to 8 batch rows run together, one register of 16 sums each: every weight is loaded and converted from `bf16` once and used by all of them, and there are no horizontal sums at all. The packing happens once, when the model is loaded.

```text
   packed bf16 weights:
         B   step time   tokens/s   vs B = 1   matrix products   the rest
         1     12.4 ms         81       1.0x           11.0 ms     1.4 ms
         2     14.0 ms        143       1.8x           11.8 ms     2.2 ms
         4     18.7 ms        214       2.6x           14.7 ms     4.0 ms
         8     29.4 ms        272       3.4x           20.4 ms     9.0 ms
        16     56.1 ms        285       3.5x           35.3 ms    20.8 ms
        32     98.8 ms        324       4.0x           60.4 ms    38.4 ms
        64    179.5 ms        357       4.4x          107.4 ms    72.0 ms
```

Two sequences now cost 1.13 times one, instead of 1.45. The matrix products of 64 sequences take 107 ms instead of 168 ms (about 160 GFLOP/s), and a single sequence is a little faster too (12.4 ms against 14.2 ms). This machine is noisy: in two other runs, the packed kernel reached 4.5x at `B = 16` and 3.4x, so read the table as a shape, not as exact values.

What limits it now is the second column. At `B = 64`, "the rest" is 72 ms, 40% of the step, growing linearly with `B`. The weights are shared; the caches are not. That is the fundamental limit of batching decode, and it gets worse with longer contexts: chapter 24 stores the cache more efficiently, and grouped-query attention (chapter 12) exists to make it smaller in the first place.

### 3.5 The engine

The batching engine is chapter 21's engine thread with a different loop. Its state:

- **slots:** `max_batch` KV caches, allocated once; a request holds one while it runs;
- **waiting:** jobs received but not yet admitted, in order of arrival;
- **active:** the requests holding slots, each with its sampler, how much of its prompt is processed, and its last token.

Every iteration of the loop:

```text
receive new jobs (sleep in recv only if there is nothing at all to do)
admit waiting jobs into free slots, oldest first
plan the step:   every decoding request   → its last token
                 prefilling requests      → the next chunk of their prompt,
                                            until the step has max_step_tokens
forward_batch    (one pass for everything planned)
for each request that got logits: sample, send the token, check whether it is done
retire finished requests: send Done, free the slot
```

Decoding requests go first in the plan, so a long new prompt never stalls them: it is prefilled in chunks alongside their decode steps (chapter 25 studies the trade-offs of that choice). A request that got logits for the first time (its prompt just finished) samples its first token in the same step.

Clients use chapter 21's `EngineHandle`, unchanged: `ch21::channel()` hands any engine loop the queue behind a handle. Chapter 22's server works with this engine without modification.

### 3.6 Eight clients at once

Part 2 is chapter 21's experiment again: eight clients submit at the same moment, 64 tokens each, first to chapter 21's engine, then to the batching engine (both with packed weights and flash attention, so batching is the only difference). One of three runs:

```text
== 2. eight clients at once, 64 tokens each
   one at a time: 512 tokens in 7524 ms, 68 tokens/s
      client  first token      done
           0       124 ms   1077 ms
           1      1157 ms   2040 ms
           ...
           7      6681 ms   7524 ms
   batched: 512 tokens in 2322 ms, 221 tokens/s
      client  first token      done
           0       615 ms   2322 ms
           1       615 ms   2322 ms
           ...
           7       616 ms   2322 ms
```

Across three runs: 65-73 tokens/s one at a time, 190-221 batched, about 3 times the throughput. The last client finishes after 2.3 s instead of 7.5 s.

Look at client 0, though: its first token comes after 615 ms instead of 124 ms (612-921 ms across runs). All eight prompts arrived together and fit in one step (about 280 tokens), so everyone waited for all eight prefills. Batching improves the total and the average; it can make the best case worse. Whether that is acceptable, and how to limit it, is chapter 25's subject.

### 3.7 Clients that arrive one after another

With continuous batching, a request that arrives while others are decoding joins at the next step. Part 3 sends a client every 150 ms:

```text
== 3. eight clients arriving 150 ms apart, 64 tokens each
   one at a time:
      client  arrives   waits  first token after  done at
           0     1 ms    0 ms              88 ms  1039 ms
           1   154 ms  886 ms             948 ms  1870 ms
           2   310 ms 1561 ms            1619 ms  2749 ms
           3   454 ms 2295 ms            2366 ms  3585 ms
           4   601 ms 2984 ms            3052 ms  4432 ms
           5   754 ms 3678 ms            3740 ms  5382 ms
           6   901 ms 4481 ms            4569 ms  6343 ms
           7  1056 ms 5287 ms            5363 ms  7244 ms
   batched:
      client  arrives   waits  first token after  done at
           0     0 ms    0 ms              77 ms  1897 ms
           1   154 ms    5 ms              67 ms  2133 ms
           2   300 ms    9 ms              79 ms  2308 ms
           3   453 ms    9 ms              93 ms  2438 ms
           4   600 ms   20 ms             119 ms  2530 ms
           5   754 ms    5 ms              91 ms  2583 ms
           6   900 ms   13 ms             108 ms  2645 ms
           7  1053 ms    8 ms             102 ms  2685 ms
```

Batched, nobody waits more than one step (20 ms at most), and every first token arrives in about 0.1 s. One at a time, the waits grow with every arrival, to 5.3 s. This is the difference between continuous and static batching as well: a static batcher would have held client 1 until client 0's batch finished.

The price is visible in the last column: client 0 alone would finish after about 1.0 s; sharing its steps with seven others, it finishes after 1.9 s. Every step is slower when more requests are in it (section 3.4), so each client's time per token grows with the batch. Throughput against per-client speed is the central trade-off of serving.

### 3.8 Does batching change the answers?

In principle it should not: each request's arithmetic is the same whatever else is in the batch. In practice, floating-point addition is not associative (chapter 2), and a kernel may add a request's numbers in a different order depending on the batch. Part 4 compares request 0 alone and in a batch of eight, logit by logit:

```text
== 4. request 0 alone vs in a batch of eight: are the results the same?
   chapter 17's tiled bf16 kernel:
      prefill logits: 44149 of 49152 differ, by at most 2.4e-5
      decode logits : 42344 of 49152 differ, by at most 1.7e-5
      greedy answers of up to 96 tokens: 8 of 8 identical
   packed bf16 weights:
      prefill logits: bit for bit identical
      decode logits : bit for bit identical
      greedy answers of up to 96 tokens: 8 of 8 identical
```

Chapter 17's kernel computes an output one way when its row is part of a group of four and another way (one dot product) when it is left over, so a request's logits depend on its position in the batch. The differences are tiny, but a greedy choice between two nearly tied tokens can flip, and from then on the answers diverge. It did not happen in these 8 × 96 tokens; over millions of requests it does.

The packed kernel is batch-invariant by construction: each output's sum runs over the columns in the same order, in its own register, whether 1 or 8 rows are processed. Attention can still break invariance: `flash_decode_many` splits a sequence's keys into parts when there are too few sequences to keep every thread busy, so the number of parts depends on the batch. In the logit comparison the context is 36 tokens, one tile, and nothing is split. In the whole answers it is: once a request decoded alone passes 64 tokens its keys are split into 2 parts (3 past 128, 4 past 192), while in the batch of eight there are enough tasks without splitting. The answers still matched here.

## 4. The code

The batched forward pass is in [`src/forward.rs`](src/forward.rs), the packed layout in [`src/packed.rs`](src/packed.rs), the engine in [`src/engine.rs`](src/engine.rs), the demo in [`src/main.rs`](src/main.rs). The batched decode attention is in chapter 20's [`attention.rs`](../20-flash-attention/src/attention.rs).

### 4.1 One pass over many sequences

<!-- file: src/forward.rs -->
```rust
pub struct BatchSeq<'a> {
    /// The tokens to process: the next chunk of its prompt, or the token it
    /// sampled last.
    pub tokens: &'a [u32],
    /// Its cache. The tokens continue the sequence from `cache.len()`.
    pub cache: &'a mut KvCache,
    /// Whether to compute logits for its last token. A prompt chunk that is
    /// not the prompt's last needs none.
    pub logits: bool,
}
```

`forward_batch` records where each sequence's tokens sit in the stacked matrix (a `Range` of rows each), embeds all tokens, runs the layers, and advances each cache by its number of tokens. The part of a layer that is per sequence:

<!-- file: src/forward.rs -->
```rust
    // ...then, per sequence: its positions and its cache...
    for (seq, rows) in seqs.iter_mut().zip(rows) {
        let start = seq.cache.len();
        for (t, r) in rows.clone().enumerate() {
            let pos = start + t;
            let (qr, kr) = (r * q_dim..(r + 1) * q_dim, r * kv_dim..(r + 1) * kv_dim);
            model.rope().apply_heads(&mut s.q[qr], pos);
            model.rope().apply_heads(&mut s.k[kr.clone()], pos);
            seq.cache.store(l, pos, &s.k[kr.clone()], &s.v[kr]);
        }
    }
```

The position of row `r` is its sequence's `cache.len()` plus its index within the sequence, not `r`: rows of different sequences are at unrelated positions. Then the attention inputs are built; the decoding ones (one row each) go to `flash_decode_many` together, prompt chunks to `flash_attention` one at a time. At the end, only the last row of each sequence that wants logits goes through the final norm and the LM head, gathered into one small matrix so that the LM head (the largest matrix: 49,152 × 576) is read once.

### 4.2 The packed kernel

<!-- file: src/packed.rs -->
```rust
            for c in 0..k {
                let bits = _mm256_loadu_si256(wp.add(c * LANES).cast::<__m256i>());
                let wv = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(bits)));
                for (a, xr) in acc.iter_mut().zip(&xs) {
                    *a = _mm512_fmadd_ps(_mm512_set1_ps(*xr.add(c)), wv, *a);
                }
            }
```

Per column: load 16 `bf16` weights (32 bytes), widen them to `f32` (a `bf16` is the top half of an `f32`: zero-extend to 32 bits, shift left by 16, as in chapter 6), then one fused multiply-add per batch row with that row's `x[r][c]` broadcast to all 16 lanes. `acc` is an array of `R` registers, `R` a const generic from 1 to 8, so the compiler keeps all of them in registers; `block_product` picks the instantiation with a `match` on the number of rows left.

Threads split the blocks of 16 outputs. Each block's results go to a scratch buffer laid out `[block][row][16]`, so every thread writes a contiguous slice of it, and one copy at the end puts them in the `[row][output]` order the rest of the model expects.

### 4.3 Planning a step

<!-- file: src/engine.rs -->
```rust
    fn plan(&mut self) -> Vec<(usize, Option<Range<usize>>)> {
        let mut budget = self.config.max_step_tokens;
        let mut plan = Vec::with_capacity(self.active.len());
        for (i, a) in self.active.iter_mut().enumerate() {
            if a.finish.is_none() && a.job.events.is_closed() {
                a.finish = Some(FinishReason::Stopped);
            }
            if a.finish.is_none() && a.decoding() {
                plan.push((i, None));
                budget -= 1;
            }
        }
        for (i, a) in self.active.iter().enumerate() {
            if a.finish.is_none() && !a.decoding() && budget > 0 {
                let n = (a.job.request.prompt.len() - a.prefilled).min(budget);
                plan.push((i, Some(a.prefilled..a.prefilled + n)));
                budget -= n;
            }
        }
        plan
    }
```

The plan is a list of (request, what it contributes): `None` for one decode token, `Some(range)` for a chunk of its prompt. It also notices clients that left (their channel is closed) before spending a step on them. `budget -= 1` cannot underflow because `spawn` checks that `max_step_tokens >= max_batch`.

### 4.4 Running the step

`step` turns the plan into `BatchSeq`s. Each needs a `&mut KvCache` to its own slot, several at once from the same `Vec`:

<!-- file: src/engine.rs -->
```rust
        let mut slots: Vec<Option<&mut KvCache>> = caches.iter_mut().map(Some).collect();
```

`iter_mut` gives a `&mut` to every cache at once (the borrow checker knows they are different elements); wrapping each in `Option` lets `slots[a.slot].take()` move each one out exactly once, and the `expect` documents that no two requests share a slot. After `forward_batch`, each request with logits samples its next token with its own `Sampler`, sends it, and is marked finished on a stop token, a failed send (the client left) or its token limit. `retire` then sends each finished request's `Done` and returns its slot.

## 5. Run it

```bash
cargo test -p ch23-continuous-batching
cargo run --release -p ch23-continuous-batching                  # all four parts, about 3 minutes
cargo run --release -p ch23-continuous-batching -- arrivals      # or: steps, serving, invariance
```

The tests check that a batch mixing prefill chunks and decode tokens computes what each sequence computes alone (chapter 14's forward pass); that five requests through three slots, with prompts prefilled in chunks of 16 tokens, each get a greedy answer (every token checked against the maximum logit of a separate forward pass over the prompt and answer); that a client leaving does not disturb the others; stop tokens; rejection of prompts that do not fit; shutdown. Chapter 20's tests check `flash_decode_many` against `flash_attention`, and `packed.rs` checks the packed kernel against plain dot products.

## 6. The Rust behind it

**Many `&mut` into one `Vec`.** `caches.iter_mut()` yields a `&mut KvCache` for every element at once, because the iterator guarantees they do not overlap. Indexing (`&mut caches[i]` twice) would not compile. `Vec<Option<&mut T>>` plus `take()` then hands them out in any order, each once.

**Borrowing fields of `self` separately.** `step` starts with `let Self { model, caches, active, scratch, attention, .. } = self;`. Afterwards `active` can be read while `caches` and `scratch` are borrowed mutably: the borrow checker tracks struct fields separately, but only when you access them as fields, not through `&mut self` methods.

**Const generics for register blocking.** `rows::<R>` has an accumulator array of `R` vector registers. With `R` a compile-time constant, the compiler unrolls the inner loop and keeps every accumulator in a register; with a run-time length, it would keep them in memory. `match (m - r0).min(8) { 1 => rows::<1>(...), ... }` turns the run-time count into one of eight compiled versions.

**`remove`, not `swap_remove`.** `retire` removes finished requests with `Vec::remove`, which keeps the order of the rest. `swap_remove` is O(1) but moves the last request into the gap, and the order of `active` decides whose prompt is prefilled first.

**`std::thread::scope` for clients.** The demo's clients are threads that borrow the engine handle and the prompts from the enclosing function. Scoped threads may borrow local data because the scope guarantees they finish before the function returns.

## 7. Mistakes you will make

- **Using the row index as the position.** Row `r` of the stacked matrix belongs to some sequence at some position; RoPE needs that position. The model still runs, and quietly produces worse answers.
- **A batched kernel that is secretly per row.** Throughput then stops improving at a small batch, as with chapter 17's kernel at `B = 2`. Measure step time against `B` before assuming the kernel batches.
- **Forgetting that the KV cache does not batch.** Throughput estimates from weights alone are too optimistic at long contexts.
- **Letting a long prompt take the whole step.** Every decoding request then stalls for the whole prefill; a token budget per step, with decodes first, bounds the stall.
- **Allocating per call inside the step.** A buffer allocated and zeroed per sequence per layer cost more than the attention itself in the first version (section 3.3).
- **Assuming batching is numerically neutral.** It is only if every kernel is batch-invariant; many are not.

## 8. How the professionals do it

- **Orca** (Yu et al., OSDI 2022) introduced iteration-level scheduling. **vLLM**, **SGLang**, **TGI**, **TensorRT-LLM** ("in-flight batching") and **llama.cpp's server** (`--parallel` slots, one combined batch per step) all batch continuously.
- On GPUs, batches of hundreds of sequences are normal, and the weights are shared by all of them in each step. The KV cache is what limits the batch size: chapter 24's paged attention was invented to fit more sequences into GPU memory.
- Weight packing is standard in CPU inference: oneDNN reorders weights for its kernels, and llama.cpp "repacks" quantized weights at load time into interleaved layouts for AVX-512, ARM and AMX kernels.
- Batch invariance became a topic of its own in 2025 (He et al., "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab): with batch-invariant kernels for matrix products, attention and normalization, a request's output no longer depends on server load.

## 9. Exercises

1. Run part 1 with 1,024 tokens of context instead of 256. Predict "the rest" per sequence from the bytes of cache it reads, then measure. At what `B` does attention take more time than the matrix products?
2. Write a static batching engine: wait until `max_batch` requests have arrived (or 100 ms have passed), run them all to completion, repeat. Run part 3 against it.
3. Make the packed kernel handle 16 rows at a time: 2 × 8 accumulators, or two blocks of 16 outputs for 8 rows. Measure `B = 16` and `B = 64`.
4. Make `flash_decode_many` batch-invariant: choose the number of key parts from the sequence's own length only, never from the batch. What does it cost when the batch is small?
5. Set `max_step_tokens` to 64 and repeat part 2. What happens to the first token of each client, and to the total time? (Chapter 25 answers this in detail.)

## 10. Check yourself

1. Why does decoding `B` sequences in one step cost much less than `B` steps of one sequence?
2. Which parts of a layer are shared by the batch, and which are per sequence?
3. How many bytes of KV cache does one SmolLM2 sequence read per step at 256 tokens of context? At 2,048?
4. Why did chapter 17's kernel gain so little at `B = 2`?
5. What does the packed layout change about how a weight is used?
6. In part 2, why did client 0's first token come later when batched?
7. What makes a kernel batch-invariant?

## 11. Recap

- A decode step is memory-bound: its weights are read once whether it serves 1 sequence or many, so batching raises throughput until arithmetic or the KV cache becomes the limit.
- `forward_batch` stacks every sequence's tokens; only RoPE positions, cache writes and attention are per sequence.
- Chapter 17's prefill kernel batched poorly (3.8x at best, 1.45 times the cost for 2 sequences); packed weights with broadcast activations and no horizontal sums brought 2 sequences to 1.13 times the cost of one, and 4.4x throughput at 64.
- Each sequence's KV cache is read by itself alone: 11.8 MB per step at 256 tokens. It grows with `B` and with context, and becomes the limit.
- The batching engine plans each step (decodes first, then prompt chunks within a token budget), so requests join and leave at every step: eight simultaneous clients got 3 times the throughput; clients arriving mid-run waited at most one step.
- Batching costs each client per-token speed, and can delay a first token when many prompts arrive together.
- The packed kernel is batch-invariant; chapter 17's is not.

## Answers

**Exercises**

1. At 1,024 tokens a sequence reads 1,024 × 46 KB = 47 MB of cache per step, four times as much as at 256. If reading the cache dominates, the roughly 1.2 ms per sequence measured at 256 tokens becomes about 4.8 ms. The matrix products take about 12 ms at `B = 2` and 15 ms at `B = 4`, so attention would take longer than them from about `B = 3` on. That is an estimate; the measurement will show how far this machine's bandwidth for many parallel reads falls short of chapter 4's streaming figure.
2. With arrivals 150 ms apart and a 100 ms timeout, every batch holds a single client (the next one arrives after the timeout), so static batching behaves like one at a time plus 100 ms of waiting. With a longer timeout, say 1 s, the first batch holds all eight clients, but client 0 waits a full second before it starts, and a client arriving while a batch runs waits for the whole batch to finish.
3. With two blocks of outputs per call, each broadcast of `x[r][c]` is used twice, halving the broadcasts per multiply-add. Expect a gain at large `B`, where the products are compute-bound, and none at `B = 1`, which is memory-bound.
4. Compute `splits` from `input.start + 1` alone (for example, one part per 256 keys), independent of how many sequences are in the call. Small batches then may not have enough tasks to keep every thread busy (one sequence with a short context gives 3 tasks for 4 threads), which is the cost chapter 20 measured; long contexts are unaffected.
5. The eight prompts no longer fit in one step, so they are prefilled over several steps, 64 tokens at a time, oldest first. The first clients get their first token sooner and the last ones later; since decoding requests take their tokens from the budget first, prompts are processed more slowly once some requests are decoding. The total time grows a little, because smaller steps use the weights less efficiently.

**Check yourself**

1. Reading the weights dominates a decode step, and a batched step reads them once for all `B` sequences. Until the arithmetic becomes the limit (the ridge point), the extra sequences only add FLOPs the processor had idle.
2. Shared: all matrix products, norms, activations, residual additions, the LM head. Per sequence: RoPE positions, storing keys and values in its cache, attention over its cache.
3. 46 KB per position (30 × 3 × 64 × 2 × 4 bytes): 11.8 MB at 256 tokens, 94 MB at 2,048.
4. With 2 rows, there is no full group of four, so the kernel fell back to one dot product per weight row and batch row: every weight row was read and converted twice, with a horizontal sum for each output.
5. Each weight is loaded and converted once and multiplied by up to 8 batch rows' activations, each accumulating into its own register of 16 outputs; outputs never need a horizontal sum.
6. All eight prompts fit in one step, so the first step prefilled all of them (about 280 tokens) before anyone got a token. Alone, client 0's prompt (35 tokens) was prefilled by itself.
7. Computing each request's results in the same order of operations whatever else is in the batch: no dependence of the summation order (or the choice of code path) on the batch size or on the request's position in it.

## Further reading

- Yu, Jeong, Kim, Kim, Chun, "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023 (sections on batching).
- He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025.
- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication", 2008 (packing and register blocking).
- Next: [Chapter 24: Paged KV cache](../24-paged-kv-cache/README.md). Memory for the cache in small blocks, so that more requests fit.
