# Chapter 24: Paged KV cache

> **In one sentence:** instead of reserving a full-length KV cache for every request, hand out memory in small blocks as each sequence grows and take it back when it ends; a table per sequence says which blocks hold its positions, attention reads them block by block, and once blocks are shared objects with reference counts, requests that start with the same tokens can share the blocks of that common beginning instead of computing them again.

**Where this fits:** chapter 23's engine batches requests, but gives each one a slot sized for the longest possible request. This chapter makes memory follow actual use, so more requests fit in the same memory and the batch can grow. Chapter 25 then decides what to do with the room: which requests to admit, and in what order to spend each step.

**You need:** chapter 14 (the KV cache layout), chapter 20 (tiles and the online softmax), chapter 23 (the batched forward pass and the batching engine).

**You will build:** a block pool with reference counts and a block table per sequence; paged attention built on chapter 20's tile kernel; chapter 23's forward pass over the paged cache; an engine that admits requests by memory, grows them a block at a time and preempts when memory runs out; and a prefix cache that finds earlier requests' blocks by a chain of hashes. Then measurements: the same memory as slots or as blocks, what small blocks cost attention, a shared system prompt, and a workload that does not fit.

---

## 1. The intuition

A notebook shared by many writers. Chapter 23 gave every writer 40 consecutive pages up front, because some might write 40 pages; most write 5, and the notebook is full of blank reserved pages while new writers wait for one.

Now every writer gets one page at a time, wherever a free page is, and keeps a small index card listing their pages in order. Nothing is reserved that is not written on, a finished writer's pages go straight back to the pile, and the notebook holds many more writers.

The index cards allow something else. Suppose ten writers all begin by copying the same two pages of instructions. With index cards, they can all list the *same* two pages: written once, read by everyone. That is prefix caching.

**Where the analogy breaks:** a shared page can only be shared if everything before it is identical too. The keys and values at a position depend on every token up to that position, so two requests share a block only if they agree on all tokens from the very beginning to the end of that block.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Block** | A fixed number of consecutive positions (here 16) of keys and values, for every layer and KV head. The unit of allocation. |
| **Block pool** | All blocks, allocated once, with a free list. |
| **Block table** | For one sequence, the list of blocks holding positions `0..len`, in order. |
| **Internal fragmentation** | Memory reserved for a request but never used by it. |
| **Reference count** | How many owners (sequences, the prefix cache) a block has; it is free when the count reaches 0. |
| **Preemption** | Taking a running request's memory away so that others can continue; here it later recomputes what it lost. |
| **Prefix caching** | Reusing the blocks of earlier requests that started with the same tokens. |
| **Copy-on-write** | Sharing a block until someone needs to change it, then giving the writer its own copy. |

## 3. The concepts in depth

### 3.1 What chapter 23 wastes

One position of SmolLM2-135M's cache holds keys and values for 30 layers × 3 KV heads × 64 dimensions, in `f32`: 30 × 3 × 64 × 2 × 4 = 46,080 bytes. A slot of 1,024 positions is 45 MB, whatever the request uses. A request with a 60-token prompt and a 100-token answer uses 16% of it; the other 84% is reserved and idle. That is internal fragmentation, and the number of requests that can run at once is set by the reserved memory, not the used memory.

### 3.2 Blocks and block tables

The fix is the one operating systems use for processes: virtual memory in pages. The KV memory is one pool of blocks. A sequence's block table maps its positions to blocks:

```text
position p  →  block table[p / 16], index p % 16 within it

sequence A (37 positions):  [ 7 | 2 | 9 ]      blocks 7 and 2 full, 9 holds 5
sequence B (16 positions):  [ 4 ]
free list:                  [ 0 1 3 5 6 8 10 ... ]
```

A sequence takes a block from the free list when it crosses a block boundary, and gives all of them back when it finishes. The waste is at most one partly filled block per sequence (15 positions here), instead of everything up to the maximum context.

Within a block, keys are stored `[layer][kv_head][position][dim]`, so the 16 keys of one (layer, head) are contiguous: a small version of chapter 20's tiles. Attention walks the table and processes one block per tile; the blocks themselves can be anywhere in the pool.

### 3.3 Measured: the same memory, as slots or as blocks

Part 1 gives both engines 90 MB of KV cache (2,048 positions): chapter 23's as 2 slots of 1,024 positions, this chapter's as 128 blocks of 16. Sixteen short requests (summarize 10 to 40 words of the novel, in 64 to 124 tokens: 2,609 positions in all with the chat template, more than the budget) are submitted at once (SmolLM2-135M, packed weights from chapter 23, 4 threads, the machine of chapter 17):

```text
== 1. 16 requests needing 2609 positions in all; 90 MB of KV cache (2048 positions)
   46080 bytes per position: 2 slots of 1024 positions, or 128 blocks of 16
   2 slots   : 1486 tokens in 13507 ms, 110 tokens/s; first token: median 6126 ms, slowest 12269 ms
   128 blocks: 1486 tokens in 8673 ms, 171 tokens/s; first token: median 2319 ms, slowest 2654 ms
      up to 16 requests in a step, up to 128 of 128 blocks in use, 3 preemptions
```

With slots, two requests run and fourteen wait; the last one starts after 12 s. With blocks, all sixteen fit (almost: memory ran out three times, section 3.5), batches are 8 times larger, throughput is 1.55 times higher, and the slowest first token arrives after 2.7 s instead of 12.3 s.

This works here because the budget is small. With a large budget and long prompts, this CPU runs out of arithmetic before memory (chapter 23: throughput stops growing near 16 sequences), and paging then buys shorter waits rather than throughput. On a GPU, which can batch hundreds of sequences before arithmetic becomes the limit, KV memory is usually what limits the batch: the paper that introduced paged attention reported 2 to 4 times the throughput of earlier systems.

### 3.4 What small blocks cost attention

Part 2 times one decode step for 8 sequences with 1,024 tokens of context each, with chapter 23's contiguous caches and with blocks of three sizes:

```text
== 2. one decode step, 8 sequences with 1,024 tokens of context each
   contiguous caches:      41.8 ms
   blocks of  16 positions: 51.7 ms
   blocks of  64 positions: 41.1 ms
   blocks of 256 positions: 42.4 ms
```

With 64-position blocks, paging is free. With 16, the step is 24% slower: every block is a separate tile, and each tile has a fixed cost (a call into chapter 20's kernel, a rescale of the running sums, a pass to find the tile's maximum) that is spread over only 16 keys instead of 64. Smaller blocks waste less memory (at most 15 positions per sequence, against 63), and prefix caching shares whole blocks only, so smaller blocks share more. The engine uses 16, as vLLM does; exercise 1 measures 64.

### 3.5 When memory runs out

Admission now asks for memory, not a slot: a request is admitted if the pool has blocks for its prompt and first token, plus a small reserve (1% of the blocks) for the running requests to grow into. But requests grow a block at a time, and the engine cannot know how long each answer will be, so the pool can run out while every running request still wants to continue. There are two ways out:

- **Swapping:** copy a request's blocks to CPU memory (on a GPU) or to disk, and copy them back later.
- **Recomputation:** throw the request's blocks away and recompute its keys and values from its tokens when it resumes. Prefill is fast, compared with generating the same tokens one at a time.

This engine recomputes, as vLLM does by default. The request preempted is the one admitted most recently (it has done the least work), and it goes back to the front of the queue with everything else it had: its tokens so far (prompt and generated), its sampler with its random state and penalty counts. When readmitted, it prefills all of its tokens, and the last chunk's logits give its next token, exactly as if nothing had happened.

Part 4 runs eight requests needing about 2,100 positions with room for 800, then with room for 4,096:

```text
== 4. eight requests needing about 2,100 positions, with room for 800 or for 4,096
    50 blocks of 16: 1600 tokens in 13185 ms, 12 preemptions, 1444 positions recomputed, up to 8 requests per step
   256 blocks of 16: 1600 tokens in 7085 ms, 0 preemptions, 0 positions recomputed, up to 8 requests per step
   the answers are identical
```

Twelve preemptions and 1,444 recomputed positions make the run 1.9 times slower, but it completes, with the same answers. Preemption is a safety valve, not a way of working: if it happens often, admit fewer requests.

### 3.6 Prefix caching

Requests often share their beginning: the chat template, a long system prompt, the documents a retrieval system puts in every prompt, the earlier turns of a conversation. The keys and values of a position depend only on the tokens up to it, so the blocks computed for one request are valid for every other request that starts with the same tokens.

Blocks are found by a chain of hashes. Block `i` of a sequence is keyed by `hash(key of block i − 1, tokens of block i)`, so equal keys mean equal tokens all the way from the start. A new request hashes its full prompt blocks in order and takes the longest run found in the cache. Only full blocks are cached; they are never written again, so sharing them needs no copying. And the last prompt token is always computed, even if its block is cached, because its logits give the first new token.

The cache holds its own reference to each block. A block that no running request uses, only the cache, stays in memory until the pool needs it; then the least recently used ones are evicted, the end of a chain before its beginning (a block is only reachable through its parents).

Part 3 sends eight requests with the same 576-token system prompt (an instruction and the novel's first 450 words) and different questions, 300 ms apart:

```text
== 3. eight requests sharing a 576-token system prompt, arriving 300 ms apart
   prefix caching off: first token after 2881 ms, 3896 ms, 4661 ms, 5660 ms, 6581 ms, 9033 ms, 9959 ms, 10276 ms
      0 of 4771 prompt tokens taken from the cache; all done after 13489 ms
   prefix caching on : first token after 2568 ms, 2266 ms, 1966 ms, 1668 ms, 1368 ms, 1324 ms, 1027 ms, 726 ms
      3776 of 4771 prompt tokens taken from the cache; all done after 4037 ms
```

Without the cache, each request recomputes 600 tokens of prompt, the prefills pile up, and the first tokens come later and later. With it, 79% of the prompt tokens come from the cache, and the whole run takes 4.0 s instead of 13.5 s. The first request pays for everyone. The next few still wait for part of it (its blocks are registered as its prefill fills them, 32 blocks per step); the last two get their first token after about a second or less.

### 3.7 Sharing and copy-on-write

Reference counts make sharing safe to free: a block returns to the free list only when its last owner releases it. Writing is safe here because shared blocks are full, and only the partly filled last block of a sequence is ever written. Other features share blocks that are not full: sampling several answers to one prompt (`n > 1`), or beam search. Their sequences share the prompt's last partial block, and the first one to write its next position must get a copy first: copy-on-write, exercise 3.

## 4. The code

The pool and tables are in [`src/blocks.rs`](src/blocks.rs), the prefix cache in [`src/prefix.rs`](src/prefix.rs), paged attention in [`src/attention.rs`](src/attention.rs), the forward pass in [`src/forward.rs`](src/forward.rs) (chapter 23's, adapted), the engine in [`src/engine.rs`](src/engine.rs), the demo in [`src/main.rs`](src/main.rs).

### 4.1 Growing a table

<!-- file: src/blocks.rs -->
```rust
    pub fn reserve(&mut self, pool: &mut BlockPool, len: usize) -> bool {
        while self.capacity(pool.block_size()) < len {
            match pool.allocate() {
                Some(b) => self.blocks.push(b),
                None => return false,
            }
        }
        true
    }
```

A block is a `u32` index into the pool: no allocation per block, no pointers, and a table is just a `Vec<u32>`. `allocate` pops the free list and sets the block's reference count to 1; `retain` and `release` count owners; the last `release` pushes the block back on the free list. Every path that ends a request (finished, cancelled, preempted) ends with `table.release(pool)`, and the engine's tests check that memory never goes above the pool.

### 4.2 Attention, block by block

<!-- file: src/attention.rs -->
```rust
            for b in (part * per_part).min(blocks)..((part + 1) * per_part).min(blocks) {
                let n = (keys - b * bs).min(bs);
                let block = input.table.blocks[b];
                let k = kv.keys(block, layer, kv_head, n);
                let v = kv.values(block, layer, kv_head, n);
                for (g, s) in state.chunks_exact_mut(stride).enumerate() {
                    let head = kv_head * group + g;
                    attend_tile(&input.q[head * d..(head + 1) * d], k, v, s, opts.simd);
                }
            }
```

This is chapter 20's decode task with the table in the middle: the task's key range is now a range of blocks (`part` of `splits`), each block is a tile (the last one partial, `n < bs`), and `attend_tile` is chapter 20's AVX-512 kernel, made public for this chapter. The states and the final merge are chapter 20's too. For a prompt chunk, `paged_chunk` does the same per block of query tokens, with the causal limit per token.

### 4.3 Finding a cached prefix

<!-- file: src/prefix.rs -->
```rust
        for block_tokens in tokens.chunks_exact(pool.block_size()).take(max_blocks) {
            let key = block_key(parent, block_tokens);
            let Some(e) = self.entries.get_mut(&key) else {
                break;
            };
            if e.parent != parent || e.tokens != block_tokens {
                break; // a hash collision: treat as a miss
            }
            e.last_used = self.clock;
            pool.retain(e.block);
            found.push(e.block);
            parent = key;
        }
```

The first miss ends the search: a later block cannot be valid if an earlier one differs. Each entry keeps its tokens and its parent's key, so a collision of 64-bit hashes (unlikely, but not impossible across millions of blocks) causes a miss, never someone else's keys and values. The engine calls it with `max_blocks = (len − 1) / block_size`, which keeps the block containing the last prompt token out of reach.

### 4.4 Preempting to make room

<!-- file: src/engine.rs -->
```rust
            // Room for one more position, preempting others if necessary.
            loop {
                let (kv, a) = (&mut self.kv, &mut self.active[i]);
                if a.table.reserve(kv, a.table.len + 1) {
                    break;
                }
                if self.prefix.evict(&mut self.kv, 1) > 0 {
                    continue;
                }
                let newest = (0..self.active.len())
                    .rev()
                    .find(|&j| self.active[j].finish.is_none() && !self.active[j].preempted)
                    .expect("request i itself is running");
                self.preempt(newest);
                if newest == i {
                    break;
                }
            }
```

For each decoding request, oldest first: try to reserve its next position; if the pool is empty, evict an idle block from the prefix cache; if there is none, preempt the newest running request, possibly this one. The loop ends because each round either succeeds or frees memory, and a request that preempts itself stops asking. Preempted requests are removed from the batch after planning and put back at the front of the queue in their original order.

The engine represents every request the same way: its `tokens` (prompt, then everything generated) and a table holding the keys and values of the first `table.len` of them. What a request contributes to a step is always `tokens[table.len..]`, capped by the step's budget: the whole remaining prompt, a chunk of it, one token when decoding, or everything again after preemption. There is no separate "prefill" and "decode" state to keep consistent.

## 5. Run it

```bash
cargo test -p ch24-paged-kv-cache
cargo run --release -p ch24-paged-kv-cache                    # all four parts, about 2 minutes
cargo run --release -p ch24-paged-kv-cache -- prefix          # or: memory, attention, preemption
```

The tests check the pool's reference counting; that a position lands in the right block; the prefix cache's lookups and eviction order; that the paged forward pass computes what chapter 14's does for block sizes 1, 4, 5 and 16, for prefill and decode; that six requests squeezed into 40 blocks of 4 are preempted (the test fails if none is) and still produce greedy answers; that a shared prefix is taken from the cache (exact token counts); rejection and shutdown.

## 6. The Rust behind it

**Indices instead of pointers.** A block is a `u32`; the pool owns all the memory in two `Vec<f32>`s. `Rc<Block>` would count references for us, but every block would be a separate allocation, and a table of `Rc`s could not be handed to the thread pool (`Rc` is not `Send`). Counting by hand in a `Vec<u32>` is a few lines, and the tests check it.

**`HashMap::entry` with `or_insert_with`.** `insert` retains a block for the cache only when the entry is new: the closure passed to `or_insert_with` runs only if the key is absent. Existing entries are left as they are.

**`std::cmp::Reverse` in a sort key.** Eviction sorts `(last_used, Reverse(depth), key)`: oldest first, and among equally old ones, deepest first. `Reverse` flips the order of one component of a tuple without a custom comparison function.

**`DefaultHasher` is deterministic.** `DefaultHasher::new()` uses fixed keys, so the same tokens always give the same key within and across runs (unlike a `HashMap`'s `RandomState`). That is what a cache key needs; the stored tokens protect against collisions.

**A copy, adapted.** `forward.rs` is chapter 23's forward pass with the cache replaced. A trait over "some kind of KV storage" could share the code, but it would make chapter 23 harder to read for a benefit to this chapter only. The file's header lists the three changes.

## 7. Mistakes you will make

- **Leaking blocks.** Every way a request can end (finished, stopped, cancelled, rejected after lookup, preempted) must release its table. A leak shows up as a pool that slowly fills.
- **Caching blocks that are not full,** or blocks keyed by their own tokens only. The first is overwritten by the next token; the second matches the same text at a different position, where its keys and values are wrong.
- **Taking the whole prompt from the cache.** Then there is nothing to compute, and no logits for the first new token.
- **Evicting a parent before its children.** The children stay in memory but can never be found.
- **Admitting everything that fits the prompt.** Every request then grows into the same last free blocks, and the engine spends its time preempting and recomputing.
- **Choosing the block size by habit.** 16 is the GPU convention; on this CPU, 64 makes paged attention free (section 3.4).

## 8. How the professionals do it

- **vLLM** introduced paged attention (Kwon et al., SOSP 2023): 16-token blocks by default, block tables, preemption by recomputation or swapping to CPU memory, and copy-on-write for parallel sampling and beam search. Its automatic prefix caching keys each full block by a hash of its tokens and its parent's hash, as here, and is on by default in vLLM V1.
- **SGLang** keeps cached prefixes in a radix tree of token sequences (RadixAttention), which finds the longest shared prefix directly and shares at token rather than block granularity.
- **TensorRT-LLM** and **TGI** also use paged KV caches, and API providers offer "prompt caching" (Anthropic, OpenAI, Google), usually billed at a discount, which rests on the same idea.
- Storing the cache in 8-bit floating point (FP8) halves its memory again, doubling the number of sequences that fit; vLLM and TensorRT-LLM support it.

## 9. Exercises

1. Run part 1 with blocks of 64 positions. Where does the gain in attention speed (section 3.4) go, and what does the larger waste per sequence cost with a budget of only 2,048 positions?
2. Implement swapping instead of recomputation: when preempting, copy the request's blocks into a `Vec<f32>` and release them; when resuming, allocate blocks and copy them back. Compare with part 4. (On a CPU, "swap" memory is the same memory; what would change on a GPU?)
3. Support `n = 2`: two answers to one prompt, sharing the prompt's blocks. What must happen when both write their first token into the prompt's last, partly filled block?
4. Store keys and values as `bf16` (chapter 2): half the bytes per position. What changes in `BlockPool` and in `attend_tile`? Measure part 1 and the model's answers.
5. A chat client sends the whole conversation with every turn. Explain why prefix caching makes the tenth turn cheap, and what a server must do to keep that true when many conversations are active at once.

## 10. Check yourself

1. How much KV memory does one position of SmolLM2-135M take in `f32`, and how much does chapter 23 reserve per request with a 1,024-token context?
2. What is the most memory paging can waste per sequence?
3. Why do 16-position blocks slow attention down on this CPU, and what do they gain?
4. What happens to a preempted request's generated tokens, its sampler, and its blocks?
5. Why is block `i` keyed by a hash of block `i − 1`'s key and its own tokens, rather than by its own tokens alone?
6. Why is the block holding the last prompt token never taken from the cache?
7. Why can shared blocks be shared without copy-on-write in this engine?

## 11. Recap

- One position of SmolLM2's cache is 46 KB in `f32`; reserving the full context per request wastes most of it.
- A block pool with a free list and reference counts, and a block table per sequence, waste at most one partial block per sequence.
- In the same 90 MB, blocks ran 16 requests at once instead of 2: 1.55 times the throughput, and the slowest first token after 2.7 s instead of 12.3 s.
- Paged attention reads keys block by block with chapter 20's kernel: 24% slower with 16-position blocks, no slower with 64.
- When memory runs out, the newest request is preempted and later recomputes its keys and values: 12 preemptions made a run 1.9 times slower with identical answers.
- Prefix caching shares full blocks between requests with the same beginning, found by chained hashes: with a 576-token shared system prompt, 79% of prompt tokens came from the cache and the run took 4.0 s instead of 13.5 s.

## Answers

**Exercises**

1. Attention becomes as fast as with contiguous caches (section 3.4). But each sequence wastes up to 63 positions instead of 15: with 16 requests of about 160 positions, that is up to 16 × 63 ≈ 1,000 positions, half of the 2,048-position budget, so fewer requests fit, more are preempted, or both. With this small budget the smaller block is likely better overall; with a large budget, the larger one. Measure both.
2. Swapping copies each block twice instead of recomputing it; on a CPU that is a memory copy of 737 KB per block, much cheaper than recomputing 16 positions through the model, so part 4 gets faster. On a GPU, the copy crosses PCIe to CPU memory (tens of GB/s, against thousands on the GPU), and recomputation is often faster, which is why vLLM recomputes by default.
3. Both sequences' tables list the same blocks, each with a reference count of 2. Before writing position `len` into the shared last block, a sequence checks its count: if above 1, it allocates a new block, copies the used positions into it, releases its share of the old one and writes into the copy. The full blocks before it stay shared.
4. `BlockPool` stores `Bf16` instead of `f32` (`store` converts, `keys`/`values` return `&[Bf16]`), and `attend_tile` needs a kernel that widens `bf16` to `f32` as it loads (chapter 6), for keys in the dot products and for values in the weighted sum. Twice the positions fit in the same memory, and attention reads half the bytes; the answers change slightly because keys and values are rounded (compare them as chapter 18 compared int8 weights).
5. Turn 10's prompt begins with turns 1 to 9, whose blocks were computed and cached during turn 9 (prompt and answer). Only the new message and the template around it are computed. With many conversations, the server must keep those blocks long enough: evicting by least recent use keeps active conversations, but a crowded server evicts before users reply, and routing all turns of a conversation to the same server matters when there are several.

**Check yourself**

1. 30 layers × 3 KV heads × 64 dimensions × 2 (keys and values) × 4 bytes = 46,080 bytes. With 1,024 positions per slot: 45 MB per request, whatever it uses.
2. One block minus one position (15 positions, 691 KB, with 16-position blocks), in the sequence's last block.
3. Each block is one tile, and each tile has fixed costs (a kernel call, a rescale of the running sums, a maximum) spread over only 16 keys. In exchange, less memory is wasted per sequence and prefix caching can share at a finer granularity.
4. The tokens stay in the request (they are part of its token list), and so does its sampler, with its random state. Its blocks are released (to the free list, or down to the prefix cache's reference). On readmission it prefills all its tokens again and continues from its next token.
5. A block's keys and values depend on every token before it, not only its own. Chaining makes the key depend on all tokens from the start to the end of the block, so equal keys mean the whole prefix is equal.
6. The first generated token is sampled from the logits of the last prompt token, which only a forward pass over that token produces. So at least that token must be computed, and its block cannot be taken whole from the cache.
7. Only full blocks are shared, and a full block is never written again: new tokens go into a sequence's own last block.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", 2024 (RadixAttention).
- The vLLM documentation, "Automatic Prefix Caching" (design notes).
- Denning, "The Working Set Model for Program Behavior", 1968, and any operating systems text on paging: the same trade-offs in their original setting.
- Next: [Chapter 25: Scheduling and SLOs](../25-scheduling/README.md). Deciding who runs, and measuring whether users are served well.
