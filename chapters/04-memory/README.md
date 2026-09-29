# Chapter 4: Memory is the bottleneck

> **In one sentence:** a processor can do arithmetic far faster than memory can feed it numbers, so the speed of most inference code is set by how many bytes it moves and from where, and the roofline model tells you which of the two limits a kernel will hit.

**Where this fits:** chapters 1-3 kept running into the same effect: the same arithmetic runs at very different speeds depending on memory access. This chapter measures the machine directly (cache sizes, bandwidths, latencies, arithmetic peak) and turns those numbers into a model that predicts the speed of any kernel before you write it. Every optimization in the rest of the course is justified with this model.

**You need:** chapters 1-3.

**You will build:** probes for read and write bandwidth at every cache level, load latency by pointer chasing, the cost of strided access, the cost of touching fresh memory, multi-core bandwidth, arithmetic throughput, and a roofline calculator fed with the measured numbers.

---

## 1. The intuition

Picture a chef (the processor) who can chop incredibly fast, and ingredients stored at different distances:

- A few items right on the cutting board (**registers**): instant.
- A small fridge under the counter (**L1 cache**, 48 KB here): a second's reach.
- A bigger fridge across the kitchen (**L2 cache**, 2 MB per core here): a few steps.
- A walk-in cold room shared by all the chefs (**L3 cache**, tens of MB): a short walk.
- A warehouse down the street (**main memory / DRAM**, gigabytes): a trip that takes as long as chopping a few hundred ingredients.

Two rules make this kitchen work:

1. **Fetching brings a whole crate, not one item.** Ask for one tomato and you get the crate of 16 next to it (a **64-byte cache line**). If your recipe uses the neighbouring tomatoes next, they are already on the counter.
2. **Assistants run ahead.** If you walk down the warehouse aisle in order, assistants notice and start bringing the next crates before you ask (the **prefetcher**). If you jump around at random, they cannot guess, and every trip is a full round trip.

For an LLM generating text, the recipe is *read every weight once per token*: gigabytes of ingredients, each used in one quick chop. The chef spends nearly all the time waiting for deliveries. Making inference fast is mostly about arranging deliveries.

**Where the analogy breaks:** a real kitchen has one delivery at a time. A modern CPU core can have a dozen or more cache-line fetches in flight at once, and many cores can fetch in parallel. So "how long one delivery takes" (latency) and "how many crates arrive per second" (bandwidth) are very different numbers, and this chapter measures both.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Cache** | Small, fast memory on the processor chip that holds recently used data. |
| **L1 / L2 / L3** | Cache levels, from smallest and fastest (L1, per core) to largest and slowest (L3, shared). |
| **DRAM / main memory** | The RAM sticks (or HBM on a GPU). Large and comparatively slow. |
| **Cache line** | The unit of transfer between memory levels: 64 bytes on x86 and most ARM chips. |
| **Working set** | The amount of memory a loop touches repeatedly. Whether it fits in a cache decides its speed. |
| **Latency** | Time from asking for a piece of data until it arrives. |
| **Bandwidth** | Bytes delivered per second when requests are kept flowing. |
| **Prefetcher** | Hardware that detects access patterns and fetches lines before they are asked for. |
| **Page** | The unit the operating system maps memory in: 4 KB normally, 2 MB or 1 GB for "huge pages". |
| **TLB** | A cache of address translations (virtual page to physical page). Missing it adds latency. |
| **Page fault** | The first touch of a page the OS has promised but not yet provided. |
| **FLOP/s** | Floating-point operations per second. |
| **Arithmetic intensity** | FLOPs performed per byte moved from memory. |
| **Roofline** | A model: speed ≤ min(peak FLOP/s, intensity × bandwidth). |
| **Memory-bound / compute-bound** | Limited by bytes moved / by arithmetic performed. |

## 3. The concepts in depth

### 3.1 The memory wall

Over the last few decades, processor arithmetic got faster much more quickly than memory did. A single core on the reference machine can do about 30-60 billion floating-point operations per second, while it can read only about 11 billion bytes per second from main memory. That is roughly 3-5 operations per *byte*, or 12-20 per `f32`. Any code that does less work than that per number it reads waits for memory. Caches exist to hide this gap for data that is reused. Weights in LLM decode are not reused (each is read once per token), which is why decode sits on the wrong side of this wall.

### 3.2 The hierarchy, measured

Part 1 of the demo reads buffers of increasing size over and over and reports GB/s. Part 2 measures latency. On the reference machine:

| Working set | Where it lives | Read bandwidth (1 core) | Latency of one dependent load |
|---|---|---|---|
| 16-32 KB | L1 | ~50 GB/s (our loop's limit, see below) | 1.2 ns (~3 cycles) |
| 128 KB-1 MB | L2 | ~50 GB/s | 3-5 ns |
| 2 MB | L2 edge | ~35 GB/s | |
| 4-64 MB | L3 | ~20-23 GB/s | 43-184 ns |
| 256 MB-1 GB | DRAM | ~10-12 GB/s | 230-295 ns |

Three things stand out.

**The cliffs line up with the cache sizes.** Bandwidth drops at 2 MB (L2 is 2 MB per core) and again beyond 64 MB, where data no longer fits in the share of L3 this VM gets.

**The L1 and L2 numbers are limited by our loop, not by the cache.** Our `sum_fast` loop, compiled for generic x86-64 (4-wide SSE vectors), cannot consume more than about 50 GB/s. The hardware can do much more: this core can load two 64-byte vectors per cycle from L1, over 250 GB/s. Recompiling for the real CPU (`-C target-cpu=native`, exercise 1) lifts the 32 KB number to 68 GB/s but leaves the DRAM number unchanged. That is the pattern you will see all course: *vector instructions matter for data in cache, and do nothing for data in DRAM.*

**Latency grows much more than bandwidth drops.** DRAM is about 4x lower bandwidth than L1 in this table but about 250x higher latency. Bandwidth survives because many requests overlap; latency is the full wait for one.

The DRAM latency here (~290 ns at 1 GB) is higher than the ~80-100 ns you may see quoted for desktop machines. Part of it is the memory itself, and part is address translation: with 4 KB pages, a 1 GB random walk misses the TLB constantly, and in a virtual machine each TLB miss needs a *two-level* page-table walk (guest and host). This is one reason inference servers sometimes back model weights with **huge pages** (2 MB), which need 512 times fewer TLB entries.

### 3.3 Latency versus bandwidth

The latency probe builds a random cycle through a buffer: slot `i` holds the index of the next slot to visit. Each load's address depends on the previous load's result, so the CPU cannot start load *n+1* until load *n* is done. Time per hop = latency.

The bandwidth probe reads consecutive addresses whose values do not decide where to read next. The CPU issues many loads ahead, the prefetcher streams whole regions in, and dozens of cache lines are in flight at once. Time per byte = bandwidth.

A useful formula (Little's law, which chapter 25 uses for request queues):

```text
bandwidth = bytes in flight / latency
```

With ~290 ns latency and 11 GB/s from one core, one core keeps about 11e9 × 290e-9 ≈ 3,200 bytes (about 50 cache lines) in flight. Each core has a limited number of slots for outstanding misses, which is why one core cannot use the whole memory system, and why part 5 shows bandwidth rising as cores are added.

**Where this matters in inference:** weight streaming (matvec) is a bandwidth problem. Anything that follows pointers (a linked list of KV cache blocks, a hash table lookup for prefix caching, a trie walk in a tokenizer) is a latency problem. Chapter 24's paged KV cache is designed so that the attention kernel reads whole blocks contiguously, turning a latency problem back into a bandwidth problem.

### 3.4 Cache lines and stride

Part 3 reads one float out of every *k* from a 256 MB buffer:

```text
   k (stride) | time for the whole pass | floats read
            1 |                 30.7 ms |    67108864
            4 |                 25.9 ms |    16777216
           16 |                 21.2 ms |     4194304
           64 |                 17.8 ms |     1048576
          128 |                 10.6 ms |      524288
         1024 |                  0.6 ms |       65536
```

Reading **one float in 64** takes almost as long as reading **all 64**: 17.8 ms against 30.7 ms, for 1/64 of the useful data. The reason is the cache line. Memory is delivered in 64-byte (16-float) lines, so with stride 16 every float you read drags in 15 you do not use. Up to stride 64 the time barely falls, because the prefetchers also fetch neighbouring lines (Intel cores fetch lines in adjacent pairs and stream ahead in a detected direction). Only when the stride is large enough that most lines are skipped outright (128 floats = 512 bytes and beyond) does the pass get proportionally cheaper.

The lesson for inference: **the cost of reading memory is paid per cache line, not per number.** A layout that makes a kernel use 4 bytes out of every 64 wastes 94% of the scarcest resource on the machine. This is why chapter 3's column-order sum was 6x slower, why GPU kernels care about "coalescing", and why quantized formats (chapters 18-19) store their scales next to the weights they scale.

### 3.5 First touch: memory the OS has promised but not delivered

Part 4 allocates 256 MB and writes it twice:

```text
   first write pass: 102.6ms   second write pass: 40.0ms
```

The first pass is 2.5x slower. `vec![0.0f32; n]` asks the allocator for zeroed memory, and for large sizes the allocator asks the OS, which hands back *virtual* pages that are guaranteed to read as zero but are not yet backed by physical memory. The first write to each 4 KB page traps into the kernel (a **page fault**), which finds a physical page, zeroes it, and maps it. That is 65,536 page faults for 256 MB.

This is not a benchmark curiosity. It is why:

- **Loading a model is slower than reading the file.** Every byte of a freshly allocated weight buffer faults on first write. Memory-mapping the file instead (chapter 9) moves the faults to first *read*, and lets the OS page cache share one copy between processes.
- **The first request to a new server is slow.** Its KV cache and scratch buffers are touched for the first time. Serious servers run a **warm-up** request at startup, before reporting themselves ready (chapter 30).
- **`vec![0.0; n]` is instant but `vec![1.0; n]` is not.** Allocating 256 MB of zeros took 7-16 µs on the reference machine (nothing is touched), while 256 MB of ones took 140-215 ms (everything is touched, faults and all).

### 3.6 One core cannot use the whole memory system

Part 5 reads a 1 GB buffer with 1, 2 and 4 threads. Over several runs on the reference machine:

```text
   1 thread(s):   10-12 GB/s
   2 thread(s):   22-23 GB/s
   4 thread(s):   30-45 GB/s
```

Bandwidth scales nearly linearly up to 4 cores here, because each core brings its own set of outstanding-miss slots (section 3.3). The spread at 4 threads (30-45 GB/s between runs) is the cloud VM sharing memory bandwidth with other tenants on the same physical server. On a desktop you would see a tighter range and a plateau once the memory controller saturates, often at 2-4 cores.

**Consequence:** a single-threaded matvec can never use the machine's full bandwidth. Chapter 7 splits every matvec across cores for exactly this reason: not for more arithmetic, which a matvec barely needs, but for more bandwidth.

### 3.7 How fast can a core do arithmetic?

Part 6 runs a loop of `a = a * m + c` over 128 independent accumulators that stay in registers. No memory traffic, so this measures pure arithmetic throughput. The result on the reference machine was 31-33 GFLOP/s per core when compiled for generic x86-64. That target only allows the 128-bit SSE2 instructions from 2003: 4 floats per instruction.

Compiled with `RUSTFLAGS="-C target-cpu=native"`, the same source uses this CPU's 512-bit AVX-512 registers (16 floats per instruction) and reached about 62 GFLOP/s. With **fused multiply-add** (FMA: `a * m + c` as one instruction with one rounding), it roughly doubles again. Rust never fuses a multiply and an add on its own, because fusing changes the rounding (chapter 2). You have to ask for it with `f32::mul_add` or intrinsics, which chapter 6 does.

The theoretical peak of one core here: 2.1 GHz × 2 FMA units × 16 lanes × 2 FLOPs per FMA ≈ 134 GFLOP/s. Four cores: about 540 GFLOP/s. The roofline below uses the measured baseline number (about 124-130 GFLOP/s for 4 cores), because that is what our code, compiled the default way, can actually do.

### 3.8 The roofline model

Every kernel does some number of FLOPs and moves some number of bytes. Their ratio is the kernel's **arithmetic intensity**:

```text
intensity = FLOPs / bytes moved from main memory       (FLOP/byte)
```

A kernel cannot go faster than the arithmetic units allow, and it cannot go faster than memory can deliver its bytes:

```text
attainable FLOP/s = min( peak FLOP/s,  intensity × bandwidth )
```

Plotted on log-log axes (intensity across, FLOP/s up), this is a slanted line that hits a flat ceiling: a roofline.

```text
 FLOP/s
   ▲
   │                    ┌──────────────────────── peak compute (124 GFLOP/s)
   │                  ╱ │
   │                ╱   │    compute-bound: add arithmetic units,
   │              ╱     │    use SIMD, FMA, more cores
   │            ╱       │
   │          ╱         │
   │        ╱  memory-bound: move fewer bytes
   │      ╱    (quantize, batch, fuse, cache)
   │    ╱               │
   └──────────────────────────────────────────────▶ FLOP/byte
                   ridge ≈ 4
```

The **ridge point** is where the two limits meet: peak FLOP/s ÷ bandwidth. For this machine, about 124 / 30-45 ≈ 3-4 FLOP/byte. Kernels with intensity below the ridge are memory-bound; above it, compute-bound.

**Worked examples:**

- **Vector add** `c[i] = a[i] + b[i]`: 1 FLOP, reads 8 bytes and writes 4. Intensity 1/12 ≈ 0.08. Hopelessly memory-bound on every machine ever built.
- **Linear layer, one input, `f32` weights** (decode at batch 1): each 4-byte weight is used in one multiply-add, 2 FLOPs. Intensity 0.5. Memory-bound.
- **Same with `bf16` weights**: 2 FLOPs per 2 bytes. Intensity 1.0. Twice as fast, *because it moves half the bytes*, not because anything computes faster. With int8, 2.0. With 4-bit, 4.0.
- **Batch of B inputs** (chapter 1's `predict_batch`): each weight is read once and used B times. Intensity 2B ÷ bytes per weight. With `bf16` and B = 16, intensity 16: compute-bound on this machine.
- **Large matmul** (N × N by N × N, f32): 2N³ FLOPs over 3N² × 4 bytes, intensity N/6. For N = 4,096, about 680. Deeply compute-bound. Prompt processing (prefill) looks like this.

This one table explains the whole strategy of LLM inference optimization:

| Technique | What it changes on the roofline | Chapter |
|---|---|---|
| Quantization | Fewer bytes per weight: moves the kernel right | 18-19 |
| Batching | Each weight read serves more inputs: moves right | 23 |
| Speculative decoding | Verifies several tokens per weight read: moves right | 26 |
| SIMD, FMA, threads | Raises the flat ceiling | 6-7 |
| More bandwidth (threads, better hardware) | Raises the slanted line | 7, 29 |
| FlashAttention, kernel fusion | Removes intermediate reads and writes: fewer bytes | 20 |

### 3.9 The same model for GPUs

GPUs are the same picture with bigger numbers. From vendor specification sheets (not measured here):

| Device | Memory bandwidth | Dense bf16 compute | Ridge point |
|---|---|---|---|
| This VM (4 cores, measured, baseline build) | 30-45 GB/s | ~0.12 TFLOP/s | ~3-4 |
| NVIDIA A100 80 GB (SXM) | 2.0 TB/s | 312 TFLOP/s | ~150 |
| NVIDIA H100 (SXM) | 3.35 TB/s | 989 TFLOP/s | ~295 |

An H100 needs about 295 FLOPs per byte to be compute-bound. A `bf16` linear layer delivers B FLOPs per byte at batch B. So a single user decoding on an H100 uses about 1/300 of its arithmetic, and it takes a batch of a few hundred sequences to use the GPU fully. That single fact is why GPU serving engines are built around batching (chapter 23), and why a bigger GPU does not make one user's tokens much faster unless it also has more memory bandwidth.

## 4. The code

All probes are in [`src/lib.rs`](src/lib.rs); [`src/main.rs`](src/main.rs) runs them and prints the tables.

### 4.1 Best of N

<!-- file: src/lib.rs -->
```rust
pub fn best_of(runs: usize, mut f: impl FnMut()) -> Duration {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .expect("runs must be at least 1")
}
```

For hardware probes we take the **minimum** of a few runs. Noise (another process, an interrupt, a noisy neighbour VM) can only make a run slower, so the fastest run is the closest to what the hardware can do. Chapter 1 took percentiles instead, because there the question was what users experience. Pick the statistic that answers your question, and say which one you used.

`impl FnMut()` accepts any closure that can be called repeatedly and may mutate what it captures. The closure is monomorphized: the compiler generates a copy of `best_of` specialized for each closure, so there is no function-pointer call inside the timed region.

### 4.2 Read bandwidth

<!-- file: src/lib.rs -->
```rust
pub fn read_bandwidth(bytes: usize) -> f64 {
    let data = vec![1.0f32; bytes / 4];
    // Read at least 1 GB in total per measurement so tiny buffers are timed
    // over many passes and the timer's resolution does not matter.
    let passes = (1usize << 30).div_ceil(bytes).max(1);
    black_box(sum_fast(&data)); // warm-up: pull the buffer into cache
    let best = best_of(3, || {
        for _ in 0..passes {
            black_box(sum_fast(black_box(&data)));
        }
    });
    (bytes * passes) as f64 / best.as_secs_f64() / 1e9
}
```

- `vec![1.0f32; ...]` writes every element, so all pages are faulted in before we start timing (section 3.5).
- A 16 KB buffer takes about 300 ns to read, too short to time reliably one pass at a time, so we repeat it enough times to read 1 GB in total. `div_ceil` is integer division rounding up.
- The warm-up call puts the buffer into whichever cache it fits in. The timed passes then measure that level.
- `black_box(&data)` on the input stops the compiler from noticing that we sum the same unchanged data repeatedly and computing it once. `black_box(sum_fast(...))` on the output stops it from deleting the unused sum.

### 4.3 Latency by pointer chasing

<!-- file: src/lib.rs -->
```rust
pub fn random_cycle(n: usize, seed: u64) -> Vec<u32> {
    assert!(n >= 2 && u32::try_from(n).is_ok());
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = seed.max(1);
    for i in (1..n).rev() {
        // xorshift64: a tiny, fast pseudo-random generator.
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut next = vec![0u32; n];
    for w in 0..n {
        next[order[w] as usize] = order[(w + 1) % n];
    }
    next
}
```

1. `order` starts as 0, 1, ..., n−1 and is shuffled with the Fisher-Yates algorithm: walk from the end, swap each element with a random earlier-or-same position. Every ordering is equally likely.
2. The shuffled `order` is the sequence of slots we will visit. `next[a] = b` whenever `b` comes right after `a` in that order, and the last slot links back to the first. Following `next` from anywhere visits all `n` slots once, in random order, and returns: one big cycle. A test checks that.
3. `u32` indices instead of `usize` halve the buffer size for the same number of slots, so the 1 GB probe really has 268 million slots.
4. `u32::try_from(n).is_ok()` checks that `n` fits in a `u32` before we use `n as u32`. Clippy suggested this form over a manual range comparison.

<!-- file: src/lib.rs -->
```rust
pub fn chase(next: &[u32], steps: usize) -> u32 {
    let mut i = 0u32;
    for _ in 0..steps {
        i = next[i as usize];
    }
    i
}
```

The whole probe is this loop. The next index is whatever the current load returns, so no two loads can overlap. Returning `i` (and wrapping the result in `black_box`) keeps the compiler from deleting the loop.

### 4.4 Strided reads with const generics

<!-- file: src/lib.rs -->
```rust
pub fn strided_sum<const STRIDE: usize>(data: &[f32]) -> f32 {
    let (blocks, _) = data.as_chunks::<STRIDE>();
    let (groups, rest) = blocks.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for group in groups {
        for lane in 0..8 {
            sums[lane] += group[lane][0];
        }
    }
    sums.iter().sum::<f32>() + rest.iter().map(|block| block[0]).sum::<f32>()
}
```

`const STRIDE: usize` is a **const generic**: the stride is part of the function's type, fixed at compile time. `strided_sum::<16>` and `strided_sum::<64>` are two separately compiled functions, each with its stride baked in as a constant.

Why bother? A first version of this probe took the stride as an ordinary argument. For stride 1 the compiler could not vectorize the loop (it did not know the stride would be 1), and the probe measured loop overhead at 1 ns per float instead of memory. With a compile-time stride, `strided_sum::<1>` becomes a vector loop like `sum_fast`, and every stride is measured with a loop that is not the bottleneck. **A probe that is itself the bottleneck measures itself.**

`as_chunks::<STRIDE>()` views the data as `&[[f32; STRIDE]]`: one fixed-size array per block. `as_chunks::<8>()` on that groups eight blocks at a time, so there are eight independent running totals. `group[lane][0]` is the first float of each block.

In `main.rs` the ten instantiations are gathered into an array of function pointers:

<!-- file: src/main.rs -->
```rust
    let probes: [(usize, Probe); 10] = [
        (1, strided_pass::<1>),
        (2, strided_pass::<2>),
```

`Probe` is a type alias for `fn(&[f32]) -> Duration`. Each `strided_pass::<N>` is a distinct function, and taking it without calling it gives a plain function pointer.

### 4.5 First touch

<!-- file: src/lib.rs -->
```rust
pub fn first_touch(bytes: usize) -> (Duration, Duration) {
    let mut data: Vec<f32> = vec![0.0; bytes / 4];
    let start = Instant::now();
    data.fill(1.0);
    black_box(&data);
    let first = start.elapsed();
    let start = Instant::now();
    data.fill(2.0);
    black_box(&data);
    (first, start.elapsed())
}
```

`vec![0.0; n]` for a large `n` is special in Rust: the standard library asks the allocator for *zeroed* memory (like C's `calloc`), and the OS provides zero pages lazily. So the allocation itself is nearly free and the page faults land inside the first `fill`. The second `fill` touches the same, now mapped, pages.

### 4.6 Many threads reading

<!-- file: src/lib.rs -->
```rust
    let read_all = || {
        std::thread::scope(|s| {
            for part in data.chunks(chunk) {
                s.spawn(move || black_box(sum_fast(black_box(part))));
            }
        });
    };
```

A preview of chapter 7. `std::thread::scope` lets threads **borrow** `data` directly: `part` is a `&[f32]` slice of the one shared buffer, and no thread copies or owns it. The scope guarantees every thread finishes before `scope` returns, which is why the borrow checker allows this without `Arc` or cloning. The `move` keyword moves the slice reference `part` (not the data) into each thread.

### 4.7 The roofline

<!-- file: src/lib.rs -->
```rust
impl Roofline {
    /// The best achievable GFLOP/s for a kernel doing `intensity` FLOPs per
    /// byte moved.
    pub fn attainable(&self, intensity: f64) -> f64 {
        self.peak_gflops.min(intensity * self.bandwidth_gbs)
    }

    /// The intensity where the two limits meet. Kernels below it are
    /// memory-bound, kernels above it are compute-bound.
    pub fn ridge_point(&self) -> f64 {
        self.peak_gflops / self.bandwidth_gbs
    }
```

<!-- file: src/lib.rs -->
```rust
pub fn linear_layer_intensity(bytes_per_weight: f64, batch: usize) -> f64 {
    2.0 * batch as f64 / bytes_per_weight
}
```

The model is two lines of arithmetic. Its value is in forcing you to count FLOPs and bytes for your kernel, then compare the measured speed to the prediction:

- **Measured close to the roofline:** the kernel is as fast as this approach allows. Change the approach (fewer bytes, more reuse).
- **Measured far below the roofline:** the kernel is wasting something (cache lines, a dependency chain, a missing SIMD path, a thread imbalance). Profile it (chapter 17).

## 5. Run it

```bash
cargo run --release -p ch04-memory      # about 1.5 minutes
```

A full run on the reference machine:

```text
== 1. bandwidth of one core vs working-set size
   size     |  read GB/s | write GB/s
      16 KB |       47.5 |       95.6
      32 KB |       51.3 |       98.7
     128 KB |       51.0 |       51.1
     512 KB |       50.9 |       50.3
       1 MB |       47.6 |       50.4
       2 MB |       35.0 |       26.7
       4 MB |       22.0 |       17.6
      16 MB |       20.0 |       16.3
      64 MB |       22.8 |       15.5
     256 MB |        9.7 |        6.5
    1024 MB |       12.2 |        7.2

== 2. latency of one dependent load vs working-set size
      16 KB |    1.2 ns
     128 KB |    3.2 ns
       1 MB |    5.2 ns
       4 MB |   43.0 ns
      16 MB |   88.8 ns
      64 MB |  183.8 ns
     256 MB |  233.7 ns
    1024 MB |  294.7 ns

== 3. reading one float out of every k, over a 256 MB buffer
   k (stride) | time for the whole pass | floats read | ns per float read
            1 |                 30.7 ms |    67108864 |              0.46
            2 |                 40.9 ms |    33554432 |              1.22
            4 |                 25.9 ms |    16777216 |              1.55
            8 |                 23.8 ms |     8388608 |              2.84
           16 |                 21.2 ms |     4194304 |              5.06
           32 |                 18.6 ms |     2097152 |              8.87
           64 |                 17.8 ms |     1048576 |             16.95
          128 |                 10.6 ms |      524288 |             20.21
          256 |                  3.3 ms |      262144 |             12.44
         1024 |                  0.6 ms |       65536 |              9.43

== 4. first touch of freshly allocated memory (256 MB)
   first write pass: 102.6ms   second write pass: 40.0ms

== 5. read bandwidth of the whole machine (1 GB buffer)
   1 thread(s):   10.2 GB/s
   2 thread(s):   22.3 GB/s
   4 thread(s):   30.5 GB/s

== 6. arithmetic throughput
   one core:  31.0 GFLOP/s   (4 cores: about 124 GFLOP/s)

== 7. roofline for this machine: 124 GFLOP/s peak, 30.5 GB/s, ridge at 4.1 FLOP/byte
   kernel                                | FLOP/byte | best GFLOP/s | limited by
   vector add c = a + b (f32)            |      0.08 |          2.5 | memory
   dot product (f32)                     |      0.25 |          7.6 | memory
   linear layer, batch 1, f32 weights    |      0.50 |         15.2 | memory
   linear layer, batch 1, bf16 weights   |      1.00 |         30.5 | memory
   linear layer, batch 1, int8 weights   |      2.00 |         61.0 | memory
   linear layer, batch 1, 4-bit weights  |      4.00 |        122.0 | memory
   linear layer, batch 16, bf16 weights  |     16.00 |        124.1 | compute
   linear layer, batch 256, bf16 weights |    256.00 |        124.1 | compute
   matmul 4096 x 4096 x 4096 (f32)       |    682.67 |        124.1 | compute
```

Things to notice beyond what section 3 already covered:

- **Writes to DRAM are slower than reads** (about 7 GB/s versus 10-12). Before a core can write part of a cache line, it must own the whole line, so it first *reads* it from memory ("read for ownership"). A write of 64 bytes costs 128 bytes of traffic. Kernels that overwrite whole buffers can use special "non-temporal" store instructions that skip the read; memcpy implementations do.
- **Small writes look faster than small reads** (95-99 GB/s at 16-32 KB). `fill` is a store-only loop the compiler turns into wide stores, and the core can retire stores into L1 faster than our sum loop consumes loads.
- **This run is at the low end for 4-thread bandwidth** (30.5 GB/s; other runs gave 40-45). The roofline moves with it: at 45 GB/s the ridge is at 2.8 and the 4-bit layer at batch 1 becomes compute-bound. On a shared cloud machine, measure several times and quote a range.
- **At batch 1, LLM decode on this machine is memory-bound for every weight format except 4-bit.** For SmolLM2 in `bf16` (270 MB of weights), the ceiling is about 30-45 GB/s ÷ 0.27 GB ≈ 110-165 tokens/s. Chapter 16 measures how close our engine gets.

## 6. The Rust behind it

**`std::hint::black_box`** is the benchmark writer's most important tool. The optimizer is allowed to delete computations whose results are unused and to precompute results whose inputs it can see. `black_box(x)` returns `x` unchanged but tells the compiler to assume it was read and modified by unknown code. Put it around inputs (so they are not constant-folded) and around outputs (so the work is not deleted).

**Const generics turn runtime parameters into compile-time constants.** `strided_sum::<16>` is a different function from `strided_sum::<1>`, each optimized for its own stride. The same technique shows up in real kernels: tile sizes, head dimensions (64 or 128 in almost every model) and block sizes are often const generics, so the compiler can fully unroll and vectorize the inner loops. The cost is code size: each value you instantiate is a separate copy.

**`vec![0; n]` versus `vec![x; n]`.** Rust specializes zero-initialized vectors to use zeroed allocation, which the OS can satisfy lazily. Any other fill value writes every element up front. This is why section 3.5's numbers came out as they did, and it is an easy way to accidentally move page-fault cost into or out of a timed region.

**Scoped threads borrow instead of copy.** `std::thread::scope` lets spawned threads hold `&` references to local data, because the scope cannot end until they finish. For inference, that means worker threads can read the weight buffer directly: no `Arc`, no clone, no reference counting in the hot path. Chapter 7 builds on this.

**Integer conversions are checked where it matters.** `u32::try_from(n)` returns an error if `n` does not fit; `n as u32` silently truncates. We check once with `try_from`, then use `as` knowing it is safe. This is the general pattern for casts in performance code: validate at the boundary, then use the cheap operation in the loop.

## 7. Mistakes you will make

- **Benchmarking a working set that fits in cache** and concluding your kernel is fast. A 1 MB weight matrix runs at L2 speed; the real 100 MB one runs at DRAM speed. Always test at production sizes.
- **Forgetting that the first pass faults pages in.** Allocate, touch, *then* time. Or time the first touch on purpose, if startup cost is what you care about.
- **Believing a probe that is its own bottleneck.** If a "bandwidth" number does not change when you double the data size across a cache boundary, the probe's loop is probably the limit. Compare against a known-good number such as `memcpy`.
- **Quoting one run on a shared machine.** Run three times at least, report the range, and look at the minimum for hardware-capability questions.
- **Reading the roofline with the wrong peak.** The peak must match how your code is compiled. A kernel compiled for generic x86-64 cannot reach the AVX-512 FMA peak, however clever its memory access.

## 8. How the professionals do it

- **Tools:** `perf stat` (Linux) counts cache misses, TLB misses and instructions per cycle for a real program; Intel VTune and AMD uProf add roofline plots; `likwid` measures bandwidth per core. On GPUs, NVIDIA Nsight Compute shows each kernel's position on the roofline directly. Chapter 17 uses `perf` where available.
- **STREAM** is the standard memory bandwidth benchmark (copy, scale, add, triad). Our probes are simplified versions of it.
- **Inference engines are designed around the roofline.** llama.cpp's speed on CPUs and Apple Silicon comes mostly from 4-8 bit weights (fewer bytes) and good threading (more bandwidth). vLLM's and TensorRT-LLM's throughput on GPUs comes mostly from large batches (more reuse per byte).
- **Capacity planning uses the same arithmetic.** "How many tokens/s can this GPU produce for this model?" starts as bandwidth ÷ bytes per token, then adds batching (chapter 30).

## 9. Exercises

1. **Native build.** Run the demo with `RUSTFLAGS="-C target-cpu=native" cargo run --release -p ch04-memory`. Which rows change, and which do not? Explain using the roofline.
2. **Tokens per second ceiling.** SmolLM2-135M in `bf16` is 270 MB of weights. Using your measured single-thread and all-thread bandwidth, compute the maximum decode tokens/s for one user in each case. What does that imply about threading the engine?
3. **Sequential chase.** Change `load_latency_ns` to use `next[i] = (i + 1) % n` instead of a random cycle, and measure the 1 GB case. Why is it so much faster, when every load still depends on the previous one?
4. **Zeros versus ones.** Time `vec![0.0f32; 64 << 20]` and `vec![1.0f32; 64 << 20]` (256 MB each). Explain the difference.
5. **Batch size for an H100.** Using the ridge point in section 3.9, what batch size does a `bf16` linear layer need to be compute-bound on an H100? On this machine?
6. **The stride plateau.** Why does the whole-pass time in part 3 barely fall between stride 4 and stride 64, and then drop sharply?

## 10. Check yourself

1. What is the difference between latency and bandwidth, and which one limits a matrix-vector product?
2. Why does reading one float per cache line cost almost as much as reading all sixteen?
3. A kernel does 10 FLOPs per byte. Is it memory-bound or compute-bound on this machine? On an H100?
4. Why does quantizing weights from 16 to 4 bits speed up decode, even though the arithmetic does not get simpler?
5. Why is one thread not enough to use the whole memory bandwidth?
6. Why is the first request to a freshly started inference server often slow?

## 11. Recap

- Memory is organized in levels: registers, L1, L2, L3, DRAM. Each is larger and slower than the one before. Here: ~50 GB/s and ~1 ns near the core, ~11 GB/s and ~290 ns from DRAM, per core.
- Memory moves in 64-byte lines. You pay for the whole line whether you use 4 bytes or 64.
- Latency (one dependent access) and bandwidth (many overlapping accesses) are different limits. Weight streaming needs bandwidth; pointer chasing needs latency.
- First touch of fresh memory costs page faults. Warm up before measuring, and before serving.
- One core cannot saturate memory; several can.
- Roofline: speed ≤ min(peak FLOP/s, intensity × bandwidth). Decode at batch 1 is memory-bound; quantization and batching raise intensity; SIMD and threads raise the ceilings.

## Answers

**Exercises**

1. Measured on the reference machine: the 32 KB read bandwidth rose from about 51 to 68 GB/s, and the single-core arithmetic peak from 31-33 to about 62 GFLOP/s. DRAM bandwidth did not change (11-12 GB/s, and 40-42 GB/s with 4 threads). Wider vectors raise the compute ceiling and help data that is already in cache; they cannot make DRAM deliver bytes faster. On the roofline: the flat part rises, the slanted part does not, and the ridge point moves right.
2. At ~11 GB/s (one thread): 11 / 0.27 ≈ 40 tokens/s at most. At 30-45 GB/s (four threads): about 110-165 tokens/s. The engine must spread each matvec across all cores to have any chance of approaching the higher number (chapter 7).
3. Measured: 2.7 ns per hop at 1 GB, against 295 ns for the random cycle, about 100x faster. Every load still depends on the previous one, but the addresses are sequential, so the prefetcher recognizes the pattern and has each line in cache before it is asked for. The dependent load then hits in L1. The latency is still there; the hardware hides it by fetching early.
4. Measured: 7-16 µs for zeros, 140-215 ms for ones. The zeroed allocation gets lazily mapped zero pages from the OS and touches nothing. The ones must be written, which faults in all 65,536 pages and writes 256 MB.
5. A `bf16` linear layer has intensity B (batch size). On an H100 (ridge ~295) it needs B ≈ 300; on this machine (ridge ~3-4), B ≈ 4. This is why CPU inference benefits from batching very quickly and then stops benefiting, while GPUs keep benefiting up to large batches.
6. Up to stride 64 floats (256 bytes, every fourth cache line) the hardware prefetchers still fetch most of the lines: adjacent-line prefetch pairs each line with its neighbour, and the streaming prefetcher runs ahead along the detected direction. So nearly the whole buffer still moves from DRAM. From stride 128 (512 bytes, every eighth line) the prefetchers stop pulling in lines that are skipped, and the time finally falls roughly in proportion to the lines actually read.

**Check yourself**

1. Latency is the time for one access to complete; bandwidth is bytes per second with many accesses in flight. Matrix-vector products read weights sequentially with no dependencies between loads, so they are limited by bandwidth.
2. Memory is transferred in 64-byte cache lines. To read one float, the whole line containing it (16 floats) is fetched, so the memory traffic is the same.
3. The ridge here is about 3-4 FLOP/byte, so 10 FLOP/byte is compute-bound on this machine. The H100's ridge is about 295, so the same kernel is memory-bound there.
4. Decode is memory-bound: its speed is bytes of weights ÷ bandwidth. 4-bit weights are a quarter of the bytes of 16-bit weights, so each token needs a quarter of the memory traffic. The extra arithmetic to unpack them is cheap because the arithmetic units were mostly idle anyway.
5. Each core can only have a limited number of cache-line misses outstanding at once. Bandwidth = bytes in flight ÷ latency, so with DRAM latency around 290 ns one core's slots cap it at about 11 GB/s here. More cores bring more slots.
6. Its buffers (KV cache, scratch space) are touched for the first time and page-fault, its code and weights may not be in cache or even in memory yet (with memory-mapped weights, the first read of each page comes from disk), and branch predictors and caches are cold. Servers run warm-up requests before accepting traffic.

## Further reading

- Ulrich Drepper, "What Every Programmer Should Know About Memory", 2007. Long, detailed, still the best explanation of caches, TLBs and prefetching.
- Williams, Waterman and Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM, 2009.
- Igor Ostrovsky, "Gallery of Processor Cache Effects" (blog post). Short experiments much like part 3.
- Next: [Chapter 5: Matrix multiplication](../05-matmul/README.md). The operation that dominates inference, taken from naive to cache-blocked.
