# Chapter 17: Measuring and profiling

> **In one sentence:** before changing code for speed, find out where the time goes (a sampling profiler for the big picture, timers around the parts you care about), write down what you expect a change to buy, test it on the real workload with a comparison that survives a noisy machine, and keep only what the numbers support; here that rejects two plausible ideas for decode and finds a 1.8x speedup for prefill.

**Where this fits:** chapter 16 got SmolLM2 running and left two questions: why does `bf16` decode reach only about 20-30 GB/s of this machine's 30-45, and how fast could prefill be? This chapter answers them with measurements, and adds the tiled kernel that chapter 23's batched decoding also relies on.

**You need:** chapter 4 (roofline), chapter 6 (SIMD kernels), chapter 7 (the spin pool), chapter 14 (the engine), chapter 16 (the real model).

**You will build:** a wrapper that times every matrix multiplication of a running model, an interleaved A/B comparison for noisy machines, a four-row decode kernel and a fused decode pass (both tested, one rejected, one inconclusive), and a 4 × 4 tiled kernel for prefill (kept).

---

## 1. The intuition

A mechanic who hears a noise does not start by replacing parts. They listen to find where it comes from, form a guess, check the guess with an instrument, and only then take something apart. Replacing the part that "usually" causes the noise is how you spend money without fixing the car.

Performance work is the same, and the part that "usually" causes slowness is the one you remember from the last time. This chapter's first idea came from an earlier micro-benchmark that showed a clear gain. On the real model, it made things slower.

**Where the analogy breaks:** a car's noise is the same every time you listen. A program on a shared cloud machine is not: the same binary, measured on the same machine on the same afternoon, decoded between 83 and 156 tokens per second in this chapter's runs. Half the craft here is measuring well enough to see a 5% change through a 40% noise.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Profile** | A breakdown of where a program spends its time. |
| **Sampling profiler** | Interrupts the program many times per second and records where it was (`perf`, VTune, Instruments). Cheap, covers everything, statistical. |
| **Instrumentation** | Timers placed in the code around the parts you care about. Exact for those parts, blind elsewhere. |
| **Hot spot** | Code where a large share of the time goes. |
| **Wall time / CPU time** | Time on a clock / time summed over all threads. A parallel program uses more CPU time than wall time. |
| **Micro-benchmark** | A small program that measures one operation in isolation. Useful, and misleading when its shapes differ from the real workload's. |
| **Noise** | Variation between runs of the same work, from other programs, the OS, the hypervisor, frequency changes. |
| **A/B comparison** | Measuring version A against version B. Interleaved: alternating them, so slow drifts affect both. |
| **Amdahl's law** | Speeding up a fraction `f` of the work by `s` gives an overall speedup of `1 / ((1 − f) + f / s)`. |
| **Register tiling** | Computing a block of outputs at once so each loaded value is used several times from registers. |
| **Load-bound / FMA-bound** | A kernel limited by how many loads, or how many multiply-adds, the core can issue per cycle. |

## 3. The concepts in depth

### 3.1 The method

1. **Measure the whole thing** with the metric that matters (tokens/s, time to first token), and compare it with what the hardware allows (chapter 4's roofline).
2. **Locate the time.** A sampling profiler first, for the big picture; then instrumentation for detail.
3. **Form a hypothesis and predict its effect in numbers.** "The four-row kernel will raise decode bandwidth from 9 to 12 GB/s per core" can be wrong; "it should be faster" cannot.
4. **Test on the real workload**, with a comparison that is robust to noise.
5. **Keep it only if it wins**, then measure again: the profile has changed.

### 3.2 A sampling profile

Linux's `perf` samples a running program. On the reference machine the `perf` matching its kernel was not packaged, so it came from `apt install linux-tools-generic` and ran from `/usr/lib/linux-tools-6.8.0-142/perf`. The virtual machine exposes no hardware performance counters (no cycle or cache-miss counts), so the samples come from a software timer, `-e cpu-clock`:

```bash
perf record -e cpu-clock -F 2000 -g -- \
    target/release/ch16-real-model ask --max-tokens 100 "Explain in two sentences why the sky is blue."
perf report --no-children --sort symbol
```

```text
    53.52%  [.] ch06_simd::x86::dot_bf16_avx512
    29.87%  [.] ch07_threads::pool::worker_loop
     4.98%  [.] ch07_threads::pool::SpinPool::run
     1.35%  [.] ch06_simd::dot_bf16_with
     1.11%  [.] ch07_threads::pool::SpinPool::for_each_chunk_mut::{{closure}}
     1.02%  [.] ch07_threads::pool::SpinPool::for_each_chunk_mut::{{closure}}
     0.83%  [.] ch06_simd::dot_bf16
     ...
```

Two findings:

- **The `bf16` dot-product kernel is the hot spot**: 54% of all samples. Everything the model computes besides matrix products (attention, norms, RoPE, sampling) is a few percent.
- **35% of the samples are threads waiting** (`worker_loop` and `SpinPool::run` spinning). These percentages are of CPU time across all four threads. Whenever the main thread does serial work (a norm, RoPE, sampling), the three workers spin; within each parallel call, threads that finish early spin until the last one finishes. Waiting is not in itself a problem to fix, but it tells you that more parallel work per call, or less serial work between calls, would be used.

### 3.3 Instrumenting the model

`perf` says "the kernel"; it does not say which matrices, or how fast each runs compared with the hardware. For that, `Timed<W>` wraps each matrix of a loaded model and times every multiplication (section 4.1). Decoding 64 tokens:

```text
== 1. where a decode step's time goes
   matrices         ms   share     GB/s
   q             0.773    8.5%     25.7
   k             0.258    2.8%     25.7
   v             0.258    2.8%     25.7
   o             0.659    7.3%     30.2
   gate          1.642   18.1%     32.3
   up            1.642   18.1%     32.3
   down          1.540   17.0%     34.5
   lm_head       1.572   17.4%     36.0
   the rest      0.710    7.8%   attention, norms, RoPE, residuals, lookup
   step          9.055  = 110.4 tokens/s, 29.7 GB/s of weights overall
```

Matrix multiplications are 92% of a decode step, and every one of them is memory-bound (chapter 14): its time is its bytes divided by the bandwidth it gets. They get between 26 and 36 GB/s in this run, the small matrices (Q, K, V: 216-648 KiB per layer) least. Over six runs the whole step moved between 22 and 42 GB/s of weights, and chapter 4 measured 30-45 GB/s for this machine with four threads. So decode runs at, or not far below, what the memory delivers, and the question is whether any gap is left to recover.

(`q`, `k` and `v` show identical GB/s because the forward pass multiplies them with one `matmul_many` call, and the wrapper divides that call's time between them by size.)

### 3.4 Measuring through noise

Here is the decode step from six runs of this chapter's program, same binary, same machine, one afternoon: 83, 95, 110, 126, 144 and 156 tokens/s. The machine is a virtual machine sharing a physical server; other tenants' work changes the memory bandwidth and CPU time it gets, minute by minute. No single comparison of "before" and "after" means anything at this level of noise.

`compare` (section 4.2) makes A/B tests usable anyway:

- **Interleave.** Run A, then B, then B, then A, and so on (alternating which goes first). Interference changes over seconds; one pair takes a fraction of a second, so both halves of a pair see nearly the same machine.
- **Compare within pairs.** The per-pair ratio `time(A) / time(B)` cancels most of what the pair shared. Report the median ratio and the spread (here, the range covering the middle 80% of pairs).
- **Repeat the whole comparison** on other runs. A real effect keeps its sign.

Each run of each decode comparison below is 30 pairs of 16 decode steps; each prefill comparison, 20 pairs.

### 3.5 Decode idea 1: stream four rows at once

The idea: `dot_bf16` streams one weight row at a time. A kernel that streams four rows against the same activation vector loads each activation once instead of four times, and keeps four streams of memory requests in flight instead of one. A micro-benchmark with rows of 8,192 values (the last line of part 2 below) shows single-core bandwidth rising from about 9 to 12 GB/s. The prediction: decode 20-30% faster.

Part 2 of the demo checks the prediction at the row lengths that matter:

```text
== 2. decode idea 1: stream four rows at once
   one thread, rows of  576 values ( 1152 bytes): one row at a time   9.2 GB/s, four rows   7.7 GB/s
   one thread, rows of 1536 values ( 3072 bytes): one row at a time   9.1 GB/s, four rows   8.2 GB/s
   one thread, rows of 8192 values (16384 bytes): one row at a time   9.1 GB/s, four rows  11.6 GB/s
   plain vs four-row kernel, decode: 125.87ms -> 129.01ms, speedup 1.00x (80% of pairs: 0.81-1.09x)
```

The gain exists only for long rows. SmolLM2's rows are 576 and 1,536 values long, and there the four-row kernel is *slower*. Over six runs, its decode speedup was 0.92, 0.92, 0.94, 0.97, 0.97 and 1.00: never faster. Rejected.

Why? The rows of a matrix are stored one after another. Reading one row at a time is a single sequential stream, the pattern the hardware prefetcher predicts best. Reading four adjacent short rows at once turns it into four interleaved streams 1 or 3 KB apart, which the prefetcher handles less well, and the extra streams gain nothing because one sequential stream already keeps the core's outstanding-request slots busy. With 16 KB rows, each stream is long enough for the prefetcher to follow separately, and the four streams do add up. The micro-benchmark was right about its own shape and wrong about ours.

### 3.6 Decode idea 2: fewer, larger calls

The idea: the table in 3.3 shows the small matrices getting the least bandwidth. Maybe every parallel call has a fixed cost (waking the threads, starting four fresh memory streams, waiting for the slowest thread), and 211 calls per token add up. Part 3 measures one parallel matrix-vector product at several sizes, on matrices that are not in any cache:

```text
== 3. decode idea 2: fewer, larger parallel calls
   4 threads,     64 x 576 matrix (    72 KiB):     5.3 µs per call,  13.8 GB/s
   4 threads,    192 x 576 matrix (   216 KiB):    10.4 µs per call,  21.3 GB/s
   4 threads,    576 x 576 matrix (   648 KiB):    26.8 µs per call,  24.7 GB/s
   4 threads,   1536 x 576 matrix (  1728 KiB):    63.5 µs per call,  27.8 GB/s
   4 threads,   6144 x 576 matrix (  6912 KiB):   243.5 µs per call,  29.1 GB/s
   4 threads,  24576 x 576 matrix ( 27648 KiB):   996.3 µs per call,  28.4 GB/s
   fitted line: 1.4 µs per call + bytes at 28.5 GB/s
```

Small calls are indeed slower per byte. The fitted fixed cost, though, came out anywhere between 1.4 and 13 µs over the runs (in one more run, a burst of interference made the fit meaningless): the effect is real but small, and at the edge of what this machine can resolve. The prediction for fusing the matrices that share an input (Q, K and V into one pass; gate and up into another), which cuts 211 calls to 121: between 1% (with 1.4 µs per call) and 10% (with 10 µs) faster decode.

The measurement:

```text
   plain vs fused q|k|v and gate|up, decode: 113.26ms -> 103.49ms, speedup 1.04x (80% of pairs: 0.96-1.24x)
```

Over six runs: 0.97, 1.01, 1.04, 1.06, 1.06, 1.07. Probably a small gain, of the size predicted, but not one that every run confirms. The chapter keeps the fused code as an experiment (`FusedBf16`) and does not build on it. On GPUs the same fusion is standard, because each kernel launch there costs several microseconds regardless of size.

The conclusion for decode: it runs close to this machine's memory bandwidth, and whatever gap remains is spread thin. The big levers are elsewhere: reading fewer bytes per token (chapters 18 and 19) and producing more tokens per byte read, by decoding several sequences at once (chapter 23).

### 3.7 Prefill: compute-bound and far from the roof

Prefill multiplies each weight matrix by many tokens at once, so it is compute-bound (chapter 14). Part 4 times a 256-token prefill:

```text
== 4. where a 256-token prefill's time goes
   matrices         ms   share  GFLOP/s
   q            70.923    8.7%     71.9
   k            23.641    2.9%     71.9
   v            23.641    2.9%     71.9
   o            64.957    8.0%     78.5
   gate        179.357   22.0%     75.8
   up          179.357   22.0%     75.8
   down        147.471   18.1%     92.2
   lm_head       1.739    0.2%     32.6
   the rest    123.965   15.2%   attention, norms, RoPE, residuals, lookup
   prefill     815.050  = 314 tokens/s
```

(The LM head multiplies only the last token, so it is a small matrix-vector product here.)

Matrix products run at 72-92 GFLOP/s on four cores, about 20 per core. What should one core manage? Count what the kernel does per 16 columns with AVX-512:

- **One dot product per output** (chapter 6's kernel, called once per weight row per token): 1 weight load, 1 activation load, a widening of the weights, 1 fused multiply-add. Two loads per FMA: the kernel is limited by loads, and every weight row is re-read from the cache once per token.
- **A 4 × 4 tile** (four weight rows against four tokens): 4 weight loads and widenings, 4 activation loads, 16 FMAs, each loaded value used four times. Half a load per FMA.

Server cores of this generation can issue up to two 16-wide FMAs and two 64-byte loads per cycle. The dot kernel can then do at most one FMA per cycle (32 FLOPs); the tile kernel can approach two (64 FLOPs). Part 5 measures a prefill-shaped product on one thread, then the whole prefill with interleaved comparisons:

```text
== 5. prefill idea: a 4 x 4 tile kernel
   one thread, 1536 x 576 weights times 64 tokens: one dot product per output 29.3 GFLOP/s, 4 x 4 tiles 82.9 GFLOP/s
   plain vs tiled, 40-token prefill: 114.39ms -> 63.53ms, speedup 1.85x (80% of pairs: 1.69-2.05x)
   plain vs tiled, 256-token prefill: 787.46ms -> 448.42ms, speedup 1.80x (80% of pairs: 1.53-2.00x)
   plain vs tiled, decode (same code path): 117.85ms -> 118.60ms, speedup 1.01x (80% of pairs: 0.94-1.24x)
```

Over six runs the single-thread kernel went from 23-29 to 60-83 GFLOP/s (2.5-2.8x), and prefill became 1.60-1.87 times faster, in every comparison. Decode, which never calls the tile kernel, stayed at 0.99-1.03. This one is kept: `TiledBf16`.

### 3.8 Amdahl's law, measured

The kernel became 2.5-2.8 times faster (2.8 in this run); prefill only 1.6-1.9. Part 6 shows why:

```text
== 6. where a 256-token prefill's time goes now
   matrices         ms   share  GFLOP/s
   q            33.537    7.2%    152.0
   k            11.179    2.4%    152.0
   v            11.179    2.4%    152.0
   o            31.448    6.7%    162.0
   gate         95.301   20.4%    142.6
   up           95.301   20.4%    142.6
   down         66.096   14.1%    205.6
   lm_head       2.461    0.5%     23.0
   the rest    121.499   26.0%   attention, norms, RoPE, residuals, lookup
   prefill     468.000  = 547 tokens/s
```

Before, matrix products took 691 of 815 ms (85%). They became 1.99 times faster (691 → 347 ms), and everything else stayed at about 122 ms. Amdahl's law predicts `1 / (0.15 + 0.85 / 1.99) = 1.73`; the measured ratio of these two runs is 815 / 468 = 1.74. The part you did not touch sets the limit: even infinitely fast matrix products would make this prefill at most 1 / 0.15 = 6.6 times faster. "The rest" has gone from 15% to 26% of prefill, and most of it is attention, whose cost grows with the square of the prompt length. That is chapter 20's subject.

The tiled products ran at 140-240 GFLOP/s on four threads across the runs, above the 124 GFLOP/s chapter 4 measured as this machine's peak. That peak was for code compiled for baseline x86-64 (256-bit vectors); the tile kernel uses AVX-512 through runtime detection, with twice the width.

## 4. The code

The timing wrapper is in [`src/timed.rs`](src/timed.rs), the kernels in [`src/kernels.rs`](src/kernels.rs), the matrix products in [`src/matmul.rs`](src/matmul.rs), the matrix types and `compare` in [`src/lib.rs`](src/lib.rs), the demo in [`src/main.rs`](src/main.rs).

### 4.1 Timing every multiplication

<!-- file: src/timed.rs -->
```rust
    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        let start = Instant::now();
        self.inner.matmul(pool, x, y, m, scratch);
        let t = &self.timings;
        t.nanos[self.slot].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        t.calls[self.slot].fetch_add(1, Ordering::Relaxed);
        t.bytes[self.slot].fetch_add(self.inner.bytes() as u64, Ordering::Relaxed);
    }
```

`Timed<W>` implements `Matrix` by delegating to the matrix it wraps, adding the elapsed time, a call count and the bytes read to one of eight slots (`q`, `k`, ..., `lm_head`). `instrument` takes a model apart with chapter 14's `into_parts`, wraps each matrix with its slot, and rebuilds it. The engine runs unchanged; it never knows it is being timed. The timers read the clock twice per multiplication, a few tens of nanoseconds against tens of microseconds for the multiplication itself.

The counters are atomics because `matmul` takes `&self` and the model is shared between threads (`Matrix: Sync`): a plain `u64` field could not be updated through a shared reference. `Relaxed` ordering is enough, since nothing else depends on the order in which the counters change.

### 4.2 An interleaved comparison

<!-- file: src/lib.rs -->
```rust
    for i in 0..pairs {
        let (x, y) = if i % 2 == 0 {
            let x = time(&mut a, pool);
            (x, time(&mut b, pool))
        } else {
            let y = time(&mut b, pool);
            (time(&mut a, pool), y)
        };
        ratios.push(x.as_secs_f64() / y.as_secs_f64());
        ta.push(x);
        tb.push(y);
    }
```

Both versions run on the same thread pool, passed into each closure as `&mut SpinPool`. Two pools would mean eight spinning threads on four cores, each pool's workers stealing time from the other's. And a closure cannot keep its own `&mut` to one pool while another closure holds one too, which is why the pool is a parameter rather than something the closures capture.

### 4.3 The tile kernel

<!-- file: src/kernels.rs -->
```rust
        for s in 0..steps {
            let col = s * 16;
            let mut vx = [_mm512_setzero_ps(); 4];
            for i in 0..4 {
                // SAFETY: col + 16 <= k, the length of every vector.
                vx[i] = unsafe { _mm512_loadu_ps(xs[i].as_ptr().add(col)) };
            }
            for r in 0..4 {
                // SAFETY: col + 16 <= k, the length of every row.
                let w = unsafe { load16(rows[r], col) };
                for i in 0..4 {
                    acc[r][i] = _mm512_fmadd_ps(w, vx[i], acc[r][i]);
                }
            }
        }
```

Sixteen accumulators (`acc[r][i]`, one per weight row and token), four activation vectors and one weight vector at a time: 21 of AVX-512's 32 vector registers, so nothing spills to memory. The loops have constant bounds, so the compiler unrolls them completely and every `acc[r][i]` becomes a register. AVX2 has only 16 registers, so its version computes the tile as two 4 × 2 halves, widening each weight twice rather than spilling accumulators.

The kernel is a safe `#[target_feature]` function that checks the lengths itself; its only remaining requirement is the CPU feature, which is why calling it is `unsafe` everywhere except inside other AVX-512 code.

### 4.4 The tiled product

<!-- file: src/matmul.rs -->
```rust
            for i0 in (0..full).step_by(4) {
                let xs = [x_row(i0), x_row(i0 + 1), x_row(i0 + 2), x_row(i0 + 3)];
                for j0 in (g..quads).step_by(4) {
                    let ws = [w_row(j0), w_row(j0 + 1), w_row(j0 + 2), w_row(j0 + 3)];
                    let out = tile(ws, xs);
                    for (r, row) in out.iter().enumerate() {
                        for (c, &v) in row.iter().enumerate() {
                            put(j0 + r, i0 + c, v);
                        }
                    }
                }
```

The structure is chapter 14's: threads split the weight rows (in multiples of four), each walks its rows in groups small enough to stay in cache while all the tokens pass through, and results are written to a transposed scratch buffer and transposed at the end. The inner loop now takes four tokens and four weight rows at a time. Tokens and rows left over (when there are not a multiple of four) fall back to single dot products.

`TiledBf16` uses this for `m > 1` and chapter 14's matrix-vector product for `m == 1`, so its decode path is exactly chapter 16's.

## 5. Run it

```bash
cargo test -p ch17-profiling
cargo run --release -p ch17-profiling      # about 80 seconds; needs the model
```

The output is shown in sections 3.3 to 3.8. Expect your numbers to move from run to run; the ratios from `compare` move much less.

The tests include [`tests/reference.rs`](tests/reference.rs), which runs every matrix type of this chapter against chapter 16's PyTorch fixture: all agree with it as closely as chapter 16's model does (largest logit difference below 1e-3), and the fused decode pass matches the plain one bit for bit.

## 6. The Rust behind it

**Interior mutability through atomics.** `Matrix` methods take `&self`, and the model is shared across the pool's threads. To record statistics through a shared reference, the counters must allow mutation through `&`: `Cell` does (single-threaded only, so not `Sync`), `Mutex` does (with locking), atomics do (lock-free, for plain numbers). `fetch_add(n, Ordering::Relaxed)` is a single atomic instruction.

**Wrapping a trait implementation.** `impl<W: Matrix> Matrix for Timed<W>` is the decorator pattern with no runtime cost: `Model<Timed<DenseBf16>>` is compiled like any other model, the wrapper's calls are inlined, and the timing code exists only in the instrumented type.

**Overriding a provided method.** `Matrix::matmul_many` (chapter 14) has a default body. `FusedBf16` replaces it with one parallel pass, and `Timed` replaces it to forward to the inner type's version, so that timing a fused model does not un-fuse it.

**Keeping the optimizer honest.** A benchmark whose results are never used can be deleted by the compiler. `std::hint::black_box(&y)` tells the compiler the value is used in ways it cannot see, so the work that produced it must happen. The micro-benchmarks in `main.rs` end with it.

**Safe `#[target_feature]` functions.** Since Rust 1.86 such functions can be safe `fn`s; calling one is `unsafe` only from code that does not itself enable the feature. The kernel must then be sound for any arguments, which is why it asserts that all lengths match instead of trusting the caller.

## 7. Mistakes you will make

- **Optimizing what you have not profiled.** The kernel you just read about is rarely the one that matters.
- **Trusting a micro-benchmark with the wrong shape.** The four-row kernel won on 16 KB rows and lost on SmolLM2's 1 KB rows.
- **Comparing one run of A with one run of B** on a shared machine. Interleave, compare pairs, repeat.
- **Reading CPU-time percentages as wall time.** `perf`'s 35% of spinning is summed over four threads; the wall-clock cost is what the timers show.
- **Forgetting warm-up.** The first run pays page faults, cold caches and lazy initialization.
- **Letting the compiler delete the benchmark.** Use `black_box`, and be suspicious of any result that is "too good".
- **Expecting a kernel's speedup end to end.** Amdahl's law: 2.6x in the kernel became 1.8x for prefill.

## 8. How the professionals do it

- **Sampling profilers:** `perf` (Linux), `samply` (cross-platform, with the Firefox profiler as viewer), Intel VTune and AMD uProf (with hardware counters: cache misses, memory bandwidth, port usage), Apple Instruments. Flame graphs (`cargo flamegraph`, `inferno`) draw a profile's call stacks so wide boxes are where the time goes.
- **Benchmark harnesses:** `criterion` and `divan` for Rust functions (warm-up, repetitions, statistics; `criterion` also compares against saved baselines), `hyperfine` for whole commands. llama.cpp ships `llama-bench`; vLLM ships latency and throughput benchmark scripts.
- **GPU profiling** uses NVIDIA Nsight Systems (a timeline of kernels and transfers) and Nsight Compute (one kernel's use of the hardware), or the PyTorch profiler. Fusion (QKV projections, gate and up, norm into matmul) is standard there, because each kernel launch has a fixed cost.
- **Kernel libraries tile at several levels**: registers (as here), L1 and L2 blocks, and, on GPUs, shared memory and tensor-core fragments (chapter 29). Libraries like oneDNN, BLIS and CUTLASS choose tile shapes per CPU or GPU, and some engines autotune them on the target machine.
- **Performance regressions are caught in CI** by running benchmarks on dedicated, quiet machines (not shared VMs) and comparing against a stored baseline with a statistical test.

## 9. Exercises

1. **Amdahl in advance.** From the table in section 3.7 alone, and the single-thread kernel speedup (2.8x in that run), predict the prefill speedup before measuring. Why is the prediction higher than the measured 1.74x?
2. **Loads per FMA.** For AVX-512, count loads and FMAs per 16 columns for the single dot product, the four-row kernel and the 4 × 4 tile. Which of them could, in principle, keep two FMA units busy?
3. **The fused-call estimate.** With the fitted line from section 3.6 at 10 µs per call and 30 GB/s, estimate the decode step for 211 calls and for 121 calls (the weights are 269 MB). What speedup does that predict?
4. **Batched decode.** Decoding four sequences at once multiplies each weight matrix by four activation vectors instead of one. Which kernel does that use, and what does it do to the bytes read per token? (Chapter 23 builds it.)
5. **Your machine's noise.** Run `cargo run --release -p ch16-real-model -- bench` five times and note the spread of decode tokens/s. Then run this chapter's decode comparison of the plain model against itself (A = B). What range of ratios do you get?

## 10. Check yourself

1. What does a sampling profiler tell you that timers do not, and the other way round?
2. Why can `perf` show 35% of time spinning without decode being 35% slower than it could be?
3. Why does interleaving A and B, and comparing within pairs, cancel most of the noise?
4. Why did the four-row kernel help 16 KB rows and hurt 1 KB rows?
5. Why is the tile kernel faster than one dot product per output, when both do the same FLOPs?
6. Why did the tile kernel not change decode speed?
7. State Amdahl's law, and apply it to this chapter's prefill.

## 11. Recap

- Measure, locate, predict, test on the real workload, keep only what wins, measure again.
- `perf` found the `bf16` kernel (54%) and waiting threads (35% of CPU time). Timers per matrix found decode running close to the machine's memory bandwidth.
- On a shared VM, the same program measured 83-156 tokens/s. Interleaved A/B comparisons with per-pair ratios still resolve differences of a few percent.
- Four-row decode kernel: faster on 16 KB rows, slower on SmolLM2's rows; never faster in six runs. Rejected.
- Fused Q|K|V and gate|up: 0.97-1.07x over six runs, about what a small per-call cost predicts. Inconclusive; not built on.
- A 4 × 4 tile kernel: half a load per FMA instead of two; 2.5-2.8x on one core, 1.6-1.9x on prefill. Kept.
- Amdahl's law predicted the end-to-end result to within 1%. The untouched 15% (mostly attention) is now 26% of prefill.

## Answers

**Exercises**

1. Matrix products are 691 of 815 ms (85%). With a 2.8x faster kernel, Amdahl predicts `1 / (0.15 + 0.85 / 2.8) = 2.2x`. The measured end-to-end matrix speedup was only 1.99x, because a four-thread product is not just the kernel: threads share the memory system, the transposed output must be written and transposed back, and edges fall back to single dot products. Amdahl with the measured 1.99x gives 1.73x, against 1.74x measured.
2. Single dot product: 2 loads (16 activations, 16 weights), 1 FMA: 2 loads per FMA. Four rows: 5 loads (1 activation, 4 weights), 4 FMAs: 1.25. Tile: 8 loads, 16 FMAs: 0.5. With two loads per cycle, the first can issue at most 1 FMA per cycle and the second at most 1.6; only the tile can in principle feed two FMA units every cycle (in practice the widening instructions compete for some of the same execution ports).
3. Bytes: 269 MB / 30 GB/s = 8.97 ms. 211 calls: 8.97 + 2.11 = 11.08 ms. 121 calls: 8.97 + 1.21 = 10.18 ms. Predicted speedup 1.09x. With the 1.4 µs fit instead: 9.27 against 9.14 ms, 1.01x. The measured 0.97-1.07x lies between the two.
4. It is an `m = 4` product, so it takes the tiled path: each weight is read once and used for four tokens, so bytes read per token drop to a quarter while the arithmetic per token stays the same. Decode moves from memory-bound towards compute-bound, the same shift chapter 14's chunked prefill showed.
5. The spread depends on the machine and the moment; on the reference machine, decode varied from 83 to 156 tokens/s between runs of this chapter's program. An A = B comparison should give a median ratio close to 1.00, and its 80% range shows the smallest difference your setup can detect: if it spans 0.9-1.1, a 3% change will need many repeated comparisons to confirm.

**Check yourself**

1. A sampling profiler covers the whole program without changing it and shows where time goes you did not think to look; its numbers are statistical and in CPU time. Timers measure exactly the parts you choose, in wall time, and can attach meaning (which matrix, how many bytes), but see nothing outside them.
2. The percentage is of CPU time summed over four threads. Workers spin while the main thread does serial work and while waiting for the slowest thread of each call; that time is idle hardware, not time added to the step. The step's wall time is set by the critical path.
3. Interference from other programs changes over seconds or longer, while one pair takes well under a second. Both halves of a pair see nearly the same conditions, so their ratio reflects the code; alternating which runs first cancels order effects.
4. Each of four long rows is a long sequential stream, and the prefetcher follows each one, so four streams keep more memory requests in flight. Four adjacent short rows interleave into a pattern the prefetcher follows less well than one sequential stream, and one stream already kept the core's request slots busy.
5. It does the same arithmetic with far fewer loads (half a load per FMA instead of two), so the core is limited by its multiply-add units instead of its load ports, and each weight row is fetched from cache a quarter as often.
6. Decode multiplies each matrix by one token, so there is nothing to tile: `m == 1` takes chapter 14's matrix-vector path. And decode is memory-bound, so faster arithmetic would not help it anyway.
7. Speeding up a fraction `f` of the work by `s` gives `1 / ((1 − f) + f / s)`. Here `f = 0.85`, `s = 1.99`: 1.73x predicted, 1.74x measured. The limit, with `s` infinite, is `1 / 0.15 = 6.6x`.

## Further reading

- Brendan Gregg, *Systems Performance* (2nd edition, 2020), and his pages on `perf` and flame graphs.
- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication", 2008: register and cache tiling.
- Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", 1967.
- Georges, Buytaert and Eeckhout, "Statistically Rigorous Java Performance Evaluation", 2007: why single runs mislead, with methods that apply to any language.
- Next: [Chapter 18: Quantization I: int8](../18-int8/README.md). Halve the bytes per weight again.
