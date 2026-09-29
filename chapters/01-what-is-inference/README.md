# Chapter 1: What inference is, and what an inference engineer does

> **In one sentence:** inference is running an already-trained model to get answers, and inference engineering is the craft of doing that fast, cheaply and reliably.

**Where this fits:** this is the map for the whole course. Every later chapter zooms into one part of it.

**You need:** a working Rust toolchain and basic Rust (functions, structs, `Vec`, references). Nothing about machine learning.

**You will build:** a tiny one-layer "model", and a program that measures its latency, its throughput as the batch grows, and what happens when you copy its weights instead of borrowing them.

---

## 1. The intuition

Think of a restaurant.

Before it opens, a chef spends months in a test kitchen working out a recipe. They try thousands of variations, taste, adjust, and write down the final amounts: 212 grams of flour, 3.5 grams of salt, and so on. That is **training**. It is slow, expensive, and happens once (or a few times).

Then the restaurant opens. Every night, cooks follow that written recipe for hundreds of customers. They never change the amounts. Their job is to get each plate out quickly, keep the kitchen from jamming when a big group arrives, and not waste ingredients. That is **inference**.

A neural network is the recipe. Its "amounts" are millions or billions of numbers called **weights** (also called **parameters**). Training finds the numbers. Inference uses them, again and again, for every request that comes in.

**Where the analogy breaks:** a recipe is a short list and a cook reads it once per night. A model's weights are gigabytes, and the computer has to read *all* of them for *every* word a language model produces. That single fact (the recipe is enormous and must be re-read constantly) drives most of this course.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Model** | A function with a large number of fixed parameters. Input numbers go in, output numbers come out. |
| **Weights / parameters** | The fixed numbers inside the model, produced by training. |
| **Training** | Searching for weights that make the model's outputs good. Not covered in this course. |
| **Inference** | Running the model with fixed weights on new inputs. Also called "serving" or "prediction". |
| **Forward pass** | One run of the model from input to output. |
| **Request** | One unit of work from a user, for example "complete this prompt". |
| **Latency** | How long one request takes, start to finish. |
| **Throughput** | How much work gets done per second (requests per second, or tokens per second). |
| **Batch** | Several requests processed together in one forward pass. |
| **Token** | The unit a language model reads and writes: a word or piece of a word (chapter 11). |
| **Percentile (p50, p99)** | p99 = 99% of requests were at least this fast. The slow 1% is the tail. |
| **FLOP** | One floating-point operation (a multiply or an add). GFLOP/s = billions per second. |
| **Memory bandwidth** | How many bytes per second the processor can pull from memory. |

## 3. The concepts in depth

### 3.1 A model is a function with a lot of numbers in it

Strip away the vocabulary and a neural network is a chain of simple steps. The most important step, by far, is the **linear layer**:

```text
y = W · x

x : input vector,  length in_dim
W : weight matrix, out_dim rows × in_dim columns
y : output vector, length out_dim

y[o] = W[o][0]·x[0] + W[o][1]·x[1] + ... + W[o][in_dim-1]·x[in_dim-1]
```

Each output number is a **dot product** of one row of `W` with the input. A large language model is roughly a hundred of these layers stacked, with a few cheaper operations in between. More than 90% of its arithmetic is in linear layers. That is why this chapter's toy model is a single linear layer: it is small, but it has the right cost profile.

### 3.2 Training versus inference

| | Training | Inference |
|---|---|---|
| Goal | Find good weights | Use fixed weights |
| How often | Once, or rarely | Every request, forever |
| Work per step | Forward pass + backward pass + weight update | Forward pass only |
| What matters | Total time to a good model | Latency per request, cost per request, reliability |
| Numbers | Needs high precision for gradients | Often works with 8-bit or 4-bit weights (chapters 18-19) |

Over its lifetime a popular model spends far more money on inference than it cost to train, because inference cost scales with the number of users. A 10% speedup in inference is money saved every day.

### 3.3 What an inference engineer actually does

The work stacks up in layers. This course walks up the stack in order:

```text
 ┌─────────────────────────────────────────────────────────────┐
 │ Serving:   HTTP API, streaming, queues, scheduling, SLOs     │  chapters 21-25
 ├─────────────────────────────────────────────────────────────┤
 │ Engine:    KV cache, batching, sampling, speculative decode  │  chapters 14-15, 23-27
 ├─────────────────────────────────────────────────────────────┤
 │ Model:     tokenizer, attention, transformer, weight loading │  chapters 9-13, 16
 ├─────────────────────────────────────────────────────────────┤
 │ Numerics:  float formats, quantization                       │  chapters 2, 18-19
 ├─────────────────────────────────────────────────────────────┤
 │ Kernels:   matmul, SIMD, threads, fused attention            │  chapters 5-8, 20
 ├─────────────────────────────────────────────────────────────┤
 │ Hardware:  caches, memory bandwidth, GPUs, multiple devices  │  chapters 4, 28-29
 └─────────────────────────────────────────────────────────────┘
```

On a given day an inference engineer might: find out why the 99th-percentile latency doubled after a deploy; make a matrix multiply 30% faster; decide whether a model can be stored in 4-bit numbers without hurting answers; work out how many users one GPU can serve; or find the bug that makes the model produce garbage after 2,048 tokens. The common thread is *measuring* where time and memory go, and knowing enough about every layer to fix it.

### 3.4 Why language models are a special case

Most models answer in one forward pass: an image goes in, a label comes out. A language model writes text **one token at a time**. To produce a 200-token answer it runs 200 forward passes, each one reading all the weights, each depending on the token before it. This is called **autoregressive generation**.

It splits a request into two phases that behave very differently:

- **Prefill.** The model reads the whole prompt at once. Many tokens are processed together, so each weight read from memory is used many times. This phase is limited by arithmetic speed.
- **Decode.** The model produces the answer token by token. Each step processes one token, so every weight is read from memory and used exactly once. This phase is limited by memory speed.

Users feel these two phases as two separate numbers:

- **TTFT (time to first token):** how long until text starts appearing. Mostly prefill.
- **TPOT (time per output token):** how fast text streams after that. Mostly decode. Its inverse is the "tokens per second" people quote.

You will build both phases in chapter 14 and measure them on a real model in chapter 16.

### 3.5 The numbers that describe an inference system

**Latency** is time per request. Never report only the average. A service where 99 requests take 10 ms and one takes 5 seconds has an average of 60 ms, which describes nobody's experience. Report percentiles:

- **p50 (median):** the typical request.
- **p90, p99:** the tail. At scale, the tail is what users complain about. If a web page makes 20 model calls, the chance that at least one of them hits the p99 is 1 − 0.99²⁰ ≈ 18%.
- **max:** useful for spotting pathological cases, too noisy to use as a target.

**Throughput** is work per second: requests/s, or for language models, tokens/s summed over all users.

**Latency and throughput pull against each other.** Batching (processing several requests together) raises throughput but makes each request wait for the whole batch. Much of serving design (chapters 23-25) is about getting throughput without letting latency blow up.

**Cost** is dollars per million tokens, which falls out of throughput and the price of the hardware.

**Memory footprint** decides what fits. Weights plus per-request state must fit in the accelerator's memory, or the request cannot run at all.

### 3.6 The two budgets: arithmetic and memory

Every computation spends two things: arithmetic (FLOPs) and memory traffic (bytes). Whichever one runs out first sets the speed.

For our linear layer with a 4096 × 4096 weight matrix in 32-bit floats:

```text
weights:        4096 × 4096 × 4 bytes      = 67.1 MB read per request
arithmetic:     4096 × 4096 × 2 FLOPs      = 33.6 MFLOP per request
ratio:          33.6 M / 67.1 M            = 0.5 FLOP per byte
```

Half an operation per byte is very low. A modern CPU core can do tens of FLOPs in the time it takes to fetch one byte from main memory, so this layer spends most of its time waiting for data. Chapter 4 turns this into a proper model (the "roofline"). For now, keep one rule of thumb:

> For a language model generating one token for one user, **time per token ≈ size of the weights in bytes ÷ memory bandwidth.**

A model with 7 billion parameters stored in 16-bit numbers is 14 GB. A GPU with 2 TB/s of memory bandwidth can read that about 140 times per second, so one user cannot get more than roughly 140 tokens/s from it, however fast the arithmetic units are. This formula explains why quantization (smaller weights) and batching (sharing each weight read among many users) are the two biggest levers in LLM inference.

### 3.7 Why Rust for inference

Python is where models are trained and prototyped, and it remains fine for that. The serving path is different: it runs for months, handles thousands of concurrent requests, and every millisecond and megabyte is money. Rust fits that job for concrete reasons:

- **Predictable speed.** No garbage collector pausing the process in the middle of a request, so tail latency stays tight. Compiled code runs at C speed.
- **Memory you can see.** Rust makes you say who owns a buffer and who is only looking at it. For a program whose main job is reading gigabytes of weights, knowing that no code path silently copies them is worth a lot. You will see the cost of one accidental copy in section 5.
- **Safe concurrency.** The compiler refuses to build code with data races. Inference servers are heavily concurrent (many requests, many threads, one shared model), so this matters.
- **Direct access to the hardware.** SIMD instructions, memory-mapped files and C/CUDA libraries are all available without a foreign runtime in between.
- **It is already used here.** Hugging Face's `tokenizers` library, the Text Generation Inference router, `candle`, `mistral.rs` and many production routers and proxies are written in Rust.

The costs are real too: the ecosystem for GPU kernels is younger than C++/CUDA, and the borrow checker takes time to learn. This course leans on the parts of Rust that pay off for inference and explains each one when it first matters.

## 4. The code

The crate is in [`src/lib.rs`](src/lib.rs) (the model and the statistics) and [`src/main.rs`](src/main.rs) (the measurements).

### 4.1 The model

<!-- file: src/lib.rs -->
```rust
#[derive(Clone)]
pub struct LinearModel {
    weights: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}
```

- `weights: Vec<f32>` is one flat, contiguous block of memory holding every weight. There is no `Vec<Vec<f32>>`: a vector of vectors would scatter the rows across the heap, and every row would need its own allocation and pointer. One flat buffer is how every serious inference engine stores tensors (chapter 3 explains the layout).
- Row `o` lives at `weights[o * in_dim .. (o + 1) * in_dim]`. This is **row-major** order.
- `#[derive(Clone)]` lets us copy the model on purpose in part 3 of the demo. Being able to clone it is fine; doing it by accident is the problem.

<!-- file: src/lib.rs -->
```rust
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weights = (0..in_dim * out_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
```

Fake weights between −0.08 and 0.08. They must not all be equal, or a clever compiler (or a reader) might notice shortcuts. Real weights come from a file (chapter 9).

<!-- file: src/lib.rs -->
```rust
    pub fn weight_bytes(&self) -> usize {
        self.weights.len() * size_of::<f32>()
    }

    /// Floating-point operations for one input: one multiply and one add
    /// per weight.
    pub fn flops_per_input(&self) -> usize {
        2 * self.weights.len()
    }
```

These two numbers are the "two budgets" from section 3.6. Keeping them next to the model means every measurement can be turned into GB/s and GFLOP/s, which is how you tell whether code is fast or merely feels fast.

### 4.2 One request

<!-- file: src/lib.rs -->
```rust
    pub fn predict(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim, "input has the wrong length");
        assert_eq!(out.len(), self.out_dim, "output has the wrong length");
        for (row, y) in self.weights.chunks_exact(self.in_dim).zip(out.iter_mut()) {
            *y = dot(row, x);
        }
    }
```

Line by line:

- `&self`: the model is **borrowed**, read-only. Any number of threads could call `predict` on the same model at the same time, and none of them copies a byte of it.
- `x: &[f32]`: the input is a borrowed slice, so the caller keeps ownership of its buffer.
- `out: &mut [f32]`: the caller also provides the output buffer. The function does not allocate. In a server that handles thousands of requests per second, allocation per request adds up, and reusing buffers is standard practice (chapter 8 measures it).
- The two `assert_eq!` lines check the shapes once, up front. A wrong shape is a programming bug, and failing loudly here is much better than reading garbage memory. In C, a mismatched length silently reads past the end of the buffer. In Rust, a slice carries its length, so the check is one comparison.
- `chunks_exact(self.in_dim)` walks the flat weight buffer one row at a time. Each `row` is a `&[f32]` pointing into the model's memory: a view, not a copy.
- `.zip(out.iter_mut())` pairs each row with the output slot it fills. Using iterators instead of `out[o]` indexing means the compiler knows the pairs line up and does not need a bounds check per element.

### 4.3 A batch of requests

<!-- file: src/lib.rs -->
```rust
        for (o, row) in self.weights.chunks_exact(self.in_dim).enumerate() {
            for (b, x) in xs.chunks_exact(self.in_dim).enumerate() {
                out[b * self.out_dim + o] = dot(row, x);
            }
        }
```

The loop order is the whole idea of batching. The **outer** loop walks the weights. The **inner** loop walks the requests. Each 16 KB weight row is fetched from main memory once, lands in the CPU's cache, and is then reused for every request in the batch. With a batch of 16, the expensive memory traffic is shared by 16 requests.

If you swapped the loops (requests outside, weights inside), you would compute the same numbers but re-read all 67 MB of weights for every request, and batching would buy almost nothing. **Same arithmetic, different memory traffic, very different speed.** You will see this pattern again in every chapter.

### 4.4 The dot product

<!-- file: src/lib.rs -->
```rust
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for (x, y) in a8.iter().zip(b8) {
        for lane in 0..8 {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        total += x * y;
    }
    total
}
```

This looks more complicated than `a.iter().zip(b).map(|(x, y)| x * y).sum()`, and it is. On the reference machine it is about five times faster (4.8 ms versus 25 ms for the whole layer). The short version:

- `as_chunks::<8>()` splits the slice into groups of exactly eight (`&[[f32; 8]]`) plus a leftover tail of fewer than eight.
- Eight separate running sums let the processor work on eight independent multiply-adds at once. With one sum, each addition must wait for the one before it to finish.
- The leftovers are handled one at a time at the end.

Chapter 6 takes this apart properly (why one sum is slow, and how the CPU's vector instructions work). For now, treat `dot` as a fast building block.

### 4.5 The mistake, written down on purpose

<!-- file: src/lib.rs -->
```rust
#[expect(
    clippy::needless_pass_by_value,
    reason = "this function exists to demonstrate the cost of taking ownership"
)]
pub fn predict_with_owned_model(model: LinearModel, x: &[f32], out: &mut [f32]) {
    model.predict(x, out);
    // `model` is dropped here: its weights are freed at the end of every call.
}
```

This function takes `model: LinearModel` **by value**. In Rust, passing by value *moves* ownership into the function. After the call, the caller no longer has the model. So a caller who wants to keep serving requests has only one option:

```rust
predict_with_owned_model(model.clone(), &x, &mut y); // copies all 67 MB
```

That `.clone()` is written out in the code, where a reviewer can see it. Compare this with languages where passing an object may or may not copy it depending on rules you have to remember. In Rust, an expensive copy is always spelled `.clone()`.

Clippy (Rust's linter) spots this pattern too: it warns that the argument "is passed by value, but not consumed in the function body". The `#[expect(...)]` attribute tells Clippy we know and why. We use `expect` rather than `allow` because `expect` fails the build if the warning ever stops firing, so the exemption cannot quietly outlive its reason.

### 4.6 Percentiles

<!-- file: src/lib.rs -->
```rust
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty());
    assert!((0.0..=100.0).contains(&p));
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}
```

The "nearest rank" percentile: sort the samples, then pick the sample at position ⌈p/100 × n⌉ (counting from 1). With 200 samples, p99 is the 198th fastest. There are several definitions of percentile in use (some interpolate between samples). Which one you use matters less than using the same one everywhere and saying which it is.

`rank.clamp(1, len)` handles `p = 0` (rank 0 would underflow when we subtract 1) and guards against floating-point rounding pushing the rank past the end.

### 4.7 The measurements

<!-- file: src/main.rs -->
```rust
    // Warm-up: the first calls pay for page faults and cold caches.
    for _ in 0..10 {
        model.predict(black_box(&x), &mut y);
    }

    let mut samples = Vec::with_capacity(200);
    for _ in 0..200 {
        let start = Instant::now();
        model.predict(black_box(&x), &mut y);
        black_box(&y);
        samples.push(start.elapsed());
    }
```

Three habits that you will use in every benchmark in this course:

1. **Warm up first.** The first run of anything pays one-time costs: the operating system mapping memory pages in, caches starting empty, the CPU raising its clock speed. Those costs are real (they show up in production as slow first requests, which is why servers "warm up" models at startup), but they are a separate question from steady-state speed.
2. **`std::hint::black_box`.** The optimizer is allowed to delete work whose result is never used. `black_box(&y)` tells it "assume something reads this", and `black_box(&x)` stops it from assuming it knows the input. Without them, a benchmark can end up measuring an empty loop.
3. **Keep every sample, not just the total.** You cannot compute p99 from an average.

## 5. Run it

```bash
cargo run --release -p ch01-what-is-inference
```

Always use `--release` for anything you time. Debug builds skip most optimizations and are often 10-50x slower on numeric code.

Output on the course's reference machine (a 4-core Intel Xeon cloud VM, see the [course README](../../README.md#the-reference-machine)). Your numbers will differ. The *shape* of the results is what matters.

```text
model: 4096 x 4096 f32 weights = 67.1 MB

== 1. latency of one request (200 samples)
   mean 3.935166ms  p50 3.788123ms  p90 4.59543ms  p99 5.826807ms  max 6.831683ms
   at p50: 17.7 GB/s of weights read, 8.86 GFLOP/s

== 2. batching
   batch | time per batch | requests/s | GFLOP/s
       1 |         3.86ms |        259 |    8.69
       2 |         6.03ms |        332 |   11.12
       4 |        10.76ms |        372 |   12.47
       8 |        19.70ms |        406 |   13.62
      16 |        31.41ms |        509 |   17.09
      32 |        65.14ms |        491 |   16.48
      64 |       123.90ms |        517 |   17.33

== 3. borrowing vs cloning the weights
   borrow (&model):       4.65ms per request
   clone  (model.clone()): 61.70ms per request
   cloning costs 13.3x the time, plus 67.1 MB of extra memory per in-flight request
```

What these numbers say:

**Part 1.** The median request takes 3.8 ms, but p99 is 5.8 ms and the worst is 6.8 ms. Nothing in the code changed between runs; the spread comes from the machine (other processes, other tenants on the same physical host, interrupts, cache contents). Real services see the same effect, only larger. Reading 67 MB in 3.8 ms is 17.7 GB/s, and the arithmetic rate is 8.9 GFLOP/s. The second number is low for this CPU, which is the first sign that a single request is limited by data movement rather than arithmetic.

**Part 2.** Batching doubles throughput, from 259 to about 510 requests per second. The **cost** is latency: at batch 16 every request in the batch waits 31 ms instead of 4 ms. That is the latency/throughput trade-off in its simplest form.

Notice that throughput stops improving after batch 16, at about 17 GFLOP/s. We have moved from waiting on memory to waiting on arithmetic: one core, running code compiled for a generic x86-64 CPU (without the wider vector instructions this chip has), cannot multiply faster than that. Chapters 5-7 raise that ceiling with better loop structure, explicit vector instructions and more cores, and the batching gain grows accordingly.

**Part 3.** Cloning the model per request makes it 13 times slower and uses an extra 67 MB for every request in flight. The copy itself (allocating 67 MB, the OS supplying fresh zeroed pages, then copying) costs more than the actual computation. With 100 concurrent requests the clones alone would need 6.7 GB. Borrowing costs nothing.

## 6. The Rust behind it

**Borrowing is how you share weights.** A model's weights are the largest thing in an inference process and they never change after loading. The natural Rust shape is: one owner (the model struct, or later an `Arc<Model>` shared between threads), and every request borrows it with `&`. A `&T` is a pointer plus a compile-time guarantee that nobody is mutating the data while you read it. It costs the same as a C pointer at runtime.

**Moves are free, clones are visible.** Passing a `Vec` by value moves three machine words (pointer, length, capacity), not its contents. Copying the contents takes an explicit `.clone()`. So "does this line copy the weights?" is a question you can answer by reading the code, not by knowing language rules.

**Slices carry their length.** `&[f32]` is a pointer plus a length. That is why `chunks_exact`, `as_chunks` and `zip` can hand out sub-views of the weight buffer without copying and without risking out-of-bounds reads.

**Caller-provided output buffers.** `predict(&self, x: &[f32], out: &mut [f32])` rather than `fn predict(&self, x: &[f32]) -> Vec<f32>`. Returning a `Vec` is more convenient and allocates on every call. In an inference hot loop, prefer the `&mut [f32]` form. You will see this signature throughout the course.

## 7. Mistakes you will make

- **Timing a debug build.** Debug numbers are meaningless for performance. Every timing command in this course has `--release`.
- **Reporting the mean.** Always report at least p50 and p99, and say how many samples.
- **Benchmarking code that got optimized away.** If a change makes something "infinitely fast", suspect the optimizer before celebrating. Use `black_box`.
- **Forgetting warm-up.** The first iteration of a benchmark is often 2-10x slower than the rest.
- **Comparing numbers from a noisy machine.** On a shared cloud VM, run each measurement a few times. Differences under about 5-10% are often noise.

## 8. How the professionals do it

- Production LLM servers (vLLM, SGLang, TensorRT-LLM, Hugging Face TGI, llama.cpp's server) all report TTFT, TPOT (also called ITL, inter-token latency), end-to-end latency percentiles, and throughput in tokens/s. You will build the same measurements in chapter 25.
- Teams set targets called **SLOs** (service level objectives), for example "p99 TTFT under 500 ms and p99 TPOT under 50 ms". Throughput is then maximized *subject to* the SLO. Throughput that breaks the SLO does not count; the throughput that does count is called **goodput**.
- Benchmarks are run on dedicated machines with fixed CPU frequencies where possible, repeated, and reported with variance. Chapter 17 is about doing this properly.

## 9. Exercises

1. **Swap the batch loops.** Change `predict_batch` so the requests are the outer loop and the weight rows are the inner loop. Predict what happens to the batching table before you run it, then run it and explain the result.
2. **Shrink the model.** Set `IN_DIM` and `OUT_DIM` to 512 (1 MB of weights). Run again. What happens to GB/s at batch 1, and why? (Hint: where do 1 MB of weights live after the first request?)
3. **Naive dot product.** Replace the body of `dot` with `a.iter().zip(b).map(|(x, y)| x * y).sum()`. Measure the new batch-1 latency. By what factor did it change?
4. **The tail.** Increase the latency samples from 200 to 5,000. Does p99 change? Does max? Which one is more stable between runs, and why?
5. **Cost of the clone alone.** Write a benchmark that only does `let m = model.clone(); black_box(&m);`. What fraction of the 61 ms in part 3 is the clone?

## 10. Check yourself

1. In one sentence each, what are prefill and decode, and which one is limited by memory bandwidth?
2. Why is the average latency a bad summary of a service's performance?
3. A 1-billion-parameter model is stored in 16-bit numbers. The machine has 50 GB/s of memory bandwidth. Roughly how many tokens per second can one user get at most?
4. Why does reordering loops change speed without changing the result?
5. What does `black_box` protect you from?
6. Why does `predict` take `out: &mut [f32]` instead of returning a `Vec<f32>`?

## 11. Recap

- Inference is running a trained model; it runs forever and at scale, so it dominates cost.
- A language model generates one token per forward pass. Prefill (reading the prompt) is limited by arithmetic, decode (writing the answer) by memory bandwidth.
- Describe performance with latency percentiles, throughput, cost and memory, never with a single average.
- For single-user decode, time per token ≈ weight bytes ÷ memory bandwidth.
- Batching shares each weight read among many requests: higher throughput, higher latency per request.
- In Rust, sharing weights by `&` borrow is free and copying them requires a visible `.clone()`. The copy you avoid is often worth more than any optimization.

## Answers

**Exercises**

1. Throughput stays flat at every batch size. Measured on the reference machine: 248, 265 and 224 requests/s at batch 4, 16 and 64, against 509 and 517 with the right loop order. Every request now re-reads all 67 MB of weights, so the batch saves nothing. Same arithmetic, many times the memory traffic.
2. With 1 MB of weights, everything fits in the CPU's L2 cache (2 MB per core on this machine) after the first call. Measured: 31 µs per request, 33.7 GB/s, 16.9 GFLOP/s. The layer is now limited by arithmetic already at batch 1, so batching gains nothing (about 17 GFLOP/s at batch 16 and 64 too). Small models behave very differently from large ones, which is why you must benchmark at realistic sizes.
3. About 5x slower: 25.4 ms against 4.8 ms on the reference machine. The single running sum forms a dependency chain, so each addition waits for the previous one to finish. Chapter 6 measures this in detail.
4. p99 moves a little and settles as you add samples (6.7 ms with 5,000 samples in one run on the reference machine). Max keeps growing: the same run hit 92 ms once, a stall caused by something outside the program. With more samples you are more likely to catch a rare event. That is why max is useful for spotting problems but a poor target.
5. On the reference machine the clone alone takes 53 ms, about 86% of the 61 ms. The model computation is the small part.

**Check yourself**

1. Prefill processes the whole prompt in one pass and is limited by arithmetic; decode generates one token per pass and is limited by memory bandwidth.
2. A few very slow requests can hide in an average, and an average does not describe any real user's experience. Percentiles describe the typical case and the tail separately.
3. Weights = 1e9 × 2 bytes = 2 GB. 50 GB/s ÷ 2 GB = 25 tokens/s at most.
4. The CPU fetches memory in cache-sized pieces and keeps recently used data close. Loop order decides how many times each piece has to come from slow memory.
5. From the compiler removing or simplifying the work being measured because it can prove the result is unused or the input is constant.
6. So the caller can reuse one buffer across many requests instead of allocating per call. Allocation in a hot loop costs time and fragments memory.

## Further reading

- Jeff Dean's "Latency numbers every programmer should know" (many updated versions online). Chapter 4 measures these on real hardware.
- "The Tail at Scale", Dean and Barroso, Communications of the ACM, 2013. Why p99 matters.
- Next: [Chapter 2: Numbers inside a model](../02-numbers/README.md). Before speeding up the arithmetic, we look at what the numbers themselves are made of.
