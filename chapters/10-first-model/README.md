# Chapter 10: A first model, end to end

> **In one sentence:** serving a model means loading its weights safely, running an allocation-free forward pass, turning outputs into answers, and choosing how to spread requests over the hardware, and each of those choices trades latency against throughput.

**Where this fits:** this chapter joins everything so far into one working system: chapter 9's loader, chapters 5-7's parallel matmul, chapter 8's operators, and chapter 1's latency statistics. The model is deliberately small (a classifier, not a language model) so that every piece is visible. Part IV then replaces it with a transformer, and Part VI grows the serving side into a real server.

**You need:** chapters 1-9.

**You will build:** a multi-layer perceptron (MLP) with its own training loop, saved to and loaded from safetensors with shape validation; a batched, parallel forward pass that reuses its buffers; accuracy measurement on held-out data; and a comparison of four ways to serve a 25-million-parameter model to 256 requests.

---

## 1. The intuition

Picture a small bakery that has perfected one recipe (training is done) and now has to sell bread all day (inference).

- **Getting the recipe out of the safe.** The recipe card (the weight file) is checked before anyone bakes from it: right number of steps, quantities that make sense. A card that says "add 10 tonnes of flour" is rejected, not attempted.
- **Keeping bowls on the counter.** The bakers do not buy new bowls for every loaf (allocation); they wash and reuse the same ones (workspace buffers).
- **How to use four bakers.** Four bakers can each make a separate loaf at the same time (request-level parallelism), or all four can work on one loaf at once so it is finished faster (operator-level parallelism), or they can wait until several orders arrive and make a whole tray together (batching). The first gives the most loaves per hour when orders come steadily; the second gives the fastest single loaf; the third gives the most loaves per hour of all but makes early customers wait for the tray.

**Where the analogy breaks:** a tray of 64 loaves takes a real oven roughly as long as a tray of 4. A batched matmul is not quite that generous: it is nearly free while it is memory-bound, then it becomes compute-bound and the time starts growing with the batch (chapter 4's roofline). Section 5 shows where that happens for this model.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **MLP** | Multi-layer perceptron: linear layers separated by non-linear activations. |
| **Logits** | The raw scores the last layer produces, one per class. |
| **Argmax** | The index of the largest value: the predicted class. |
| **Held-out set** | Data the model never saw during training, used to measure real accuracy. |
| **Workspace** | Buffers allocated once and reused by every forward pass. |
| **Operator-level parallelism** | Splitting each operation of one request across cores. |
| **Request-level parallelism** | Running different requests on different cores at the same time. |
| **Batching** | Running several requests through the model together, as one matrix. |
| **Throughput** | Requests completed per second. |
| **Latency** | Time for one request, from arrival to answer. |

## 3. The concepts in depth

### 3.1 The model

The task: points in the plane lie on one of three interleaved spiral arms, and the model must say which arm a point belongs to. No straight line separates the arms, so a single linear layer cannot solve it; a few layers with non-linear activations can.

```text
input (x, y) ──► Linear 2→64 ──► ReLU ──► Linear 64→64 ──► ReLU ──► Linear 64→3 ──► logits ──► softmax ──► probabilities
```

4,547 parameters in total: 128 + 64 weights and biases for the first layer, 4,096 + 64 for the second, 192 + 3 for the third. Small enough to train in seconds on one core, and big enough to be a real neural network with every part that matters for inference.

### 3.2 Training, in one paragraph

This course is about inference, but real weights have to come from somewhere, so `src/train.rs` implements the minimum: compute the model's predictions for all 900 training points, measure how wrong they are with the cross-entropy loss (the negative log-probability of the correct class), compute how each weight should change to reduce that loss (backpropagation: the chain rule applied layer by layer, from the output back to the input), and move every weight a small step in that direction. Repeat 2,000 times. After training, the weights are fixed forever; everything else in this chapter only reads them. Inference engines never contain any of the code in `train.rs`.

### 3.3 Accuracy is part of the contract

A fast model that gives wrong answers is worthless, and the only honest way to know it gives right answers is to test on data it has not seen. The training binary reports 99.8% on the training points and 99.3% on 300 fresh points generated with a different random seed. The serving binary re-measures the held-out accuracy after loading, which checks the whole chain: the file was written correctly, read correctly, and the inference code computes the same function the training code did.

This check is the small version of something every inference team runs constantly: after any change to kernels, formats or quantization, re-run an accuracy evaluation and compare. Chapter 18 does it with perplexity for a language model.

### 3.4 Loading: validate, then copy into owned buffers

The file format is chapter 9's safetensors. The loader:

1. Reads the number of layers from the file's metadata.
2. For each layer, fetches `layers.{i}.weight` and `layers.{i}.bias`, checks that the weight is 2-D, that the bias length equals the weight's row count, and that the layer's input size equals the previous layer's output size.
3. Copies the weights out of the memory map into 64-byte-aligned buffers (chapter 6's `AlignedVec`).

Step 2 catches the classic "loaded the wrong file" and "file from a different version of the model" failures with a clear message, before any computation. Step 3 is chapter 9's "owned copies" design: at 100 KB of weights, copying is free, the model needs no lifetime parameter, and the alignment is guaranteed.

### 3.5 The forward pass and its workspace

Each layer computes `Y = X·Wᵀ + b` for all requests in the batch, then applies ReLU (except after the last layer, whose outputs are logits). The activations flow through two buffers owned by a `Workspace`: layer 1 reads buffer A and writes buffer B, layer 2 reads B and writes A, and so on ("ping-pong"). Both buffers are sized once for the widest layer at the largest batch the caller will use.

After that, a forward pass allocates no activation memory, however many requests it serves. Each serving thread owns its own workspace, while all threads share the one read-only model. That split (shared immutable weights, per-request mutable scratch) is the basic memory design of every inference engine, and chapters 14 and 24 extend it to the KV cache.

### 3.6 Four ways to use four cores

For serving, the question is how to map requests onto cores. Part 2 of the demo uses a bigger model (1024 → 4096 → 4096 → 1000, 25 million parameters, 100 MB of `f32` weights) and serves 256 requests four ways:

- **A. One thread, one request at a time.** The baseline.
- **B. Operator-level parallelism.** One request at a time, but every matmul is split across 4 threads with chapter 7's spin pool. Each request finishes as fast as possible.
- **C. Request-level parallelism.** 4 threads, each serving its own requests one at a time, each with a single-threaded pool and its own workspace, all sharing one model.
- **D. Batching.** Requests are grouped into batches of 4, 16, 64 or 256 and each batch goes through the model as one matrix, with every matmul split across 4 threads.

## 4. The code

The model, loader and forward pass are in [`src/lib.rs`](src/lib.rs), training in [`src/train.rs`](src/train.rs), and the two programs in [`src/bin/train.rs`](src/bin/train.rs) and [`src/main.rs`](src/main.rs).

### 4.1 Layers and model

<!-- file: src/lib.rs -->
```rust
#[derive(Clone)]
pub struct Linear {
    pub weight: AlignedVec<f32>,
    pub bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}
```

The weight is `[out × in]`, one row per output: the PyTorch layout and chapter 5's NT form, so every output is a dot product of two contiguous rows.

<!-- file: src/lib.rs -->
```rust
    pub fn random(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        Self {
            weight: AlignedVec::from_fn(in_dim * out_dim, |_| rng.normal() * scale),
            bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }
```

Initial weights are random, scaled by √(2/in). A ReLU layer with that scale keeps the size of its outputs roughly equal to the size of its inputs ("He initialization"), so the signal neither explodes nor vanishes through the layers. The same scale makes the random 25M-parameter model in part 2 produce numbers of a sensible size, which matters for honest timing: arithmetic on subnormal floats can be much slower (chapter 2).

### 4.2 The forward pass

<!-- file: src/lib.rs -->
```rust
        let Workspace { a, b, .. } = ws;
        a[..x.len()].copy_from_slice(x);
        let (mut input, mut output) = (a, b);
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let x_in = &input[..batch * layer.in_dim];
            let y = &mut output[..batch * layer.out_dim];
            matmul_nt_pool(
                pool,
                x_in,
                &layer.weight,
                y,
                batch,
                layer.in_dim,
                layer.out_dim,
            );
            for row in y.chunks_exact_mut(layer.out_dim) {
                for (v, b) in row.iter_mut().zip(&layer.bias) {
                    *v += b;
                }
                if i != last {
                    relu(row);
                }
            }
            std::mem::swap(&mut input, &mut output);
        }
        &input[..batch * self.output_dim()]
```

Line by line:

- `let Workspace { a, b, .. } = ws;` **destructures** the `&mut Workspace` into two separate `&mut Vec<f32>` borrows, one per field. The borrow checker understands that different fields of a struct are different memory, so holding `&mut a` and `&mut b` at the same time is allowed. (Calling two `&mut self` methods to get them would not be.)
- `input` and `output` are those two mutable references. `std::mem::swap(&mut input, &mut output)` swaps the *references*, not the buffers: after each layer, the buffer just written becomes the next layer's input. No data is copied.
- `&input[..batch * layer.in_dim]` takes only the part of the buffer this layer uses; the buffers are sized for the widest layer.
- `matmul_nt_pool` is chapter 7's parallel NT product. Passing `pool` in (rather than creating threads inside) lets the caller decide how many threads each forward pass uses, which is exactly what part 2 varies.
- The bias add and ReLU run row by row over the batch. `relu` is chapter 8's in-place operator.
- The return type `&'w [f32]` ties the logits to the workspace's lifetime. The caller reads them before the next forward pass overwrites them, and the borrow checker enforces that: calling `forward` again while still holding the previous logits does not compile, because both need `&mut ws`.

An honest note: the *activations* never allocate, but chapter 7's `matmul_nt_pool` still allocates a small list of chunks per call and, for batches, a transposed temporary buffer. Chapter 14's engine removes both. Finding allocations like these, which hide inside library functions, is what an allocation profiler (chapter 17) is for.

### 4.3 Loading with shape checks

<!-- file: src/lib.rs -->
```rust
            let [out_dim, in_dim] = w.shape[..] else {
                return Err(LoadError::Shape(format!("layer {i} weight is not 2-D")));
            };
            if b.shape != [out_dim] {
                return Err(LoadError::Shape(format!(
                    "layer {i} bias has shape {:?}, expected [{out_dim}]",
                    b.shape
                )));
            }
            if let Some(prev) = layers.last()
                && prev.out_dim != in_dim
            {
```

- `let [out_dim, in_dim] = w.shape[..] else { ... }` destructures a slice of exactly two elements and rejects anything else in one statement.
- `if let Some(prev) = layers.last() && prev.out_dim != in_dim` is a **let chain** (stable since Rust 1.88): "if there is a previous layer, and its output size differs from this layer's input size". It reads like the sentence.
- Every failure becomes a `LoadError`, an enum that wraps I/O errors, format errors (chapter 9's `Error`) and shape errors. The `From` impls let `?` convert the lower-level errors automatically, so the loader body is mostly `?`s and checks.

### 4.4 Serving with shared weights

<!-- file: src/main.rs -->
```rust
    let mut lat: Vec<Duration> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..cores)
            .map(|t| {
                let model = &model;
                let mine = &inputs[t * per_thread * dim..(t + 1) * per_thread * dim];
                s.spawn(move || {
                    let mut pool = SpinPool::new(1);
                    let mut ws = Workspace::new(model, 1);
                    one_at_a_time(model, &mut pool, &mut ws, mine, dim).1
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker"))
            .collect()
    });
```

Four threads, one shared model, four private workspaces. Because these are scoped threads (chapter 7), each can borrow the model with a plain `&Mlp`: no `Arc`, no copy of the 100 MB of weights. This compiles only because `Mlp` is `Sync` (every field is: `AlignedVec<f32>` declared it in chapter 6, `Vec` and `usize` are automatically). A real server's request handlers are not scoped (they live as long as the server), and there the shared model goes in an `Arc<Mlp>`, which chapter 21 does.

Each thread's `SpinPool::new(1)` has no workers at all: `run` simply calls the job on the current thread. So design C is truly single-threaded per request.

## 5. Run it

```bash
cargo run --release -p ch10-first-model --bin train   # once: trains and saves the model
cargo run --release -p ch10-first-model
cargo test -p ch10-first-model
```

Training, on the reference machine:

```text
training a 2 -> 64 -> 64 -> 3 MLP (4547 parameters) on 900 points
   step    0  loss 1.2453
   step  250  loss 0.0176
   ...
   step 1999  loss 0.0067
trained in 10.1s
accuracy: training set 99.8%, held-out test set 99.3%
saved .../models/spiral-mlp.safetensors (18700 bytes)
```

Serving:

```text
== 1. the trained spiral classifier
   loaded .../models/spiral-mlp.safetensors in 95.10µs
   layers [2, 64, 64, 3], 4547 parameters
   accuracy on 300 held-out points: 99.3%
   point [0.0, 0.0] -> class 2 with probabilities [0.337, 0.326, 0.338]
   point [0.5, 0.1] -> class 2 with probabilities [0.000, 0.000, 1.000]
   point [-0.3, 0.6] -> class 0 with probabilities [0.999, 0.000, 0.001]
   point [0.2, -0.7] -> class 1 with probabilities [0.000, 1.000, 0.000]

== 2. serving a 1024 -> 4096 -> 4096 -> 1000 MLP (25.1 M parameters, 100 MB) to 256 requests
   design                               | total time | requests/s | per-request latency p50 / p99
   one thread, one request at a time    |       1.8s |        141 |    7.26ms /   11.01ms
   4 threads split each request         |    393.1ms |        651 |    1.38ms /    4.23ms
   4 threads, one request each          |    271.2ms |        944 |    4.07ms /    6.46ms
   4 threads, batches of 4              |    169.0ms |       1515 |    2.27ms /    9.93ms
   4 threads, batches of 16             |    145.9ms |       1755 |    8.91ms /   10.08ms
   4 threads, batches of 64             |    139.9ms |       1830 |   32.52ms /   40.84ms
   4 threads, batches of 256            |    205.5ms |       1245 |  205.54ms /  205.54ms
```

Reading part 1:

- **The held-out accuracy after loading matches the accuracy after training** (99.3%): the save/load round trip is lossless and the inference path computes the same function.
- **The point (0, 0) gets about 1/3 for every class.** All three spirals start at the origin, so the model is genuinely unsure, and its probabilities say so. Reporting probabilities, not only the argmax, lets callers tell confident answers from guesses.

Reading part 2, which is the real lesson of this chapter:

- **This VM is noisy**: across four runs, B ranged from 494 to 651 requests/s and C from 571 to 944. The ranking of the designs held in every run; the exact ratios did not.
- **Splitting each request across 4 cores (B) gives the lowest latency**: 1.38 ms median, 5.3x faster than one thread. The gain is more than 4x because four cores also bring four L2 caches (chapter 7, section 3.4).
- **Giving each core its own request (C) gives 45% more throughput than B** (944 against 651 requests/s), but each request takes 3x longer (4.07 ms median). No time is spent coordinating threads; but each core reads all 100 MB of weights for its own request, so the four cores compete for memory bandwidth.
- **Batching (D) gives the most throughput by far**: 1,830 requests/s at batch 64, 13x the baseline. Each weight row is fetched once per batch instead of once per request (chapter 1's insight, chapter 4's roofline).
- **Batching costs latency.** At batch 64, a request's answer takes 33 ms, because it cannot come back before the whole batch is done. And these numbers ignore the time a request spends *waiting for a batch to fill*, which in a real server can be much longer (chapter 23).
- **Throughput stops improving after batch 64.** In this run batch 256 was even slower (1,245 requests/s). Over four runs it ranged from 1,267 to 1,820 requests/s against 1,518-2,137 for batch 64: no better, sometimes worse, and much noisier, since each batch-256 run is a single forward pass. Once the layers are compute-bound, a bigger batch only adds latency. At 256 rows the input activations (4 MB per layer) also stop fitting in L2, which chapter 7's matmul was not blocked for (exercise 4).

There is no single best design. B minimizes latency, D maximizes throughput, and C sits in between with no batching delay. A real serving system picks per workload, and Part VI builds the machinery (continuous batching, scheduling) that gets most of D's throughput with most of B's latency.

## 6. The Rust behind it

**Destructuring borrows.** `let Workspace { a, b, .. } = ws;` splits one `&mut Workspace` into disjoint `&mut` borrows of its fields. Combined with `std::mem::swap` on the references, it gives zero-copy ping-pong buffers with no `unsafe` and no index juggling.

**Output lifetimes tied to a workspace.** `fn forward<'w>(..., ws: &'w mut Workspace) -> &'w [f32]` returns a view into the workspace instead of a new `Vec`. The borrow checker then guarantees the caller is done with the logits before the next call overwrites them.

**Error enums with `From`.** `LoadError` wraps the lower-level errors, and `impl From<io::Error> for LoadError` is what lets `?` convert them. For binaries, crates such as `anyhow` make this less verbose; for libraries, a typed error enum like this one tells callers exactly what can go wrong.

**Let chains and slice patterns.** `if let Some(prev) = layers.last() && prev.out_dim != in_dim` and `let [out_dim, in_dim] = w.shape[..] else { ... }` express validation logic directly, without nested `if`s.

**`Sync` for shared models.** A model shared by serving threads must be `Sync`. That is automatic when every field is `Sync`, and a compile error when one is not (for example if someone added a `Cell` or `Rc` field for caching). The compiler checks the thread-safety of the whole serving design.

**`default-run` in `Cargo.toml`.** With two binaries in one package (`train` and the default), `default-run` picks which one `cargo run -p ch10-first-model` runs.

## 7. Mistakes you will make

- **Measuring accuracy on the training data only.** A model can memorize its training set. Always keep a held-out set.
- **Skipping shape validation when loading.** A file for a slightly different model loads "successfully" and produces garbage, or panics deep in a kernel with an unhelpful index error.
- **Allocating per request** (a new `Vec` for every activation). Reuse a workspace per thread.
- **Comparing serving designs by throughput alone** (or by latency alone). Report both, at the same load.
- **Forgetting the wait for a batch to fill.** A batch-64 design only reaches 1,830 requests/s if 64 requests are available at once; at low load it adds waiting time that the table above does not show.
- **Sharing a workspace between threads.** It will not compile (it is borrowed `&mut`), which is the compiler catching a data race for you.

## 8. How the professionals do it

- **Model servers** such as NVIDIA Triton, TorchServe and TensorFlow Serving offer both request-level parallelism ("model instances") and **dynamic batching** (collect requests for up to a few milliseconds, then run them as one batch), configured per model, precisely because of the trade-off in part 2.
- **LLM servers** (vLLM, TGI, SGLang) use **continuous batching** (chapter 23), which re-forms the batch at every generation step instead of waiting for whole requests.
- **Validation at load time** is universal: PyTorch's `load_state_dict(strict=True)` checks that every expected tensor is present with the right shape; Hugging Face transformers warns about missing and unexpected weights.
- **Accuracy regression tests** run in continuous integration for any change to a model's numerics: kernels, quantization, compiler flags.

## 9. Exercises

1. **A smaller network.** Train `[2, 16, 3]` instead of `[2, 64, 64, 3]`. What accuracy does it reach? How many parameters does it have?
2. **Break the loader.** Write a file with `layers.1.weight` of shape `[3, 32]` after a first layer that outputs 64, and check that `Mlp::load` reports it clearly.
3. **A latency target.** Suppose you must serve with p99 latency under 12 ms. Which of the designs in part 2 qualify, and which of those has the highest throughput?
4. **Fix batch 256.** Change `matmul_nt_pool_with` (chapter 7) so that, for large `m`, it processes input rows in blocks that fit in L2 (for example 32 rows at a time) inside each weight group. Does batch 256 now beat batch 64?
5. **Probabilities as a signal.** Generate 1,000 random points in the square [-1, 1]². How many does the model classify with a top probability below 0.6? Where are they?
6. **Compare with no pool.** Replace `SpinPool::new(1)` in design C with a direct single-threaded loop over `dot`. Is there any difference? Why or why not?

## 10. Check yourself

1. Why is accuracy on a held-out set a better check than accuracy on the training set?
2. What does the loader check about each layer, and what failure does each check catch?
3. Why does a workspace make the forward pass allocation-free, and why does each serving thread need its own?
4. Why does batching increase throughput? Why does it increase latency?
5. Why did request-level parallelism (C) give more throughput but higher latency than operator-level parallelism (B)?
6. How can four threads share one model without `Arc` in this chapter's demo, and why would a real server need `Arc`?

## 11. Recap

- A complete inference path: validated loading, an allocation-free forward pass with reused workspaces, post-processing (softmax, argmax), and an accuracy check against held-out data.
- Weights are shared and read-only; workspaces are per thread and mutable. That split is the memory design of every inference engine.
- For 256 requests to a 25M-parameter model on 4 cores: splitting each request gives the lowest latency (1.4 ms), one request per core gives more throughput (944/s), batching gives the most (1,830/s) at the highest latency (33 ms at batch 64).
- Kernels tuned for small batches can collapse at large ones; measure across the whole range you will serve.
- In Rust: destructuring gives disjoint `&mut` field borrows, output lifetimes tie results to workspaces, and `Sync` makes the compiler check that a shared model is safe to share.

## Answers

**Exercises**

1. `[2, 16, 3]` has 2×16 + 16 + 16×3 + 3 = 99 parameters. Measured (same data, same 2,000 steps): 99.8% on the training set and 99.3% on the held-out set, exactly as good as the 4,547-parameter model. Before measuring, I expected it to do clearly worse; it does not, because this dataset is simpler than it looks. The larger model was 46 times bigger than it needed to be. Finding the smallest model (or the lowest precision, chapters 18-19) that keeps quality is one of the most valuable things an inference engineer does, and it can only be done by measuring.
2. `Mlp::load` returns `LoadError::Shape("layer 1 takes 32 inputs but layer 0 produces 64")`. The test `load_rejects_a_broken_layer_chain` builds exactly this kind of file.
3. From the table: A (p99 11.0 ms), B (4.2 ms), C (6.5 ms), D with batch 4 (9.9 ms) and batch 16 (10.1 ms) all meet a 12 ms p99; batch 64 (40.8 ms) and 256 do not. Of those that qualify, batch 16 has the highest throughput (1,755 requests/s). This is the shape of every real serving decision: maximize throughput subject to a latency target, which chapter 25 formalizes as an SLO.
4. Inside each thread's group of weight rows, loop over input rows in blocks (say 16 or 32 rows) so both the weight group and the input block stay in L2. Measured in one session on the reference machine: batch 256 went from 1,607 to 1,642 requests/s with blocks of 32 rows and to 1,728 with blocks of 16, a 7% gain, while batch 64 was unchanged (about 1,860). So blocking helps a little at large batches, but the main effect is simply that throughput has already reached its plateau by batch 64: the layers are compute-bound, and no reordering of memory accesses changes the arithmetic.
5. Measured: 16 of 1,000 random points had a top probability below 0.6. Only one was near the origin (radius under 0.3) and one outside the unit circle; most were at radii between 0.6 and 0.85, on the boundaries *between* neighbouring arms, where the decision boundary itself runs. (I had expected them to cluster at the origin; the model instead learned sharp boundaries everywhere except between arms.) Note also that points far outside the training data are often classified *confidently*, not uncertainly: a model's confidence says nothing about inputs unlike anything it was trained on, a point that matters for any deployed classifier.
6. No measurable difference: with one thread, `SpinPool::run` calls the job directly on the current thread, so it is already a plain single-threaded loop around the same `dot`. The pool costs one function call and one branch per layer.

**Check yourself**

1. A model can memorize its training points and still fail on new ones. Held-out data measures what users will actually see.
2. That each weight is 2-D (catches the wrong tensor under the right name), that each bias length equals its layer's output count (catches mismatched pairs), and that each layer's input size equals the previous layer's output size (catches files for a different architecture). Plus everything chapter 9 checks about the file itself.
3. It holds the activation buffers, allocated once and reused on every call, so the per-request path does not allocate. Each thread needs its own because the buffers are written during the forward pass; sharing them would be a data race, which the borrow checker forbids.
4. Throughput: each weight is read from memory once per batch instead of once per request, so memory-bound layers do up to B requests' work for the price of one read. Latency: every request in a batch waits until the whole batch is finished, plus any time spent waiting for the batch to fill.
5. In C, four requests run at once, each on one core with no coordination, so total work per second is higher; but each core streams all 100 MB of weights for its request, and the cores share memory bandwidth, so each request is slower than when four cores work on it together (B), which also coordinates at every layer.
6. Scoped threads are guaranteed to finish before the scope ends, so they can borrow `&model` from `main`. A server's request handlers outlive any particular scope, so the model must be owned by something they all share: `Arc<Mlp>`, which keeps it alive until the last handler drops its reference.

## Further reading

- Goodfellow, Bengio and Courville, *Deep Learning*, chapter 6 (feedforward networks and backpropagation), for the training side this chapter skips over.
- The NVIDIA Triton Inference Server documentation on "dynamic batching" and "instance groups": a production version of part 2's designs.
- Next: [Chapter 11: Tokenization](../11-tokenization/README.md). Language models do not read text; they read token IDs. We build the component that turns one into the other.
