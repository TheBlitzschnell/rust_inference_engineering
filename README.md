# Inference Engineering in Rust

A course that teaches inference engineering from the ground up, in Rust. You start with what a model is and how numbers are stored. You finish with a from-scratch engine that runs a real pretrained language model (SmolLM2) and serves it over HTTP with streaming, continuous batching, a paged KV cache, quantized weights and speculative decoding. Every step is explained in plain language, with complete code and measurements.

**Who it is for:** programmers who know basic Rust and want to understand how models are actually run in production, and why engines like vLLM, llama.cpp and TensorRT-LLM are built the way they are. You do not need a machine learning background. Rust basics are not taught, but every Rust feature that matters for inference (borrowing to avoid copies, lifetimes on memory-mapped weights, `unsafe` SIMD, `Send`/`Sync` across threads, async serving) is explained where it first appears.

**What "from scratch" means here:** the numeric kernels, tensor layout, weight-file parser, tokenizer, transformer, KV cache, sampler, quantization, batching scheduler and paged cache are all written in this repository in plain Rust. Third-party crates are used only for things that are not the point of the lesson (JSON parsing, memory-mapping a file, a thread pool, an async runtime and an HTTP server), and each one is introduced with a reason.

---

## Syllabus

Each chapter is a folder with a lesson (`README.md`) and a Rust crate you can run and test. Read them in order: each chapter uses what the previous ones built.

### Part I: Foundations

| # | Chapter | You will build |
|---|---|---|
| 1 | [What inference is](chapters/01-what-is-inference/README.md) | A toy model and a harness that measures latency percentiles, batching throughput and the cost of copying weights |
| 2 | [Numbers inside a model](chapters/02-numbers/README.md) | `f16` and `bf16` from scratch, with exhaustive tests; precision and accumulation experiments |
| 3 | [Tensors, strides and views](chapters/03-tensors/README.md) | An owned `Tensor` and borrowed zero-copy `TensorView`s: transpose, slice, broadcast without copying |
| 4 | [Memory is the bottleneck](chapters/04-memory/README.md) | Bandwidth and latency probes for every cache level; the roofline model |

### Part II: Compute kernels

| # | Chapter | You will build |
|---|---|---|
| 5 | [Matrix multiplication](chapters/05-matmul/README.md) | Matmul from naive to cache-blocked, measured in GFLOP/s |
| 6 | [SIMD](chapters/06-simd/README.md) | Dot products with AVX2/FMA and NEON intrinsics, runtime CPU detection |
| 7 | [Threads](chapters/07-threads/README.md) | Parallel matvec and matmul with scoped threads and rayon; false sharing |
| 8 | [Neural network operators](chapters/08-operators/README.md) | Stable softmax, RMSNorm, LayerNorm, SiLU, GELU, tested against f64 references |

### Part III: From files to a working model

| # | Chapter | You will build |
|---|---|---|
| 9 | [Weights on disk](chapters/09-safetensors/README.md) | A safetensors parser and writer, zero-copy loading with `mmap` and lifetimes |
| 10 | [A first model, end to end](chapters/10-first-model/README.md) | Train, export, load and serve a small classifier; batching and shared models |

### Part IV: Language models

| # | Chapter | You will build |
|---|---|---|
| 11 | [Tokenization](chapters/11-tokenization/README.md) | Byte-level BPE training and encoding; streaming UTF-8 decoding |
| 12 | [Attention](chapters/12-attention/README.md) | Scaled dot-product attention, causal masking, multi-head, GQA and RoPE |
| 13 | [The transformer](chapters/13-transformer/README.md) | A complete Llama-style model; parameter and FLOP counting |
| 14 | [The KV cache](chapters/14-kv-cache/README.md) | Prefill and decode with a KV cache, proven equal to full recomputation |
| 15 | [Sampling](chapters/15-sampling/README.md) | Temperature, top-k, top-p, min-p, penalties, and a seeded RNG |
| 16 | [Running a real model](chapters/16-real-model/README.md) | SmolLM2-135M in bf16 with its real tokenizer, checked against PyTorch |

### Part V: Making it fast and small

| # | Chapter | You will build |
|---|---|---|
| 17 | [Measuring and profiling](chapters/17-profiling/README.md) | A benchmark harness and a per-operator profiler for the real model |
| 18 | [Quantization I: int8](chapters/18-int8/README.md) | Symmetric int8 weights and activations, integer dot products, perplexity checks |
| 19 | [Quantization II: 4-bit](chapters/19-4bit/README.md) | Block-wise 4-bit weights with packed nibbles; quality versus size |
| 20 | [FlashAttention](chapters/20-flash-attention/README.md) | Online softmax, tiled attention and split-KV decoding |

### Part VI: Serving

| # | Chapter | You will build |
|---|---|---|
| 21 | [The engine thread](chapters/21-engine-thread/README.md) | An engine on its own thread, streaming tokens over channels, cancellation by drop |
| 22 | [An HTTP inference server](chapters/22-http-server/README.md) | An OpenAI-style API with server-sent events, backpressure and graceful shutdown |
| 23 | [Continuous batching](chapters/23-continuous-batching/README.md) | An iteration-level batch scheduler; throughput versus batch size on SmolLM2 |
| 24 | [Paged KV cache](chapters/24-paged-kv-cache/README.md) | A block allocator, block tables, paged attention and prefix caching |
| 25 | [Scheduling and SLOs](chapters/25-scheduling/README.md) | Load generators, TTFT/TPOT percentiles, chunked prefill and admission control |

### Part VII: Advanced topics

| # | Chapter | You will build |
|---|---|---|
| 26 | [Speculative decoding](chapters/26-speculative-decoding/README.md) | Draft-and-verify with greedy and rejection sampling; measured on real models |
| 27 | [Structured output](chapters/27-structured-output/README.md) | Grammar-constrained decoding that always produces valid JSON |
| 28 | [More than one device](chapters/28-parallelism/README.md) | Tensor, pipeline and expert parallelism, simulated with threads and all-reduce |
| 29 | [GPUs](chapters/29-gpu/README.md) | How GPUs run inference; CUDA kernels written in Rust with NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust) |
| 30 | [Inference in production](chapters/30-production/README.md) | Capacity planning, layer-by-layer numerical debugging, metrics and reliability |

A [glossary](GLOSSARY.md) collects every term the course defines.

---

## How each chapter is laid out

Every lesson follows the same order, so you always know where to look:

1. **In one sentence**, where it fits, what you need, what you will build.
2. **The intuition**: a physical analogy, and where the analogy stops being true.
3. **Vocabulary**: every new term, defined before it is used.
4. **The concepts in depth**.
5. **The code**: the chapter's source, excerpted and explained line by line. The complete files are in the chapter's `src/` folder.
6. **Run it**: the exact command and the output it produced on the reference machine, followed by what the numbers mean.
7. **The Rust behind it**: the Rust features the code relies on, and why they matter for inference.
8. **Mistakes you will make**.
9. **How the professionals do it**: how production engines handle the same problem.
10. **Exercises**, **check yourself** questions, a **recap**, and **answers**.

## Getting started

Install Rust with [rustup](https://rustup.rs/). The course is tested with Rust 1.94.

```bash
git clone <this repository>
cd rust_inference_engineering
cargo test                                      # every chapter's tests
cargo run --release -p ch01-what-is-inference   # run one chapter
```

Always pass `--release` when you time anything. Debug builds are 10-50x slower on numeric code.

### The real model (from chapter 16)

Chapters 16 onward run SmolLM2-135M-Instruct, a 135-million-parameter chat model released by Hugging Face under the Apache 2.0 license. Download it once (about 270 MB):

```bash
./tools/download_model.sh            # SmolLM2-135M-Instruct into models/
./tools/download_model.sh 360M       # the larger sibling, used as a target in chapter 26
```

Tests that need the model skip themselves when it is absent, so `cargo test` always works. To keep models somewhere else, set `INFER_MODELS` to the directory that contains the `smollm2-*` folders.

### Optional: the PyTorch reference

Chapter 16 checks our Rust model against the official PyTorch implementation. The small reference outputs are committed in `chapters/16-real-model/fixtures/`, so you do not need Python. To regenerate them, see [`tools/README.md`](tools/README.md).

## The reference machine

Every number in the lessons was measured, not estimated, on this machine:

- Intel Xeon cloud VM (Sapphire Rapids generation, 2.1 GHz), **4 vCPUs**, 15 GB RAM, Linux
- Supports AVX2, FMA and AVX-512. L2 cache 2 MB per core
- Rust 1.94.1, release builds, no `-C target-cpu=native` unless a lesson says otherwise

A shared cloud VM is noisy: expect run-to-run variation of 5-10%, sometimes more. Your machine will give different numbers. The lessons are about the *shape* of the results (what gets faster, by roughly how much, and why), and each lesson explains how to read your own numbers.

Some code cannot run on this machine: GPU code in chapter 29 and multi-machine communication in chapter 28. Those chapters say clearly which parts were verified and how, and never present unverified output as real.

## How the code is checked

Every chapter must pass four checks before it goes into the course:

```bash
cargo build --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Clippy runs with the `pedantic` group enabled. A few pedantic lints are switched off for the whole workspace in [`Cargo.toml`](Cargo.toml), because numeric code triggers them constantly and deliberately. The most common are the `cast_*` lints (`usize as f32` and similar): we accept those casts and explain the ranges involved where it matters. Everywhere else, when we do something Clippy objects to, we write `#[expect(lint, reason = "...")]` next to it. `expect` rather than `allow`, so the build fails if the exception stops being needed.

A fifth check keeps the lessons honest:

```bash
cargo xtask check
```

Every code excerpt in a lesson is marked with the file it came from, and this command verifies that each excerpt still appears in that file, and that every relative link in the Markdown resolves. If the code changes and the lesson does not, the check fails. CI runs all five checks on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

The SIMD code has separate paths for x86-64 (AVX2) and ARM (NEON, for Apple Silicon and ARM servers). The ARM paths were tested under emulation on the reference machine:

```bash
cargo test --target aarch64-unknown-linux-gnu -p ch06-simd   # needs qemu-user and an aarch64 linker
```

## License

Code and text are dual-licensed under MIT or Apache-2.0, at your option. SmolLM2 weights are not included in this repository and are licensed separately by Hugging Face (Apache 2.0).
