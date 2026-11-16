# Glossary

Every term the course defines, in alphabetical order, with the chapter that defines it (the link goes to that chapter's vocabulary table, where the term is used in context). A term defined in more than one chapter lists each definition: some are refinements of the same idea, some are different meanings of the same word.

## 0-9

- **429 Too Many Requests**: The HTTP status for "not now"; a `Retry-After` header says how many seconds to wait. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))

## A

- **A/B comparison**: Measuring version A against version B. Interleaved: alternating them, so slow drifts affect both. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Acceptance rate**: The share of draft tokens the target keeps. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **Accumulator**: The running total in a sum or dot product. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Activation function**: A non-linear function applied element-wise between linear layers. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Admission control**
  - Deciding at the door whether to accept a request, based on current load. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
  - Deciding at arrival whether to accept a request at all. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Alignment**
  - Whether an address is a multiple of some power of two (here, 64 bytes). ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
  - Whether an address is a multiple of a given number of bytes. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **All-gather**: Every device contributes a piece; every device ends with all the pieces. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **All-reduce**: Every device contributes a buffer; every device ends with their element-wise sum. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **All-to-all**: Every device sends a different piece to every other device. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Amdahl's law**
  - If a fraction s of the work is serial, speed-up can never exceed 1/s. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
  - Speeding up a fraction `f` of the work by `s` gives an overall speedup of `1 / ((1 − f) + f / s)`. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Argmax**: The index of the largest value: the predicted class. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Arithmetic intensity**
  - FLOPs performed per byte moved from memory. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
  - FLOPs per byte moved. Decode's is tiny; prefill's grows with the number of tokens. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Async runtime**: The scheduler (here tokio) that runs many async tasks on a few threads, switching between them at each `.await`. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
- **Atomic**: A value whose operations cannot be interrupted halfway; the basis of lock-free coordination. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Attention weights**: The softmax of the scores: how much each earlier token contributes. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))

## B

- **Bandwidth**: Bytes delivered per second when requests are kept flowing. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Bank conflict**: Several lanes accessing different words in the same shared-memory bank, served one after another. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Barrier (`__syncthreads`)**: A point every thread of a block must reach before any continues. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Batch**
  - Several requests processed together in one forward pass. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
  - The sequences processed in one forward pass. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Batching**: Running several requests through the model together, as one matrix. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Batch invariance**: Getting bit-for-bit the same result for a request whatever else is in its batch. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Bias**: A constant subtracted from the stored exponent so it can be negative. 127 for `f32`, 15 for `f16`. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Bit pattern**: The raw 0s and 1s a number is stored as. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Block**: A fixed number of consecutive positions (here 16) of keys and values, for every layer and KV head. The unit of allocation. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Blocking the runtime**: Running long computation inside an async task, so that the thread cannot switch to other tasks until it finishes. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
- **Blocking / tiling**: Cutting a loop into chunks so that the data a chunk uses fits in a cache. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Block pool**: All blocks, allocated once, with a free list. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Block table**: For one sequence, the list of blocks holding positions `0..len`, in order. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **BPE**: Byte Pair Encoding: build tokens by repeatedly merging the most frequent adjacent pair. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Broadcasting**: Treating a smaller tensor as if it were repeated to match a larger shape. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Bubble**: Time a pipeline stage sits idle, waiting for work from the stage before. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Byte-level**: The base alphabet is the 256 byte values, so any input can be encoded; nothing is "unknown". ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Byte-to-unicode table**: GPT-2's mapping that gives each byte a printable character so tokens can be stored as JSON strings. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))

## C

- **Cache**: Small, fast memory on the processor chip that holds recently used data. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Cache line**
  - The 64-byte unit in which memory is moved between RAM and the CPU. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
  - The unit of transfer between memory levels: 64 bytes on x86 and most ARM chips. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Calibration**: Running the full-precision model on sample text to measure statistics used by quantization. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Canary**: Sending a small share of traffic to a new version first, and comparing. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Capacity factor**: How many tokens an expert accepts per batch, relative to an even share; tokens beyond it are dropped. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Causal mask**: The rule that a token may only attend to itself and earlier tokens. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Channel**: A queue between threads: senders push messages, a receiver takes them in order. `mpsc`: multiple producers, single consumer. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
- **Chat template**
  - The special-token format that marks roles and turns in a conversation. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
  - The rule for turning a list of messages into one prompt string, stored as a Jinja template. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Checkpoint**
  - A file (or set of files) holding a model's trained weights. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
  - The files of a trained model: weights plus the configuration needed to use them. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Chunked prefill**
  - Processing a long prompt in pieces of at most `max_chunk` tokens, each continuing the cache. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
  - Processing a prompt over several steps, a bounded number of tokens per step, alongside other requests' decode tokens. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Chunked transfer encoding**: HTTP/1.1's way of sending a body whose length is not known in advance: each piece is preceded by its size. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
- **Clipping**: Choosing a range slightly smaller than the extremes: the extremes are rounded to the edge, everything else gets finer steps. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Closed loop**: A load generator with a fixed number of users, each sending its next request when the previous one is answered. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Coalescing**: A warp's 32 loads served by few memory transactions because they hit contiguous addresses. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Code**: The stored 4-bit integer, 0..15. The weight is `scale × code + min`. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Collective**: An operation every device of a group takes part in: all-reduce, all-gather, all-to-all. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Column-major**: Layout where the *first* index changes fastest. Used by Fortran, MATLAB, BLAS by default. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Column-, row-parallel**: Splitting a matrix by output rows (each device computes some outputs) or by input columns (each device computes a partial sum of every output). ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Communicator**: A device's connections to the others, through which collectives run (NCCL and MPI's term). ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Completion**: Here: the shortest text that turns a valid prefix into a complete valid document. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))
- **Config**: The numbers that define a model's shape: sizes, counts, θ, ε. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **`config.json`**: Hugging Face's description of a model's architecture: sizes, counts, θ, ε, special token ids. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Constrained decoding**: Restricting, at every step, which tokens may be sampled, so that the output follows the format. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))
- **Context length**
  - How many tokens a query can attend to. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
  - Number of positions in the cache: prompt plus generated tokens. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Contiguous**: The view's elements, in row-major order, sit side by side in memory with no gaps. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Continuous (iteration-level) batching**: Rebuilding the batch at every step: requests join and leave between steps (Yu et al., "Orca", 2022). ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Copy-on-write**: Sharing a block until someone needs to change it, then giving the writer its own copy. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Cosine similarity**: `a·b / (|a||b|)`: 1 when two vectors point the same way. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Counter, gauge**: A metric that only goes up (requests served), and one that goes up and down (requests waiting). ([chapter 30](chapters/30-production/README.md#2-vocabulary))

## D

- **Data parallelism**: Whole copies of the model on several devices, each serving different requests. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Data race**: Two threads access the same memory, at least one writes, with no synchronization. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Deadline propagation**: Passing the client's deadline with the request, so the server can skip work nobody will receive. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Decode**: Every later forward pass: one new token, attending to everything in the cache. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Decoder-only transformer**: A stack of causal self-attention + MLP layers, predicting the next token. GPT, Llama, SmolLM2. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **Detokenization**: Turning token IDs back into text. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Device, rank**: One processor in a group (a GPU, here a thread); its rank is its number in the group. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Distribution**: Probabilities over tokens, from `softmax(logits)`. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Draft length `k`**: How many tokens the drafter proposes per round. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **Draft model, drafter**: Anything that cheaply proposes the next tokens: a smaller model, extra heads on the target, or a lookup in the text. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **DRAM / main memory**: The RAM sticks (or HBM on a GPU). Large and comparatively slow. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Dynamic quantization**: Choosing activation scales at run time, from the actual values. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))

## E

- **Element**: One number in the tensor. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Element-wise**: Applied to each number independently (activations, residual add). ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Embedding**: A table with one learned vector per token; lookup is by token ID. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Engine thread**: The one thread that owns the model and runs every forward pass. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
- **Expert parallelism (EP)**: Spreading the experts across devices and sending tokens to them. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Exponent**: Which power of two the number sits near. Sets the *range*. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Exposition format**: The text a server returns at `/metrics` for Prometheus to scrape. ([chapter 30](chapters/30-production/README.md#2-vocabulary))

## F

- **False sharing**: Threads writing *different* variables that share one cache line, forcing the line to bounce between cores. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Finish reason**: Why generation ended: `length`, `stop` (token or string) or cancellation. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **FlashAttention**: Attention computed tile by tile with the online softmax, never storing a full row of scores (Dao et al., 2022). ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))
- **FLOP**: One floating-point operation (a multiply or an add). GFLOP/s = billions per second. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **FLOP/s**: Floating-point operations per second. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **FMA**: Fused multiply-add: `a × b + c` in one instruction, with a single rounding. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **Forward pass**: One run of the model from input to output. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Frequency / presence penalty**: OpenAI style: subtract from the logits of generated tokens, per occurrence / once. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))

## G

- **GEMM**: "General matrix multiply", the standard library name (BLAS `sgemm` for `f32`). ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **GEMV**: Matrix-vector multiply: GEMM with m = 1. The shape of decode. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **GGUF**: llama.cpp's single-file format: metadata, tokenizer, and (often quantized) tensors. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Global memory**: The GPU's main memory (HBM, 80 GB at 3.35 TB/s on an H100). ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Golden fixtures**: Reference outputs, generated once with the trusted implementation and committed next to the tests. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Goodput**: Requests per second that met the SLO. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **GPTQ / AWQ**: Widely used 4-bit methods that use calibration data (Frantar et al.; Lin et al.). ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Graceful shutdown**: Stop accepting new connections, finish the requests in progress, then exit. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
- **Granularity**: How many values share a scale: per tensor, per row (output channel), per block (group). ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Greedy decoding**
  - Always choosing the highest-scoring next token (chapter 15 covers alternatives). ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
  - Always choosing the most likely token. Deterministic. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))

## H

- **Head**: One independent attention computation on a slice of the vector (`head_dim` numbers). ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Header**: The part of a file that describes the rest. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Head-of-line blocking**
  - A request waiting only because another one is ahead of it. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
  - A long job at the front of a queue delaying every short job behind it. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Headroom**: Capacity kept unused at the expected peak, for bursts and failures. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Health check**: An endpoint that load balancers and orchestrators call to see whether the server is alive and ready. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
- **Held-out set**: Data the model never saw during training, used to measure real accuracy. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Hidden state**: The vector representing one token inside the model (576 numbers for SmolLM2). ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Histogram**: Counts of observations per bucket (`le`, "less than or equal"), with their sum and count. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Horizontal sum**: Adding the lanes of one vector together. Needed at the end of a dot product. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **Hot spot**: Code where a large share of the time goes. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))

## I

- **Importance**: Here, the mean square of the activations entering each input column of a matrix. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Inference**: Running the model with fixed weights on new inputs. Also called "serving" or "prediction". ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Instruct / chat model**: A model fine-tuned to answer in turns, using the template's special tokens. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Instrumentation**: Timers placed in the code around the parts you care about. Exact for those parts, blind elsewhere. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Internal fragmentation**: Memory reserved for a request but never used by it. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Inter-token latency (ITL)**: One gap between two consecutive tokens of a request. Their average for a request is its TPOT; their tail shows stalls. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Intrinsic**: A function that maps directly to one CPU instruction, like `_mm256_fmadd_ps`. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **ISA**: Instruction set architecture: SSE2, AVX2, AVX-512, NEON... ([chapter 6](chapters/06-simd/README.md#2-vocabulary))

## J

- **JSON Schema**: A JSON description of what a JSON document may contain: types, required properties, enums, lengths. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))
- **Jump-forward decoding**: Appending text the format forces (a fixed key, a closing bracket) without asking the model for each token. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))

## K

- **Kernel**: A function that runs on the GPU, launched by the CPU (the **host**) over many threads. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **KL divergence**: How far one probability distribution is from another; here, how far the quantized model's predictions are from the original's. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **KV cache**
  - Stored keys and values of earlier tokens, so they are not recomputed (chapter 14). ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
  - The keys and values of every processed position, for every layer, kept between forward passes. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))

## L

- **L1 / L2 / L3**: Cache levels, from smallest and fastest (L1, per core) to largest and slowest (L3, shared). ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Lane**: One element position in a vector register. A 256-bit register holds 8 `f32` lanes. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **Latency**
  - How long one request takes, start to finish. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
  - Time from asking for a piece of data until it arrives. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
  - Cycles from an instruction starting until its result can be used. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
  - Time for one request, from arrival to answer. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Layer / block**: One repeated unit: attention block + MLP block, each with a norm and a residual. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **Little-endian**: Byte order where the least significant byte comes first. safetensors, x86 and ARM all use it. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Little's law**: `L = λW`: the average number of requests in a system equals their arrival rate times the average time each spends in it. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **LM head**: The final projection from hidden size to vocabulary size, producing logits. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **Load balance**: Whether every thread gets the same amount of work. The slowest thread decides when a parallel step ends. ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))
- **Load-bound / FMA-bound**: A kernel limited by how many loads, or how many multiply-adds, the core can issue per cycle. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Logits**
  - Raw, unnormalized scores. Softmax turns them into probabilities. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
  - The raw scores the last layer produces, one per class. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
  - The model's raw scores, one per vocabulary token. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Logprob**: The natural log of a token's probability, reported by APIs to show the model's confidence. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Loop order**: Which of the three loops (i, j, p) is outermost. Six orders, same result, different speed. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))

## M

- **Machine epsilon**: The ULP of 1.0. For `f32` it is 2⁻²³ ≈ 1.19 × 10⁻⁷. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Mantissa** (or significand, fraction): The digits after the leading 1. Sets the *precision*. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Matmul**: Matrix multiplication. `C = A · B` with `A` of shape m×k and `B` of shape k×n gives `C` of shape m×n. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Memory bandwidth**: How many bytes per second the processor can pull from memory. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Memory-bound / compute-bound**
  - Limited by bytes moved / by arithmetic performed. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
  - Limited by bytes moved from memory, or by arithmetic (chapter 4's roofline). ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Memory ordering**: Rules (`Relaxed`, `Acquire`, `Release`...) for when one thread's writes become visible to another. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Merge**: A learned rule "tokens a and b next to each other become token c". ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Metastable failure**: An overload that sustains itself after its trigger is gone. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **MHA / GQA / MQA**: Multi-head / grouped-query / multi-query attention: every query head has its own K/V head, groups share one, or all share one. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Micro-benchmark**: A small program that measures one operation in isolation. Useful, and misleading when its shapes differ from the real workload's. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Micro-kernel**: The innermost, register-tiled piece of a fast matmul. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Minimax polynomial**: The polynomial of a given degree whose worst-case error over a range is smallest. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Min-max (asymmetric)**: The 16 levels span the block's own minimum to maximum. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Min-p**: Keep tokens whose probability is at least `min_p` times that of the most likely token. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Mixture of experts (MoE)**: A layer with many MLPs (experts) and a router that sends each token to a few of them. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **MLP**: Multi-layer perceptron: linear layers separated by non-linear activations. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **MLP / feed-forward network**: The per-token part of a layer: gate and up projections, SwiGLU, down projection. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **mmap**: Memory-mapping: making a file's contents appear in the process's address space, loaded lazily. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **M/M/c queue, Erlang C**: Poisson arrivals, `c` servers with exponential service times; Erlang's C formula gives the probability of waiting. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Model**: A function with a large number of fixed parameters. Input numbers go in, output numbers come out. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **α-β model**: Transfer time = α (latency per message) + bytes / β (bandwidth). ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Monomorphization**: The compiler generating a separate copy of generic code for each concrete type it is used with. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Multiversioning**: Compiling the same source code several times, for different instruction sets, and choosing at run time. ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))

## N

- **N+1**: One more replica than the load needs, so that one can fail or be upgraded. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **NaN**: "Not a number": the result of 0/0, ∞−∞ and similar. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Nibble**: Half a byte, 4 bits: one 4-bit code. Two per byte. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **NN / NT layout**: Whether the second operand is stored normally (k×n) or transposed (n×k). Linear layers are NT. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Noise**: Variation between runs of the same work, from other programs, the OS, the hypervisor, frequency changes. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Normalization**: Rescaling a vector to a standard size (RMSNorm, LayerNorm). ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Numerically stable**: Gives accurate results over the whole input range, without overflow or catastrophic cancellation. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))

## O

- **Occupancy**: How many warps an SM holds at once: more warps hide more memory latency. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Offset**: Where in the buffer the view's first element is. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Online softmax**: Softmax (and the weighted sum after it) in one pass, keeping a running maximum and sum and rescaling when the maximum grows (Milakov and Gimelshein, 2018). ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))
- **OpenAI-compatible**: Accepting the same JSON requests and returning the same responses as OpenAI's API, so that its SDKs and tools work unchanged. vLLM, TGI, llama.cpp, Ollama all offer this. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
- **Open loop**: A load generator that sends requests at times decided in advance, whatever the server is doing. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Operator**: One step of a model's computation: a matmul, a softmax, a norm... ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Operator-level parallelism**: Splitting each operation of one request across cores. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Outlier**: A value much larger than its neighbours; it inflates the scale for everything sharing it. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Overflow**: A result too large for the format: becomes infinity (or saturates to the maximum). ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))

## P

- **Packing**: Rearranging a weight matrix once, at load time, into the order a kernel reads it. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Page**: The unit the operating system maps memory in: 4 KB normally, 2 MB or 1 GB for "huge pages". ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Page cache**: The OS's in-memory cache of file contents, shared by all processes. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Page fault**: The first touch of a page the OS has promised but not yet provided. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Percentile**: The p-th percentile is the smallest value that p% of the measurements are at or below. p50 is the median; p99 is the latency one request in a hundred exceeds. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Percentile (p50, p99)**: p99 = 99% of requests were at least this fast. The slow 1% is the tail. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Perplexity**: `exp(mean negative log-likelihood)` of real text: how surprised the model is, on average. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Pickle**: Python's object serialization. PyTorch's `.bin`/`.pt` files use it; loading one can run arbitrary code. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Pipeline parallelism (PP)**: Giving each device a block of consecutive layers (a **stage**). ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Poisson process**: Arrivals at random moments, independent of each other, at a constant average rate; the gaps between them are exponentially distributed. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Preemption**: Taking a running request's memory away so that others can continue; here it later recomputes what it lost. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Prefetcher**: Hardware that detects access patterns and fetches lines before they are asked for. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Prefill**: The first forward pass over the whole prompt. Fills the cache and produces the first new token. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Prefill / decode**: Processing the whole prompt at once / generating one token at a time (chapter 1). ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Prefix caching**: Reusing the blocks of earlier requests that started with the same tokens. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Pre-norm**: Normalizing the input of each block (not its output), as Llama does. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **Pre-tokenization**: Splitting text into chunks (roughly words) that merges may not cross. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Profile**: A breakdown of where a program spends its time. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Prompt lookup (n-gram) decoding**: Drafting by finding the last few tokens earlier in the text and proposing what followed them there. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **PTX**: NVIDIA's virtual instruction set; compilers emit it, the driver turns it into machine code. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Pushdown automaton**: A state machine with a stack: enough to follow nested structure (objects in arrays in objects), which a plain state machine cannot. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))

## Q

- **Quantization**: Storing numbers with fewer bits by mapping them to a small set of levels. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Query, key, value (Q, K, V)**: Three vectors computed from each token's hidden state by learned linear layers. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Queueing delay**: Time a request waits before the engine starts it. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))

## R

- **Rank**
  - The number of dimensions. A matrix has rank 2. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
  - A merge's position in the learned order. Lower rank = learned earlier = applied first. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Reduction dimension**: `k`: the dimension that is summed over. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Reference count**: How many owners (sequences, the prefix cache) a block has; it is free when the count reaches 0. ([chapter 24](chapters/24-paged-kv-cache/README.md#2-vocabulary))
- **Reference implementation**
  - A simple, trusted implementation used to check fast ones. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
  - A trusted version of the model (here PyTorch) whose intermediate values are the expected ones. ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **Register spill**: When a loop needs more values than there are registers, and the compiler parks some in memory. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Register tiling**
  - Computing a small block of outputs at once so each loaded value is used several times from registers. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
  - Computing a block of outputs at once so each loaded value is used several times from registers. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Rejection sampling**: Accepting a sample from one distribution with a probability chosen so that the accepted results follow another. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **Repetition penalty**: Hugging Face style: shrink the logits of tokens already seen, by division or multiplication. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Request**: One unit of work from a user, for example "complete this prompt". ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Request-level parallelism**: Running different requests on different cores at the same time. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Resident memory (RSS)**: The physical memory a process currently occupies, split into anonymous (its own) and file-backed (mapped files). ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Residual connection**: Adding a layer's input back to its output: `x = x + f(x)`. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Residual stream**: The hidden-state vector each layer reads from and adds to. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **Reuse**: How many times a value fetched from memory is used before it is evicted. ([chapter 5](chapters/05-matmul/README.md#2-vocabulary))
- **Ring all-reduce**: All-reduce in `2 (n − 1)` steps around a ring, each device sending `2 (n − 1) / n` buffers in total. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Rollback**
  - Shrinking the cache to an earlier length, to discard tokens (used by speculative decoding). ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
  - Discarding the KV cache entries of rejected draft tokens. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **Roofline**: A model: speed ≤ min(peak FLOP/s, intensity × bandwidth). ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **RoPE**: Rotary position embedding: positions encoded by rotating pairs of Q and K dimensions. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Round half to even**: Round to nearest; exact ties go to the value whose last bit is 0. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Rounding mode**: The rule for picking a representable value when the exact one falls between two marks. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Row-major**: Layout where the *last* index changes fastest in memory. Used by C, Rust, PyTorch, NumPy, safetensors. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Row-wise**: Needs a whole row at once (softmax, normalization), because it uses a sum or maximum. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Runtime dispatch**: Detecting the CPU's features when the program runs, then calling the best kernel. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))

## S

- **safetensors**: A simple, safe weight format from Hugging Face: a JSON header plus raw bytes. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Sampling profiler**: Interrupts the program many times per second and records where it was (`perf`, VTune, Instruments). Cheap, covers everything, statistical. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Scale**: The step between levels: a value is reconstructed as `scale × q`. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Scoped thread**: A thread guaranteed to finish before a scope ends, so it can borrow local data. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Score**: The dot product of a query and a key, scaled by 1/√d. ([chapter 12](chapters/12-attention/README.md#2-vocabulary))
- **Scratch space**: Temporary buffers allocated once and reused by every forward pass. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))
- **Sector**: The 32-byte unit in which GPU memory delivers data. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Seed**: The starting state of the random generator; same seed, same draws. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **`Send`**: A type that may be moved to another thread. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Shape**: The size of each dimension, e.g. `[2, 3, 4]`. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Sharded checkpoint**: A large model split over several files, plus an index saying which tensor is in which file. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Shared memory**: Fast on-chip memory per SM, shared by a block's threads (up to 228 KB per SM on an H100). ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Sign bit**: One bit: 0 for positive, 1 for negative. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **`SIGTERM`**: The signal process managers (`docker stop`, Kubernetes, systemd) send to ask a program to shut down, before forcing it with `SIGKILL`. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
- **SIMD**: Single Instruction, Multiple Data: one instruction operating on a vector of values. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **SLO**: Service-level objective: the latencies a service promises itself, such as "first token within 1 s, then at most 100 ms per token". ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **SLO attainment**: The share of requests that met the SLO. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Slot**: A KV cache reserved for one request while it runs. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **SM (streaming multiprocessor)**: A GPU core: runs several blocks at once, with its registers, shared memory and execution units. An H100 has 132. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Softmax**: `e^xᵢ / Σⱼ e^xⱼ`: turns a vector of scores into probabilities that sum to 1. ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Special (added) token**: A token like `<|im_end|>` matched literally in text, never split or merged. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Special token**: A reserved token like `<\|endoftext\|>` or `<\|im_start\|>`, matched literally, never merged. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Speed-up**: Time with one thread ÷ time with N threads. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Spinning / busy-waiting**: A waiting thread repeatedly checks a condition instead of sleeping. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Split-KV / flash-decoding**: Splitting one head's keys into parts processed in parallel, each producing a partial state, merged at the end. ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))
- **SSE (server-sent events)**: A streaming format over one HTTP response: `data: ...` lines, each event ended by a blank line. Content type `text/event-stream`. ([chapter 22](chapters/22-http-server/README.md#2-vocabulary))
- **Static batching**: Forming a batch, running it to completion, then forming the next. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Step**: One forward pass of the engine: one new token for every decoding request, plus prompt chunks. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Stop sequence**: A string (like `"\nUser:"`) that ends generation when it appears in the text. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Stop token**: A token (like `<|im_end|>`) that ends generation when produced. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Streaming**: Sending each token to the client as soon as it is produced. ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
- **Stride**: How many elements to skip in the flat buffer to move one step along a dimension. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Structured output**: Model output that must follow a format exactly: JSON, a JSON schema, a regular expression, a grammar. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))
- **Subnormal**: Tiny numbers near zero with reduced precision, filling the gap down to 0. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **Symmetric**: Range centred on zero: codes stand for −7..7 times the scale. ([chapter 19](chapters/19-4bit/README.md#2-vocabulary))
- **Symmetric quantization**: Levels spread evenly around zero (−127..127); no offset stored. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **`Sync`**: A type that may be shared (by `&` reference) between threads. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))

## T

- **Tail**: The leftover elements when the length is not a multiple of the vector width. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **Target feature**: A CPU capability the compiler may assume (`avx2`, `fma`, `avx512f`, `neon`). ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **Target model**: The model whose output we want (here SmolLM2-360M). ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **Task**: Here, one KV head × one block of query tokens × one part of the keys. The unit of work handed to a thread. ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))
- **Temperature (T)**: Logits are divided by T before the softmax. T < 1 sharpens, T > 1 flattens, T → 0 is greedy. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Tensor**: A multi-dimensional array of numbers. A vector is a 1-D tensor, a matrix 2-D. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **Tensor core**: A matrix unit that multiplies small matrix tiles in one instruction; most of a GPU's FLOPs. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Tensor parallelism (TP)**: Splitting each weight matrix across devices; the devices work on every token together. ([chapter 28](chapters/28-parallelism/README.md#2-vocabulary))
- **Thread**: An independent sequence of execution, scheduled by the OS onto a core. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Thread, block, grid**: A kernel runs as a grid of blocks; a block is up to 1,024 threads that can share memory and synchronize. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Thread pool**: A set of long-lived threads that run submitted tasks, so threads are not created per task. ([chapter 7](chapters/07-threads/README.md#2-vocabulary))
- **Throughput**
  - How much work gets done per second (requests per second, or tokens per second). ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
  - How many such instructions can start per cycle. ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
  - Requests completed per second. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
  - Tokens produced per second by the whole engine. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
- **Tied embeddings**: Using the embedding matrix as the LM head, saving vocab × hidden parameters. ([chapter 13](chapters/13-transformer/README.md#2-vocabulary))
- **Tile**: A block of consecutive keys (and their values) processed together; here 64. ([chapter 20](chapters/20-flash-attention/README.md#2-vocabulary))
- **Timeout, retry, backoff, jitter**: A client gives up on an attempt after a timeout and tries again, waiting longer each time (backoff), by a random amount (jitter). ([chapter 30](chapters/30-production/README.md#2-vocabulary))
- **TLB**: A cache of address translations (virtual page to physical page). Missing it adds latency. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Token**
  - The unit a language model reads and writes: a word or piece of a word (chapter 11). ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
  - One entry of the vocabulary; the unit a model reads and writes. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Token ID**: The integer that names a token. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **`tokenizer.json`**: Hugging Face's tokenizer file: vocabulary, merges, pre-tokenizer, special tokens. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Token mask**: For one step, the set of tokens that may come next. The others get a logit of minus infinity. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))
- **Top-k**: Keep only the k most likely tokens. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Top-p (nucleus)**: Keep the smallest set of most likely tokens whose probabilities add up to at least p. ([chapter 15](chapters/15-sampling/README.md#2-vocabulary))
- **Total variation distance**: Half the sum of absolute differences between two distributions; 0 when equal, 1 when they share nothing. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **TPOT**
  - Time per output token, as one client sees it: the time between its tokens. ([chapter 23](chapters/23-continuous-batching/README.md#2-vocabulary))
  - Time per output token: the average gap between a request's tokens after the first. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **Training**: Searching for weights that make the model's outputs good. Not covered in this course. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Trie**: A tree of strings in which strings with a common beginning share a path. ([chapter 27](chapters/27-structured-output/README.md#2-vocabulary))
- **TTFT**
  - Time to first token: from submitting a request to receiving its first token. What a user perceives as "responsiveness". ([chapter 21](chapters/21-engine-thread/README.md#2-vocabulary))
  - Time to first token: from sending a request to receiving its first token. Waiting plus prefill. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))
- **TTFT / TPOT**: Time to first token (mostly prefill) and time per output token (one decode step), from chapter 1. ([chapter 14](chapters/14-kv-cache/README.md#2-vocabulary))

## U

- **ULP**
  - "Unit in the last place": the gap between one representable number and the next. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
  - Unit in the last place: the gap between adjacent floats (chapter 2). ([chapter 8](chapters/08-operators/README.md#2-vocabulary))
- **Underflow**: A result too small: loses precision, then becomes zero. ([chapter 2](chapters/02-numbers/README.md#2-vocabulary))
- **UTF-8**: The encoding of Unicode text as 1-4 bytes per character. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))
- **Utilization**: The arrival rate divided by the rate at which the server can complete requests. ([chapter 25](chapters/25-scheduling/README.md#2-vocabulary))

## V

- **Vector register**: A wide register: 128-bit (SSE, NEON), 256-bit (AVX2) or 512-bit (AVX-512). ([chapter 6](chapters/06-simd/README.md#2-vocabulary))
- **Verification**: One forward pass of the target over the last token and the `k` draft tokens, giving `k + 1` rows of logits. ([chapter 26](chapters/26-speculative-decoding/README.md#2-vocabulary))
- **View**: A tensor that borrows another tensor's buffer with its own shape/strides/offset. ([chapter 3](chapters/03-tensors/README.md#2-vocabulary))
- **VNNI**: "Vector Neural Network Instructions": x86 instructions that multiply 8-bit integers and accumulate in 32 bits (`vpdpbusd`). ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Vocabulary**: The fixed list of all tokens. SmolLM2 has 49,152. ([chapter 11](chapters/11-tokenization/README.md#2-vocabulary))

## W

- **W8A32 (weight-only)**: Int8 weights, full-precision activations. Saves memory traffic; math stays in floating point. (On GPUs usually called W8A16.) ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **W8A8**: Int8 weights *and* activations; products computed on integers. ([chapter 18](chapters/18-int8/README.md#2-vocabulary))
- **Wall time / CPU time**: Time on a clock / time summed over all threads. A parallel program uses more CPU time than wall time. ([chapter 17](chapters/17-profiling/README.md#2-vocabulary))
- **Warp**: 32 threads of a block that execute each instruction together (SIMT: single instruction, multiple threads). ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Warp shuffle**: Reading another lane's register, within a warp, without memory. ([chapter 29](chapters/29-gpu/README.md#2-vocabulary))
- **Weights / parameters**: The fixed numbers inside the model, produced by training. ([chapter 1](chapters/01-what-is-inference/README.md#2-vocabulary))
- **Working set**: The amount of memory a loop touches repeatedly. Whether it fits in a cache decides its speed. ([chapter 4](chapters/04-memory/README.md#2-vocabulary))
- **Workspace**: Buffers allocated once and reused by every forward pass. ([chapter 10](chapters/10-first-model/README.md#2-vocabulary))
- **Work stealing**: A scheduling strategy where idle threads take tasks from busy threads' queues (rayon). ([chapter 7](chapters/07-threads/README.md#2-vocabulary))

## Z

- **Zero-copy**: Using data where it already is, without copying it into a new buffer. ([chapter 9](chapters/09-safetensors/README.md#2-vocabulary))
- **Zero-copy loading**: Using weights straight from a memory-mapped file instead of copying them. ([chapter 16](chapters/16-real-model/README.md#2-vocabulary))
- **Zero point**: The offset used by asymmetric quantization (levels `0..255` mapped to `[min, max]`). ([chapter 18](chapters/18-int8/README.md#2-vocabulary))

## Chapters

1. [What inference is, and what an inference engineer does](chapters/01-what-is-inference/README.md)
2. [Numbers inside a model](chapters/02-numbers/README.md)
3. [Tensors, strides and views](chapters/03-tensors/README.md)
4. [Memory is the bottleneck](chapters/04-memory/README.md)
5. [Matrix multiplication](chapters/05-matmul/README.md)
6. [SIMD](chapters/06-simd/README.md)
7. [Threads](chapters/07-threads/README.md)
8. [Neural network operators](chapters/08-operators/README.md)
9. [Weights on disk](chapters/09-safetensors/README.md)
10. [A first model, end to end](chapters/10-first-model/README.md)
11. [Tokenization](chapters/11-tokenization/README.md)
12. [Attention](chapters/12-attention/README.md)
13. [The transformer](chapters/13-transformer/README.md)
14. [The KV cache](chapters/14-kv-cache/README.md)
15. [Sampling](chapters/15-sampling/README.md)
16. [Running a real model](chapters/16-real-model/README.md)
17. [Measuring and profiling](chapters/17-profiling/README.md)
18. [Quantization I: int8](chapters/18-int8/README.md)
19. [Quantization II: 4-bit](chapters/19-4bit/README.md)
20. [FlashAttention](chapters/20-flash-attention/README.md)
21. [The engine thread](chapters/21-engine-thread/README.md)
22. [An HTTP server](chapters/22-http-server/README.md)
23. [Continuous batching](chapters/23-continuous-batching/README.md)
24. [Paged KV cache](chapters/24-paged-kv-cache/README.md)
25. [Scheduling and SLOs](chapters/25-scheduling/README.md)
26. [Speculative decoding](chapters/26-speculative-decoding/README.md)
27. [Structured output](chapters/27-structured-output/README.md)
28. [More than one device](chapters/28-parallelism/README.md)
29. [GPUs](chapters/29-gpu/README.md)
30. [Inference in production](chapters/30-production/README.md)
