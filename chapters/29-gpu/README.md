# Chapter 29: GPUs

> **In one sentence:** a GPU is thousands of simple lanes grouped into warps of 32 that execute each instruction together, fed by memory several times faster than a CPU's and by matrix units tens of times faster than its vector units, so the same roofline that governed this course's CPU decides everything, at a different scale; writing fast kernels comes down to making each warp's memory accesses contiguous, reusing loaded data from on-chip shared memory, and synchronizing threads correctly, which this chapter checks in a small simulator and then writes for NVIDIA GPUs in Rust with cuda-oxide.

**Where this fits:** every engine in this course ran on a 4-core CPU. Production inference runs on GPUs. This chapter explains what is different about them, shows the kernels of a forward pass in the form a GPU needs, and estimates what real GPUs can do for the models of this course. Chapter 30 then turns to running inference as a service.

**You need:** chapter 4 (memory bandwidth, the roofline), chapter 5 (tiled matrix multiplication), chapter 6 (SIMD lanes), chapter 7 (threads and races), chapter 8 (RMSNorm), chapter 20 (FlashAttention, which was invented for GPUs), chapter 23 (batching and the ridge point).

**You will build:** a SIMT simulator (grids, blocks, threads, warps, shared memory, barriers, warp shuffles) that counts the memory traffic a GPU would see and detects shared-memory races; RMSNorm, two matrix-vector products and three matrix products written for it and checked on SmolLM2's weights; roofline estimates for five processors; and the same kernels for NVIDIA GPUs, in Rust, with cuda-oxide.

**What ran where.** This machine has no GPU. The simulator and its kernels ran here, and every result is checked against the CPU. The cuda-oxide kernels in [`oxide/main.rs`](oxide/main.rs) were written against cuda-oxide's documentation and were **not compiled or run** for this course: cuda-oxide needs an NVIDIA GPU, CUDA 13, LLVM 21 and its own pinned toolchain. They mirror the simulator's kernels line for line. The GPU numbers in section 3.6 are computed from published specifications, not measured.

---

## 1. The intuition

A CPU is a few expert workers, each with a large desk (caches), who work through long, complicated tasks on their own and guess cleverly what they will need next. A GPU is a factory floor with thousands of workers in teams of 32. A team does one thing at a time, all 32 together: "everyone, add". When a team has to wait for materials from the warehouse (memory), the floor manager simply switches to another team that is ready; there are so many teams that the floor is always busy. Each workshop (a streaming multiprocessor) has a small shared shelf (shared memory) that its teams can stock and use together, if they agree on when the shelf is ready.

Three rules follow. A team that fetches materials should fetch from one place, not 32 places (coalescing). Materials used many times should be fetched once onto the shelf (tiling). And no one should take from the shelf before it is stocked (barriers).

**Where the analogy breaks:** the 32 workers of a team are not independent: they share one instruction stream, so when some must do one thing and others another (a branch), the team does both, one after the other, with half of it idle each time.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Kernel** | A function that runs on the GPU, launched by the CPU (the **host**) over many threads. |
| **Thread, block, grid** | A kernel runs as a grid of blocks; a block is up to 1,024 threads that can share memory and synchronize. |
| **Warp** | 32 threads of a block that execute each instruction together (SIMT: single instruction, multiple threads). |
| **SM (streaming multiprocessor)** | A GPU core: runs several blocks at once, with its registers, shared memory and execution units. An H100 has 132. |
| **Global memory** | The GPU's main memory (HBM, 80 GB at 3.35 TB/s on an H100). |
| **Shared memory** | Fast on-chip memory per SM, shared by a block's threads (up to 228 KB per SM on an H100). |
| **Coalescing** | A warp's 32 loads served by few memory transactions because they hit contiguous addresses. |
| **Sector** | The 32-byte unit in which GPU memory delivers data. |
| **Bank conflict** | Several lanes accessing different words in the same shared-memory bank, served one after another. |
| **Barrier (`__syncthreads`)** | A point every thread of a block must reach before any continues. |
| **Warp shuffle** | Reading another lane's register, within a warp, without memory. |
| **Tensor core** | A matrix unit that multiplies small matrix tiles in one instruction; most of a GPU's FLOPs. |
| **Occupancy** | How many warps an SM holds at once: more warps hide more memory latency. |
| **PTX** | NVIDIA's virtual instruction set; compilers emit it, the driver turns it into machine code. |

## 3. The concepts in depth

### 3.1 What a GPU is

An H100 has 132 SMs. Each SM has four schedulers, each issuing one warp instruction per cycle, and holds up to 64 warps (2,048 threads) at once, with 256 KB of registers and up to 228 KB of shared memory. Behind them: 50 MB of L2 cache and 80 GB of HBM3 memory at 3.35 TB/s. The tensor cores deliver 989 TFLOP/s on `bf16` matrix products; ordinary arithmetic is far slower.

The difference from a CPU is not only size. A CPU core hides memory latency with large caches and by predicting the future; it runs one or two threads. An SM hides latency with **other warps**: when one waits for memory (hundreds of cycles), the scheduler issues instructions from another, at no cost. The program's job is to give it enough independent warps (occupancy) and enough work per byte fetched.

The programming model: the host (CPU) copies data to device memory, **launches a kernel** over a grid of blocks, and copies results back. Every thread runs the same function and finds its work from its indices: `blockIdx`, `threadIdx`, `blockDim`. Blocks are independent: they may run in any order, on any SM. Threads within a block may share memory and wait for each other at barriers.

### 3.2 A simulator to check kernels on a CPU

Kernels are hard to test without a GPU, and wrong ones are easy to write. This chapter's simulator runs real kernel code on the CPU in the GPU's execution model:

- A block's code between two barriers is a **segment**. The simulator runs it for thread 0, then thread 1, and so on. On a GPU the threads run concurrently, but within a segment no thread may depend on another's work (that is what barriers are for), so any order, including this one, gives the same result for a correct kernel.
- To catch kernels that are *not* correct, the simulator tracks shared memory between barriers: a word written by one thread and read or written by another without a barrier in between is reported as a **race**.
- Per-thread values that survive a barrier (registers) live in arrays indexed by thread; warp shuffles operate on those arrays.
- Every global load is recorded with its warp and its rank among the lane's loads, which identifies one warp-wide load instruction. The simulator counts the 32-byte sectors each instruction touches, as GPU memory delivers them, and the shared-memory banks each shared access hits.

It is slow (the matrix products below take a few seconds), and it is not a performance model: it counts traffic, not time. But traffic is what decides an inference kernel's speed.

### 3.3 Coalescing: how a warp reads memory

Decoding multiplies each weight matrix by one vector (chapter 23). The obvious GPU kernel gives each thread one row of `W`: thread `r` computes `y[r] = Σ W[r][c]·x[c]`. At each step, the 32 lanes of a warp read `W[r][c]` for 32 consecutive rows: 32 addresses one row (576 floats) apart. Each lands in a different 32-byte sector, and each sector delivers one useful float out of eight.

The better kernel gives each **warp** one row. Lane `l` reads columns `l, l + 32, l + 64, ...`: at each step the warp reads 32 consecutive floats, 128 contiguous bytes, exactly 4 sectors, all useful. At the end, each lane has a partial sum, and the warp adds the 32 of them with five **shuffles**: each lane adds the value of the lane whose index differs in one bit (16, 8, 4, 2, 1), and after five rounds every lane holds the total.

Part 1 runs both, and the other kernels, in the simulator on layer 0 of SmolLM2-135M (its MLP up projection, 1,536 × 576, and a real token's hidden state):

```text
== 1. kernels in the SIMT simulator, on SmolLM2-135M's layer 0 (f32 values)
   kernel                           loads   requested     fetched  useful  wavefronts sync race max error
                                                                          per access
   RMSNorm, 1 × 576                  1728        7 KB        7 KB    100%      1.00     9    0    7.2e-7
   W·x, 1536 × 576, thread/row    1769472     6912 KB    28512 KB     24%         -     0    0     0.0e0
   W·x, 1536 × 576, warp/row      1769472     6912 KB     6912 KB    100%         -     0    0    1.4e-6
   A·Wᵀ, 64 × 256 × 576, naive   18874368    73728 KB   165888 KB     44%         -     0    0     0.0e0
   A·Wᵀ, tiled                    1179648     4608 KB     4608 KB    100%      4.29  4608    0    1.7e-6
   A·Wᵀ, tiled, B transposed      1179648     4608 KB     4608 KB    100%      1.21  4608    0    1.7e-6
```

Both matrix-vector kernels make the same 1.77 million loads. Thread-per-row makes the memory deliver 28 MB to use 6.9 MB: 24% useful (the loads of `x`, which every lane reads at the same address, are served once per warp and are fully useful; the loads of `W` are 1/8 useful). Warp-per-row uses every byte delivered. Decoding is memory-bound, so on a GPU the first kernel would take about four times as long as the second, for the same arithmetic. The results differ from the CPU by 1.4 × 10⁻⁶ for the warp version (the shuffles add in a tree, not in order) and by nothing for the thread version (which adds in the CPU's order).

### 3.4 Shared memory, barriers, tiling

RMSNorm needs the sum of squares of a whole row before any value can be scaled: a **block reduction**. Each of 256 threads adds the squares of a strided slice of the row (`c = t, t + 256, ...`, coalesced), writes its partial sum to shared memory, and the block adds them up by halving: threads 0-127 each add the value 128 places above theirs, then threads 0-63, and so on. Between steps, a **barrier**: a thread must not read a partial sum before the thread that writes it has written it. Part 1's RMSNorm row shows 9 barriers per row (one after the partial sums, eight halvings) and no race. Remove a barrier, and the simulator reports races (its tests do exactly that).

Prefill multiplies matrices (chapter 5). The naive kernel computes each output from a row of `A` and a row of `B` read straight from global memory, `2k` loads per output: 18.9 million loads for 64 × 256 outputs. The **tiled** kernel lets a block of 16 × 16 threads compute a 16 × 16 tile of the output. For each 16-wide slice of `k`, each thread loads one value of `A` and one of `B` into shared memory; after a barrier, each thread multiplies a row of the `A` tile by a column of the `B` tile, reading only shared memory; after another barrier, the next slice. Each value fetched from global memory is used 16 times. In part 1: 1.18 million loads instead of 18.9 million, and since they are also coalesced, 4.6 MB fetched instead of 166 MB, 36 times less traffic.

Shared memory has its own rule: it is divided into 32 **banks** of 4-byte words, and a warp's 32 accesses are served at once only if they hit different banks (or the same word). In the inner loop, the 16 lanes reading `B`'s tile at `tile[tx][i]` (stride 16 words) hit only 2 banks: 8 lanes per bank, 8 passes. Storing `B`'s tile transposed (`tile[i][tx]`, stride 1) makes the inner loop conflict-free; the stores now conflict instead, but there is one store per 16 reads. Part 1: 4.29 passes ("wavefronts") per shared access, against 1.21.

A real GPU matrix product goes much further: larger tiles (128 × 128 per block), each thread computing several outputs in registers (chapter 5's register blocking), loading the next tiles while computing the current ones, and above all using **tensor cores**, which multiply whole tiles in one instruction. A modern `bf16` GEMM reaches 70-80% of the 989 TFLOP/s peak; the simple tiled kernel above, without tensor cores, would reach a few percent.

### 3.5 Inference on GPUs

The roofline (chapter 4) applies unchanged, with different numbers. Part 2 computes, from published specifications, the least time to decode one token at batch size 1 (read every `bf16` weight once), the batch size at which decoding becomes compute-bound (the ridge point: each weight byte does one FLOP per sequence), and the least time to prefill 1,000 tokens of an 8B model:

```text
== 2. roofline estimates from published specifications (bf16 weights; not measured)
                       ridge   decode at batch 1, least time per token          compute-bound  prefill 1,000
   device         (FLOP/byte)  SmolLM2-135M  SmolLM2-360M        8B       70B   from batch   tokens, 8B
   H100 SXM               295       0.08 ms       0.22 ms   4.78 ms   too big          295        16 ms
   A100 SXM 80 GB         153       0.13 ms       0.36 ms   7.85 ms   too big          153        51 ms
   RTX 4090               164       0.27 ms       0.72 ms  15.87 ms   too big          164        97 ms
   L4                     403       0.90 ms       2.41 ms  53.33 ms   too big          403       132 ms
   this CPU (4 cores)      14       6.75 ms      18.10 ms   too big   too big           14     29630 ms
```

What it says:

- **Decoding at batch 1 is memory-bound everywhere.** An 8B model needs at least 4.8 ms per token on an H100: 209 tokens per second at most, whatever its 989 TFLOP/s. Hence quantization (chapters 18-19: half the bytes, half the time) and speculative decoding (chapter 26: more tokens per pass).
- **Batching pays far longer on a GPU.** The ridge point is 295 FLOP/byte on an H100 (153 on an A100, 403 on an L4), against 14 on this CPU. A GPU decodes hundreds of sequences per step before it runs out of arithmetic, which is why GPU servers run large batches and why KV-cache memory (chapter 24) becomes the limit first.
- **Prefill is compute-bound.** 1,000 tokens of an 8B model need 16 TFLOP: at least 16 ms on an H100, 30 seconds on this CPU.
- **Small models are limited by something else.** SmolLM2-135M's weights take 0.08 ms to read on an H100. But a forward pass is about ten kernels per layer, 300 for 30 layers, and launching a kernel costs a few microseconds of CPU and GPU time: about a millisecond, ten times the memory time. Engines record a whole decode step once as a **CUDA graph** and replay it with a single launch, and **fuse** kernels (norm, projection and activation in one) so that there are fewer to launch and intermediate results stay on chip.
- **Memory is a hard limit.** A 70B model in `bf16` (140 GB) fits on none of these GPUs: chapter 28's parallelism, or quantization, is required.

Attention has its own GPU story: FlashAttention (chapter 20) was invented to keep attention's tiles in shared memory instead of writing the score matrix to HBM, and paged attention's kernels (chapter 24) read KV blocks through a block table, exactly as on the CPU.

### 3.6 Rust on GPUs: cuda-oxide

GPU kernels are traditionally written in CUDA C++. NVIDIA's **cuda-oxide** ([book](https://nvidia.github.io/cuda-rust/cuda-oxide/), repository [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust)) is a Rust compiler backend that compiles ordinary Rust functions marked `#[kernel]` to PTX, embeds the PTX in the host program, and generates typed launch functions. Version 0.1.0, released in 2026, is described by its authors as an early-stage alpha. Its documentation lists the requirements: Rust 1.99.1 (pinned by the project; NVIDIA's announcement mentions a pinned nightly), LLVM 21 or later with the NVPTX backend, CUDA 13.0 or later with driver 580 or later, an Ampere or newer GPU (`sm_80`+), and Linux (Ubuntu 24.04 tested).

A kernel looks like the simulator's, with the GPU's own primitives:

| simulator (`src/kernels.rs`) | cuda-oxide (`oxide/main.rs`) |
|---|---|
| `t.index`, `t.block` | `thread::threadIdx_x()`, `thread::blockIdx_x()` (`u32`) |
| `b.each(...)` between barriers | straight-line code |
| `b.sync_threads()` | `thread::sync_threads()` |
| `b.warp_sum(...)` | five `warp::shuffle_xor_f32(sum, offset)` |
| `t.shared_write(i, v)` | `unsafe { TILE[i] = v; }` on a `static mut SharedArray` |
| `y[row] = v` | `unsafe { *y.get_unchecked_mut(row) = v; }` on a `DisjointSlice` |

Two design choices stand out. Outputs are `DisjointSlice<T>`, a slice whose safe accessor (`get_mut`) only hands a thread the element at its own index, so that two threads cannot write the same element without `unsafe`; a kernel like the warp-per-row product, where lane 0 writes at a computed index, must say `unsafe` and take responsibility for disjointness. And shared memory is a `static mut` array, accessed in `unsafe` blocks: Rust's aliasing rules cannot see the barriers that make those accesses safe.

The project also has a second track, **cutile-rs**, for NVIDIA's tile programming model: kernels describe operations on whole tiles, with no threads, shared memory or barriers to manage, and the compiler maps them to the hardware. Outside NVIDIA, Rust can target GPUs through rust-cuda (another codegen backend), cudarc (host-side bindings to the CUDA driver, for loading kernels written in any language), and wgpu with WGSL shaders (portable across vendors).

## 4. The code

The simulator is in [`src/sim.rs`](src/sim.rs), the kernels in [`src/kernels.rs`](src/kernels.rs), the roofline in [`src/roofline.rs`](src/roofline.rs), the demo in [`src/main.rs`](src/main.rs), the cuda-oxide version in [`oxide/main.rs`](oxide/main.rs).

### 4.1 Running a segment

<!-- file: src/sim.rs -->
```rust
    pub fn each(&mut self, mut f: impl FnMut(&Thread<'_>)) {
        for index in 0..self.dim {
            let t = Thread {
                index,
                block: self.index,
                block_dim: self.dim,
                grid_dim: self.grid_dim,
                rec: &self.rec,
                shared: &self.shared,
            };
            f(&t);
        }
        self.flush();
    }
```

One segment, all threads, in order. Each `Thread` carries its indices and references to the block's recorder and shared memory; `flush` then turns the recorded accesses into sectors and wavefronts. The kernel is an ordinary closure: it can capture output slices and per-thread arrays mutably, because the simulator calls it one thread at a time. That convenience is also the simulator's blind spot: two threads writing the same element of a *global* output would not be caught (only shared memory is checked), which is what cuda-oxide's `DisjointSlice` guards against on the GPU.

### 4.2 Shuffles

<!-- file: src/sim.rs -->
```rust
    pub fn warp_sum(&self, v: &mut [f32]) {
        for warp in v.chunks_mut(WARP) {
            let mut offset = WARP / 2;
            while offset > 0 {
                let before = warp.to_vec();
                for (lane, x) in warp.iter_mut().enumerate() {
                    *x += before[lane ^ offset];
                }
                offset /= 2;
            }
        }
    }
```

A shuffle reads the partner's register *as it was before the instruction*: all 32 lanes exchange at once. Hence the copy (`before`): updating in place would let lane 1 read lane 0's already-updated value. `lane ^ offset` pairs lanes whose indices differ in one bit; after offsets 16, 8, 4, 2 and 1, every lane has added all 32 values, in the same tree order a GPU would use.

### 4.3 Counting sectors

<!-- file: src/sim.rs -->
```rust
        let sector = (m.base + i * 4) / SECTOR;
        rec.global
            .entry((self.warp(), n))
            .or_default()
            .insert(sector);
```

Each load is filed under (warp, the lane's `n`-th load in this segment): the loads of one warp instruction. The number of distinct sectors in each set is what the memory system delivers for that instruction. Each array has a base address, so that two arrays never share a sector by accident.

### 4.4 The coalesced matrix-vector product

<!-- file: src/kernels.rs -->
```rust
        b.each(|t| {
            let row = t.block * warps + t.warp();
            if row < rows {
                let mut sum = 0.0;
                let mut c = t.lane();
                while c < cols {
                    sum += t.load(w, row * cols + c) * t.load(x, c);
                    c += WARP;
                }
                partial[t.index] = sum;
            }
        });
        b.warp_sum(&mut partial);
```

Eight warps per block, one row per warp; `row` depends only on the warp, so all 32 lanes take the branch together, which shuffles require. Lane 0 then writes the row's result. The same code for cuda-oxide:

<!-- file: oxide/main.rs -->
```rust
        if row < rows {
            let mut sum = 0.0f32;
            let mut c = lane;
            while c < cols {
                sum += w[row * cols + c] * x[c];
                c += WARP;
            }
            let mut offset = 16;
            while offset > 0 {
                sum += warp::shuffle_xor_f32(sum, offset);
                offset /= 2;
            }
            if lane == 0 {
                unsafe {
                    *y.get_unchecked_mut(row) = sum;
                }
            }
        }
```

Weights are `f32` here: cuda-oxide 0.1's documentation lists no scalar `bf16` type for kernel code, only packed `bf16x2` operations. A production kernel would read `bf16` (or 4-bit) weights to halve (or quarter) the bytes, which is the whole point for a memory-bound product.

### 4.5 Launching from the host

<!-- file: oxide/main.rs -->
```rust
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(BLOCK / WARP) as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
```

A grid of `rows / 8` blocks of 256 threads, one row per warp. The launch itself, `module.gemv(&stream, config, &w_dev, &x_dev, &mut y_dev, rows as u32, cols as u32)`, is generated by `#[cuda_module]` from the kernel's signature: slices become device buffers, scalars are passed by value. It is `unsafe`, because nothing checks that the grid covers the output and stays within it; cuda-oxide's *launch contracts* can declare the grid's shape so that a safe launch can check it. Launches are asynchronous: `to_host_vec` waits for the GPU to finish before copying the result back.

## 5. Run it

```bash
cargo test -p ch29-gpu
cargo run --release -p ch29-gpu                     # both parts, about 10 seconds
cargo run --release -p ch29-gpu -- roofline         # or: traffic (needs the model)
```

The tests check the simulator itself (32 lanes reading consecutive floats touch 4 sectors, strided reads 32, the same address 1; shared accesses with strides 1, 2 and 32 take 1, 2 and 32 wavefronts; a read of another thread's write without a barrier is a race, with a barrier it is not; warp sums); that both matrix-vector kernels compute `W·x` for sizes that are not multiples of 32 or 256, with the same number of loads and very different efficiency; that RMSNorm matches chapter 8's on three rows, with 9 barriers per row and no race; that the tiled matrix products (with and without the transposed tile) match the naive one with over 10 times fewer loads and no race; and the roofline arithmetic for an H100.

To run the cuda-oxide kernels on an NVIDIA GPU (Ampere or newer, Linux): install cuda-oxide's tool as its book describes (`cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide` at the time of writing), check the setup with `cargo oxide doctor`, create a project with `cargo oxide new llm_kernels`, replace its `src/main.rs` with [`oxide/main.rs`](oxide/main.rs), and run `cargo oxide run`. It prints each kernel's maximum error against the CPU and the matrix-vector product's time per launch. **That output is not shown here, because it was never run.**

## 6. The Rust behind it

**`RefCell` for state that a shared reference must change.** Each `Thread` holds `&RefCell<Recorder>`: recording a load needs to mutate the recorder through a shared reference, and the borrow is checked at run time (`borrow_mut` panics if a borrow is already active). Kernels never hold a borrow across calls, so the check never fails.

**Borrowing disjoint fields.** `each` creates threads that borrow `self.rec` and `self.shared` while `self` is borrowed mutably by the method: allowed, because the threads are dropped before `flush` needs `&mut self`, and the borrow checker tracks fields separately.

**Closures as kernels.** A kernel passed to `launch` is `impl FnMut(&mut Block)`, and each segment is an `impl FnMut(&Thread)`. A closure can capture the output slice mutably because the simulator calls it sequentially: Rust would refuse the same capture for closures running on several threads at once (`FnMut` is not `Sync`), which is the type system saying what `DisjointSlice` says on the GPU.

**`unsafe` where the compiler cannot see the reason.** In cuda-oxide, shared memory is a `static mut`; every access is `unsafe`, because its safety depends on barriers that the type system does not model. The same holds for writing a computed index of a `DisjointSlice`. The kernel's `unsafe` blocks mark exactly the lines whose correctness the simulator's race detector and the CPU comparison check.

**Rust compiled for another processor.** cuda-oxide is a `rustc` codegen backend: the same front end (types, borrow checking, generics) feeds a different code generator, as the course's ARM build did for NEON (chapter 6). What a kernel can use is limited by what the GPU can do: no heap (`Vec`, `Box`, `String`), no `println!` (cuda-oxide offers `gpu_printf!`), no trait objects.

## 7. Mistakes you will make

- **One thread per row for a matrix-vector product.** The arithmetic is right and three quarters of the memory traffic is wasted (section 3.3).
- **A barrier inside a branch that some threads skip.** The threads that skip it never arrive; the block deadlocks or misbehaves. Put barriers where every thread of the block passes.
- **A missing barrier.** Reading shared memory before the writer has written. The result is right most of the time, which is the worst kind of bug; a race detector finds it.
- **Shuffles in divergent code.** All lanes taking part must execute the shuffle; branch on something uniform across the warp.
- **Ignoring bank conflicts.** A column-wise walk through a shared tile can make every access take 8 or 16 passes.
- **Too few blocks or threads.** A grid smaller than the GPU leaves SMs idle; too few warps per SM leaves memory latency unhidden.
- **Launching hundreds of tiny kernels per token.** For small models the launch overhead dominates; use CUDA graphs and fused kernels.
- **Timing without synchronizing.** Launches return immediately; measure after the GPU has finished.

## 8. How the professionals do it

- **cuBLAS and CUTLASS** provide matrix products on tensor cores; CUTLASS (C++ templates) and its Python front end, CuTe DSL, are the basis of many custom kernels. Hopper's `wgmma` instructions and its Tensor Memory Accelerator (TMA) copy tiles asynchronously from global to shared memory.
- **Triton** (OpenAI) lets kernels be written in Python at the level of tiles; the compiler handles the threads, shared memory and tensor cores. PyTorch's compiler generates Triton kernels, and vLLM and SGLang use many hand-written ones.
- **FlashAttention** (Dao et al., 2022; versions 2 and 3) and **FlashInfer** provide attention kernels for prefill, decode and paged KV caches.
- **Fusion and CUDA graphs**: TensorRT-LLM compiles whole models into fused kernels; vLLM and SGLang capture decode steps as CUDA graphs for each batch size.
- **llama.cpp** has CUDA, Metal, Vulkan and other back ends, with its own kernels for quantized weights.
- **Rust**: besides cuda-oxide and cutile-rs, the Candle and Burn frameworks run models on GPUs through their own kernels and cuBLAS, and wgpu reaches GPUs from every vendor through WebGPU.

## 9. Exercises

1. **Vectorized loads.** Change the warp-per-row product so each lane loads 4 consecutive floats per step (`c = 4·lane, 4·lane + 128, ...`). Count sectors in the simulator. Why does a real GPU benefit even though the sector count is the same?
2. **Remove a barrier.** Delete the barrier after the partial sums in `rms_norm`, run the test, and read what the race detector reports. Then remove the one inside the halving loop instead.
3. **Register tiling.** Make each thread of the tiled product compute a 2 × 2 block of outputs (blocks of 8 × 8 threads for a 16 × 16 tile). Count loads, shared accesses and wavefronts. What does each thread now reuse from registers?
4. **Bf16 weights.** Store `W` as `u16` (`bf16` bits) in the simulator's matrix-vector product, widening each value as in chapter 6. How do requested and fetched bytes change? How would you write the load in cuda-oxide without a `bf16` type?
5. **Decode estimates.** Using part 2's numbers, how many tokens per second could one H100 produce for an 8B model at batch 1, 64 and 512? At what batch does the KV cache (chapter 24; say 128 KB per token for an 8B model with grouped-query attention, sequences of 2,000 tokens) exceed the memory left after the weights?
6. **Port a kernel.** Write the simulator's RMSNorm with warp shuffles instead of shared memory for the first five halvings (each warp reduces its 32 values, then only the 8 warp sums go through shared memory). How many barriers remain?

## 10. Check yourself

1. What is a warp, and why does it matter for how memory is accessed?
2. Why did the thread-per-row product use only 24% of the bytes it fetched, and the warp-per-row product 100%?
3. What does a barrier guarantee, and what happens if a thread reads shared memory too early?
4. Why does the tiled product fetch 36 times less than the naive one?
5. Why did transposing `B`'s tile reduce bank conflicts?
6. Why is decoding at batch size 1 memory-bound on every GPU in part 2, and from what batch size is it compute-bound on an H100?
7. Why is SmolLM2-135M on an H100 not limited by memory bandwidth in practice?
8. What does `DisjointSlice` prevent, and why does writing at a computed index need `unsafe`?

## 11. Recap

- A GPU is many SMs running warps of 32 lanes in lockstep, hiding memory latency with other warps; its memory (3.35 TB/s on an H100) and tensor cores (989 TFLOP/s) are many times a CPU's.
- Kernels run as grids of blocks of threads; threads find their work from their indices, and synchronize within a block at barriers.
- A simulator that runs each barrier-delimited segment for every thread in turn checks kernels on a CPU, counts the sectors and bank wavefronts they would cost, and detects shared-memory races.
- Coalescing: one row per warp used 100% of the fetched bytes, one row per thread 24%, for the same arithmetic.
- Tiling through shared memory cut a matrix product's loads 16 times and its traffic 36 times; transposing a tile cut bank conflicts from 4.29 to 1.21 passes per access.
- By the roofline, decoding at batch 1 is memory-bound on every GPU (an 8B model needs at least 4.8 ms per token on an H100), batching pays up to the ridge point (295 on an H100, against 14 on this CPU), and prefill is compute-bound; small models are limited by kernel launches.
- cuda-oxide compiles Rust kernels to PTX; the chapter's kernels are written for it, mirroring the simulator's, but were not compiled or run here.

## Answers

**Exercises**

1. Each lane reads 16 bytes per load, and the warp 512 contiguous bytes per instruction: 16 sectors for 128 values instead of 4 sectors for 32, so the same 100% efficiency. A GPU still benefits because it issues a quarter of the load instructions (one 128-bit load instead of four 32-bit ones), leaving the schedulers more room for arithmetic; the simulator counts sectors, not instructions.
2. Without the first barrier, the first halving step reads partial sums that other threads write in the same segment: the detector reports read-after-write races. Without the one in the loop, a step reads values the previous step's threads write: again races. On a GPU both would usually give the right answer and sometimes not, depending on timing.
3. Each thread loads, per slice of `k`, 2 values of `A` and 2 of `B` from shared memory per `i` and does 4 multiply-adds with them, instead of 2 loads for 1 multiply-add: shared-memory traffic per output halves. Global loads per output also halve (the block has 4 times fewer threads loading the same tile). The two `A` values and two `B` values are reused from registers.
4. Requested bytes for `W` halve (2 bytes per value), and so do fetched bytes with coalescing (32 lanes read 64 bytes: 2 sectors). In cuda-oxide, read the weights as `&[u16]` and widen each with `f32::from_bits((w as u32) << 16)` (chapter 2's conversion), or read pairs and use the packed `bf16x2` operations the documentation lists; this is untested.
5. Batch 1: at most 1 / 4.78 ms ≈ 209 tokens per second. Batch 64: the weights are read once per step, still memory-bound (64 < 295), so about 64 × 209 ≈ 13,000 tokens per second minus the KV cache reads. Batch 512 is beyond the ridge: compute-bound at 989 TFLOP/s / 16 GFLOP per token ≈ 62,000 tokens per second at best. Memory: 80 − 16 = 64 GB left; 2,000 tokens × 128 KB = 256 MB per sequence, so about 250 sequences fit, before the batch reaches the ridge point: memory is the limit first.
6. Each warp sums its 32 partial sums with five shuffles (no barrier), lane 0 of each warp writes one value to shared memory, one barrier, then the first warp reads the 8 values and sums them with three more shuffles, and one more barrier (or a broadcast through shared memory) to share the total: 2 barriers instead of 9.

**Check yourself**

1. 32 threads that execute each instruction together; a warp's 32 loads are one memory instruction, served by as few or as many 32-byte sectors as their addresses require.
2. Thread-per-row: at each step the 32 lanes read 32 rows, each in its own sector, using 4 bytes of each 32 (the loads of `x` were fully used, which brings the total to 24%). Warp-per-row: the 32 lanes read 32 consecutive floats, 4 whole sectors.
3. That every thread of the block has finished everything before the barrier before any thread continues past it. Reading early returns whatever was there before (stale or uninitialized data), depending on timing.
4. Each value loaded into a shared tile is used by 16 threads (reuse cuts loads 16 times), and the tile loads are coalesced (the naive kernel's loads of `B` used one value per sector).
5. Lanes reading a tile column with stride 16 words hit words in only 2 of the 32 banks (8 lanes per bank, 8 passes); with the tile transposed, the 16 lanes read 16 consecutive words in 16 banks, in one pass.
6. At batch 1 every weight byte is used for one FLOP, far below every GPU's ridge point (153 to 403 FLOP/byte); on an H100 decoding becomes compute-bound from about 295 sequences per step.
7. Its weights take 0.08 ms to read, but a forward pass launches hundreds of kernels, each costing a few microseconds: launch overhead, not bandwidth, sets the time (unless CUDA graphs and fusion remove it).
8. Two threads writing the same element of an output slice: the safe accessor only gives a thread the element at its own index. Writing at a computed index (lane 0 writing a row's result) is outside what the type can check, so the programmer asserts disjointness in an `unsafe` block.

## Further reading

- NVIDIA, *CUDA C++ Programming Guide* (the execution and memory model) and *CUDA C++ Best Practices Guide* (coalescing, shared memory, occupancy).
- NVIDIA, *The cuda-oxide Book*, https://nvidia.github.io/cuda-rust/cuda-oxide/.
- Kirk, Hwu and El Hajj, *Programming Massively Parallel Processors*, 4th edition, 2022.
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022.
- Tillet, Kung and Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL 2019.
- NVIDIA, "NVIDIA H100 Tensor Core GPU Architecture" whitepaper, 2022.
- Next: [Chapter 30: Inference in production](../30-production/README.md). Running all of this as a service: capacity, debugging, metrics and reliability.
