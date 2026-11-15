# Chapter 28: More than one device

> **In one sentence:** when a model does not fit on one device, or one device is too slow, it is split: **tensor parallelism** splits every layer across devices and adds up their partial results twice per layer (faster tokens, constant communication), **pipeline parallelism** gives each device a block of consecutive layers (little communication, but no faster tokens and idle devices unless several requests are in flight), and **expert parallelism** spreads a mixture-of-experts layer's experts across devices (tokens travel to their experts, and the busiest expert sets the pace); all three rest on a few collective operations whose cost is a latency per step plus bytes over bandwidth.

**Where this fits:** every chapter so far ran on one processor. A 70-billion-parameter model has 140 GB of `bf16` weights, more than the 80 GB of an H100, so the largest models cannot run on one device at all. This chapter splits SmolLM2 the three ways real engines do, on threads standing in for devices, and checks each split against the single-device model. Chapter 29 then looks at the devices themselves: GPUs.

**You need:** chapter 7 (threads, scoped threads), chapter 12 (attention heads and grouped-query attention), chapter 13 (the transformer's layers), chapter 14 (the forward pass and its `Matrix` trait), chapter 23 (packed weights).

**You will build:** simulated devices that share nothing and communicate through a communicator with ring and naive all-reduce, all-gather and all-to-all; tensor parallelism that reuses chapter 14's forward pass unchanged, with the communication hidden inside a matrix type; pipeline stages with a scheduler that keeps several sequences in flight; and a mixture-of-experts layer with expert parallelism and expert capacity. Then measurements on SmolLM2-135M, and estimates for real interconnects (clearly marked as such).

---

## 1. The intuition

A restaurant gets a second and third cook. There are three ways to use them:

- **Everyone works on every dish** (tensor parallelism). Each cook prepares a third of each dish, and the three thirds are put together before the next step. A dish is ready much sooner, but the cooks must hand things over constantly, so they must stand next to each other.
- **An assembly line** (pipeline parallelism). One cook does starters, the next mains, the third desserts. A single dinner is not ready any sooner, and while it is on the second station, the first and third cooks are idle. With many dinners in flight, every station is busy.
- **Specialists** (expert parallelism). The pastry cook makes every dessert, the grill cook every steak. Each order goes to the right specialists and comes back. On a night when everyone orders steak, the grill cook is swamped and the others wait.

**Where the analogy breaks:** a cook's handover takes no time; a transfer between devices costs a fixed delay plus time proportional to its size, and for the small transfers of decoding, the fixed delay dominates. And the "devices" in this chapter are threads on one machine sharing one memory: a transfer here is a copy, and the chapter says which numbers that makes meaningful and which it does not.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Device, rank** | One processor in a group (a GPU, here a thread); its rank is its number in the group. |
| **Communicator** | A device's connections to the others, through which collectives run (NCCL and MPI's term). |
| **Collective** | An operation every device of a group takes part in: all-reduce, all-gather, all-to-all. |
| **All-reduce** | Every device contributes a buffer; every device ends with their element-wise sum. |
| **Ring all-reduce** | All-reduce in `2 (n − 1)` steps around a ring, each device sending `2 (n − 1) / n` buffers in total. |
| **All-gather** | Every device contributes a piece; every device ends with all the pieces. |
| **All-to-all** | Every device sends a different piece to every other device. |
| **Tensor parallelism (TP)** | Splitting each weight matrix across devices; the devices work on every token together. |
| **Column-, row-parallel** | Splitting a matrix by output rows (each device computes some outputs) or by input columns (each device computes a partial sum of every output). |
| **Pipeline parallelism (PP)** | Giving each device a block of consecutive layers (a **stage**). |
| **Bubble** | Time a pipeline stage sits idle, waiting for work from the stage before. |
| **Data parallelism** | Whole copies of the model on several devices, each serving different requests. |
| **Mixture of experts (MoE)** | A layer with many MLPs (experts) and a router that sends each token to a few of them. |
| **Expert parallelism (EP)** | Spreading the experts across devices and sending tokens to them. |
| **Capacity factor** | How many tokens an expert accepts per batch, relative to an even share; tokens beyond it are dropped. |
| **α-β model** | Transfer time = α (latency per message) + bytes / β (bandwidth). |

## 3. The concepts in depth

### 3.1 Why more than one device

**Memory.** A model's weights must sit in the device's memory: 2 bytes per parameter in `bf16`. A 70B model needs 140 GB, plus its KV cache (chapter 24); an H100 has 80 GB. It needs at least two devices whatever their speed.

**Speed.** Decoding reads every weight for every token (chapter 23), so a device's memory bandwidth bounds tokens per second: 140 GB at 3.35 TB/s is 42 ms per token on one H100, if it fitted. Eight devices each reading an eighth of the weights could do it in about 5 ms.

**Throughput.** More requests per second need more devices. If the model fits on one device, the simplest answer is **data parallelism**: identical copies, each serving its own requests, with no communication at all. It does not help a model that does not fit, nor make one request faster.

The other three schemes split one copy of the model. All of them move data between devices, so they rest on a handful of collective operations.

### 3.2 Collectives, and what they cost

The most important is **all-reduce**: each device has a buffer, and each must end with the sum of all of them. Naively, every device sends its buffer to device 0, which adds them and sends the result back. Device 0 then sends and receives `n − 1` whole buffers, and everyone waits for it.

The **ring all-reduce** spreads the work. Cut each buffer into `n` chunks. In `n − 1` steps, every device sends one chunk to its right neighbour and adds the chunk it receives from its left one; after these steps, each device holds one chunk summed over all devices (a *reduce-scatter*). In `n − 1` more steps, the summed chunks travel around the ring until everyone has all of them (an *all-gather*):

```text
4 devices, chunks A B C D; after reduce-scatter, device 0 holds B summed, 1 holds C, 2 D, 3 A
step:   1           2           3      (reduce-scatter: send one chunk right, add the one from the left)
        4           5           6      (all-gather: send the summed chunk right, keep the one from the left)
```

Every device sends `2 (n − 1)` chunks of `1/n` of the buffer: `2 (n − 1) / n` buffers in total, under twice the buffer whatever `n` is, and every link is busy at every step. The price is `2 (n − 1)` sequential steps, each paying the link's latency.

That gives the **α-β model** of a ring all-reduce of `B` bytes:

```text
time = 2 (n − 1) · (α + B / (n · β))
```

with α the latency of one step and β the bandwidth of one link. Large buffers are bandwidth-bound (the second term); small ones are latency-bound (the first).

Part 1 measures both algorithms with threads. A "link" here is a memory copy through a channel, so the times only show the shape; the bytes are exact:

```text
== 1. all-reduce of one buffer per device (threads; a transfer is a memory copy)
   devices   buffer       ring: time   sent by each      naive: time   sent by device 0
         2         2 KB       0.03 ms            2 KB          0.03 ms               2 KB
         2       256 KB       0.07 ms          256 KB          0.09 ms             256 KB
         2     16384 KB       7.93 ms        16384 KB         10.67 ms           16384 KB
         4         2 KB       0.11 ms            3 KB          0.05 ms               6 KB
         4       256 KB       0.15 ms          384 KB          0.14 ms             768 KB
         4     16384 KB       8.06 ms        24576 KB         21.54 ms           49152 KB
```

With 16 MB on 4 devices, the ring is 2.7 times faster than the naive version (2.0 to 3.7 across three runs): each device sends 24 MB (1.5 buffers) against 48 MB for device 0 alone. With 2 KB, the ring is *slower* (0.11 ms against 0.05): six steps of latency against two. Real libraries choose the algorithm by message size for this reason (NCCL uses rings, trees, or NVLink's switch hardware).

### 3.3 Tensor parallelism

Split every layer across `n` devices so that each device does a third (for `n = 3`) of every matrix product, with as little communication as possible. Megatron-LM's scheme (Shoeybi et al., 2019) uses two kinds of split:

- **Column-parallel** (by output rows of `W`, in this course's `y = x · Wᵀ` convention): each device holds some of the output features. `wq`, `wk`, `wv` are split by attention head, `w_gate` and `w_up` by MLP unit. Each device computes its outputs from the full input; no communication.
- **Row-parallel** (by input columns): `wo` takes the attention heads as input and `w_down` the MLP units, which each device already has only its share of. Each device multiplies its share and gets a *partial sum* of the full output; an **all-reduce** adds them.

```text
                     device r (of 3, SmolLM2-135M)
x (576) ─ norm ─► wq rows 192r.. (3 heads), wk/wv rows 64r.. (1 KV head) ─► attention on 3 heads
                ─► wo columns 192r.. ─► partial sum (576) ─► ALL-REDUCE ─► + x
x (576) ─ norm ─► w_gate, w_up rows 512r.. ─► SwiGLU (512) ─► w_down columns 512r..
                ─► partial sum (576) ─► ALL-REDUCE ─► + x
```

Attention needs no communication in between, because heads are independent: each device attends with its own heads over its own part of the KV cache. With grouped-query attention, a device must hold the KV heads its query heads use. SmolLM2-135M has 9 query heads and 3 KV heads, so 3 devices get exactly 3 query heads and 1 KV head each; 2 devices would have to split a KV head's group, which this code does not do. The KV cache is split too: each device stores a third of it.

So: **two all-reduces per layer**, each of one hidden state per token (576 floats). The embedding and norms are small and copied to every device; the LM head (the largest matrix) is split by vocabulary, and an **all-gather** puts the logits together.

Part 2 runs SmolLM2-135M on 1 and 3 devices, one core each:

```text
== 2. tensor parallelism, SmolLM2-135M, one core per device: 64 tokens after a 30-token prompt
   devices   per token   in communication   messages   bytes sent, per device and token
         1    39.48 ms            0.00 ms          0            0 B
         3    19.31 ms            4.91 ms        242       315392 B
   3 devices against 1: largest logit difference 8.5e-5; same 64 tokens: true
```

Three devices produce a token in 19.3 ms instead of 39.5 (2.0 times faster; 1.8-2.0 across runs), not 3: each token needs 60 all-reduces and an all-gather, 242 messages, and 4.9 ms per token are spent in communication. That time includes *waiting*: each all-reduce is a meeting point, and the slowest device decides when it ends; 61 meeting points per token are 61 chances for one device to be late. The logits differ from the single device's by at most 8.5 × 10⁻⁵, because the partial sums are added in a different order (chapter 2), and the 64 tokens are the same.

### 3.4 Pipeline parallelism

Give device `s` a block of consecutive layers: with 3 stages, layers 0-9, 10-19 and 20-29. A token's hidden state goes through stage 0's layers, is sent to stage 1, and so on; the last stage computes the logits and sends the chosen token back to stage 0, which feeds it in as the next input. Communication is one hidden state per stage boundary per token: almost nothing.

But a token cannot start through stage 0 before the previous token of the same sequence has come out of the last stage. With one sequence, exactly one stage works at a time and the others wait: the **bubble**. Pipelining only helps when several sequences (or micro-batches) are in flight, so that stage 1 works on one while stage 0 works on the next.

Part 3 decodes 32 tokens for 1 and for 3 sequences, on 1 to 3 stages of one core each. Prompt tokens do not depend on the output and stream through the stages freely; the decoding phase, timed separately, is where the bubble shows:

```text
== 3. pipeline parallelism, SmolLM2-135M, one core per stage: 32 new tokens per sequence
   stages   sequences   decoding: tokens/s   between a sequence's tokens   stages busy
        1           1                 27.2                      36.80 ms          100%
        2           1                 22.5                      44.48 ms       56% 71%
        3           1                 20.8                      48.11 ms   37% 40% 59%
        1           3                 22.7                     121.96 ms          100%
        2           3                 36.8                      79.28 ms       81% 99%
        3           3                 49.4                      58.78 ms   72% 71% 99%
   (the answers are the same with 1, 2 and 3 stages)
```

With one sequence, more stages make decoding no faster; slightly slower, from the hand-overs (36.8 ms between tokens on one stage, 44 to 48 ms on two or three). With three sequences in flight, throughput grows to 1.6 times on two stages and 2.2 times on three (1.4-1.6 and 1.9-2.2 across runs), and each sequence's tokens come twice as fast.

"Stages busy" shows a second problem. The last stage is busy 99% of the time, the others about 70%: it does its 10 layers *and* the LM head, which on SmolLM2-135M is a 49,152 × 576 matrix, 21% of the model's weights. A pipeline runs at the pace of its slowest stage, so real deployments balance stages by cost, not by layer count (exercise 3).

Pipeline parallelism's real use is capacity: each device holds only its stages' weights and KV cache, and devices can be on different machines, because the traffic is small. For a single request's speed, tensor parallelism is the tool.

### 3.5 Mixture of experts and expert parallelism

A **mixture-of-experts** layer replaces the MLP by `E` experts (each an MLP) and a router. For each token, the router scores all experts; the `k` best run, and their outputs are added with weights from a softmax over those `k` scores. The layer has `E` experts' worth of parameters, but each token costs only `k` of them: DeepSeek-V3 has 671B parameters and activates 37B per token. Mixtral, Qwen-MoE, DeepSeek and most large open models since 2024 are MoE.

**Expert parallelism** spreads the experts across devices. Each device starts with some of the batch's tokens, routes them, sends each (token, expert) pair to the device holding that expert (an **all-to-all**), runs its experts on what it received, and sends the results back (a second all-to-all), where they are weighted and added.

Part 4 builds an MoE layer with 8 random experts of width 768 and top-2 routing, over 1,024 real token embeddings of the novel, on 4 devices of 2 experts each:

```text
== 4. an MoE layer: 8 experts of width 768, top-2, on 1024 token embeddings of the novel
   tokens per expert: [183, 207, 236, 362, 151, 231, 429, 249] (an even split: 256)
   one device: 172.68 ms
   capacity factor   dropped   time    busiest device's experts   all-to-all, sent per device
                 1     13.6%  50.28 ms                   47.71 ms                    1983 KB
              1.25      7.4%  56.80 ms                   54.14 ms                    1983 KB
                 2      0.0%  84.96 ms                   82.51 ms                    1983 KB
              none      0.0%  77.66 ms                   75.04 ms                    1983 KB
```

The router does not spread tokens evenly: expert 6 gets 429 tokens and expert 4 gets 151, against an even share of 256. Without a limit, the device holding experts 6 and 7 (678 tokens) works the longest, and four devices are only 2.2 times faster than one (2.0 to 2.7 across three runs). A **capacity** bounds every expert's work per batch: with a capacity factor of 1 (256 tokens), the slowest device finishes much sooner, but 13.6% of the (token, expert) pairs are dropped: those tokens get only one expert, or none, and skip the rest through the residual connection. A factor of 1.25 drops 7.4%; 2 drops nothing here. Accelerators prefer fixed sizes, which is why capacities exist; training adds a load-balancing objective so that routers spread tokens more evenly, and DeepSeek-V3 adjusts a per-expert bias instead (exercise 4).

### 3.6 Real links

The threads above talk through memory. Real devices talk through NVLink (between GPUs in one server, hundreds of GB/s), PCIe (within a server, tens of GB/s), or a network (InfiniBand or Ethernet, between servers). Part 5 applies the α-β model to tensor parallelism's decode traffic, with assumed values (α of 1, 5 and 10 µs per step; 400, 25 and 12 GB/s): estimates, not measurements:

```text
== 5. estimated communication per decoded token for tensor parallelism (not measured)
   SmolLM2-135M on 3 devices: 60 all-reduces of 1 KB:  NVLink-class 0.24 ms  PCIe-class 1.20 ms  100 Gb Ethernet 2.41 ms
   a 70B model on 8 devices: 160 all-reduces of 16 KB:  NVLink-class 2.25 ms  PCIe-class 11.38 ms  100 Gb Ethernet 22.78 ms
```

For decoding, the all-reduces are small (one token's hidden state, 16 KB for an 8,192-wide model), so the latency term dominates: 160 all-reduces × 14 ring steps × α. With NVLink-class links, a 70B model on 8 devices spends about 2 ms per token in communication, against about 5 ms reading its weights (140 GB over 8 devices at 3.35 TB/s): significant, but worth it. Over PCIe or Ethernet the communication alone would exceed the computation. Hence the standard layout: **tensor parallelism inside a server** (over NVLink), **pipeline parallelism across servers** (small traffic, tolerant of slower links), **data parallelism** on top for throughput, and expert parallelism wherever the experts are. For SmolLM2-135M, whose weights take 0.03 ms to read on such devices, tensor parallelism would be all communication and no gain: small models are served by data parallelism.

## 4. The code

The communicator is in [`src/comm.rs`](src/comm.rs), tensor parallelism in [`src/tp.rs`](src/tp.rs), pipelines in [`src/pp.rs`](src/pp.rs), experts in [`src/moe.rs`](src/moe.rs), the demo in [`src/main.rs`](src/main.rs).

### 4.1 Ring all-reduce

<!-- file: src/comm.rs -->
```rust
            let (next, prev) = ((self.rank + 1) % n, (self.rank + n - 1) % n);
            for step in 0..n - 1 {
                let send = self.chunk(p, (self.rank + n - step) % n);
                self.send(next, &x[send]);
                let recv = self.chunk(p, (self.rank + n - step - 1) % n);
                let incoming = self.recv(prev);
                for (a, b) in x[recv].iter_mut().zip(&incoming) {
                    *a += b;
                }
            }
```

At step `s`, device `r` sends chunk `r − s` and adds the chunk `r − s − 1` coming from the left (all modulo `n`). At step 0 it sends its own chunk; at each later step it forwards the chunk it has just added to, which now carries one more device's contribution. After `n − 1` steps, the chunk it received last, `r + 1`, has every contribution. The all-gather half then passes the summed chunks around. Every device sends before it receives; with blocking sends that would deadlock if every device sent at once, which is why the channels here have unbounded buffers and why real libraries pipeline sends and receives carefully. `chunk` splits any length into `n` parts, so buffers need not divide evenly.

### 4.2 Communication inside a matrix

<!-- file: src/tp.rs -->
```rust
            After::AllReduce => {
                self.inner.matmul(pool, x, y, m, scratch);
                self.comm.all_reduce(&mut y[..m * self.rows]);
            }
```

A row-parallel slice computes its local product and then all-reduces it. To chapter 14's forward pass, `wo` and `w_down` are matrices whose products happen to be the full sums; it adds them to the residual stream as always. So each device runs chapter 14's `forward_last`, unchanged, on its own `Model` whose config has 3 heads, 1 KV head and an MLP of 512; its KV cache and scratch buffers come out a third of the size by themselves. The vocabulary-parallel LM head does the same with an all-gather.

### 4.3 Slicing the weights

<!-- file: src/tp.rs -->
```rust
    // A column-parallel slice: some rows, every column.
    let column = |w: &W, rows: Range<usize>| {
        let n_rows = rows.len();
        tp(
            rows_of(&to_f32(w), w.cols(), rows),
            n_rows,
            w.cols(),
            After::Nothing,
            n_rows,
        )
    };
```

`shard` reads every matrix through the `Matrix` trait (`row_to_f32`), cuts out this device's rows or columns, and builds the slice with a function the caller chooses: plain `f32` matrices in the tests, chapter 23's packed `bf16` in the demo. Query heads `3r..3r + 3` and KV head `r` go together, because chapter 12 maps query head `h` to KV head `h / group`.

### 4.4 A pipeline stage

<!-- file: src/pp.rs -->
```rust
            loop {
                let msg = comm.recv(s - 1);
                if msg[0] == STOP {
                    if s < last {
                        comm.send(s + 1, &[STOP]);
                    }
                    break;
                }
                let (seq, pos, want) = (msg[0] as usize, msg[1] as usize, msg[2] > 0.0);
                x.copy_from_slice(&msg[3..]);
                match work(seq, pos, want, &mut x, &mut busy) {
```

Every stage after the first runs this loop: receive a hidden state with three numbers of header (sequence, position, whether logits are wanted), run its layers on it with that sequence's KV cache, and pass it on, or, at the last stage, send back the token. Messages between two stages arrive in the order they were sent, so each stage sees each sequence's positions in order and its caches stay consistent. Stage 0 decides what enters: any prompt token immediately, a generated token once the last stage has sent it back, choosing sequences in turn. The stage's layers run in `run_layers`, chapter 14's layer written for one token, because chapter 14's forward pass always starts from token ids and ends with logits, and a middle stage needs neither.

### 4.5 Sending tokens to their experts

<!-- file: src/moe.rs -->
```rust
    let routes = moe.route(pool, x_local, m);
    let mut outgoing = vec![Vec::new(); n];
    let mut sent: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for (t, route) in routes.iter().enumerate() {
        for &(e, w) in route {
            let d = owner(e);
            outgoing[d].push(e as f32);
            outgoing[d].extend_from_slice(&x_local[t * h..(t + 1) * h]);
            sent[d].push((t, w));
        }
    }
    let incoming = comm.all_to_all(&outgoing);
```

Each pair travels as the expert's number and the token's hidden state. The device remembers what it sent where (`sent`), so the results, which come back in the same order, can be matched to their tokens and weights without sending the weights. Experts take pairs in the order of the senders' ranks, which is the batch's token order, so a capacity drops the same pairs as the single-device version; the tests check that the outputs are equal, with and without a capacity.

## 5. Run it

```bash
cargo test -p ch28-parallelism
cargo run --release -p ch28-parallelism                   # all five parts, about 40 seconds
cargo run --release -p ch28-parallelism -- tensor         # or: collectives, pipeline, experts, estimates
```

The tests check that ring and naive all-reduce compute the sum on 1 to 5 devices for buffers of 1 to 1,000 values (including lengths that do not divide evenly), with the expected number of messages and bytes; that all-gather and all-to-all deliver every piece to the right place; that tensor parallelism on 1 and 2 devices gives the single device's logits for a prefill and a decode step, with exactly the expected number of messages; that a pipeline of 1 or 2 stages generates exactly chapter 14's greedy tokens for three sequences; and that expert parallelism on 1, 2 and 4 devices gives the single-device MoE output and expert loads, with and without a capacity.

**What was verified, and how.** Everything that is about correctness ran here: the collectives, the three kinds of parallelism and their outputs are checked against single-device computation. The times measure threads on one 4-core machine, where a "transfer" is a memory copy and "devices" share one memory system; they show the shapes (what scales, where the bubble is, what imbalance costs), not what GPUs would achieve. Part 5's numbers are computed from assumed link parameters, not measured.

## 6. The Rust behind it

**A trait implementation that hides communication.** `Tp<W>` implements chapter 14's `Matrix` trait by wrapping another `Matrix`. The forward pass is generic over `W: Matrix`, so it runs on `Model<Tp<PackedBf16>>` without knowing that some of its products talk to other threads. Generic code written against a trait gets new behaviour without being changed: the same reason chapter 17 could time every matrix.

**Lifetimes that tie data to a thread's scope.** `Tp<'c, W>` holds `&'c Comm`, a reference to its device's communicator, so a sharded model cannot outlive its communicator. `run` creates the communicators and spawns scoped threads (`std::thread::scope`) that borrow them and the model; the scope guarantees the threads end before the borrowed data does.

**`Send` and `Sync` decide what may cross threads.** `Matrix: Send + Sync`, so `Tp` must be `Sync`, so `Comm` must be. A `Sender` is `Sync`; a `Receiver` is not (two threads receiving from one channel would race for messages), so each receiver sits in a `Mutex`. The compiler refuses the program until this is right.

**Moving ownership through channels.** `send` copies a slice into a new `Vec` and moves the `Vec` into the channel; the receiver gets ownership of the buffer, with no further copy. The copy is deliberate: without it, "sending" would cost nothing at all, and the simulation would hide what a transfer costs.

**Atomics for statistics.** Message and byte counters are `AtomicUsize`, updated with `Relaxed` ordering: they are counts read after the threads end, not used to synchronize anything, so no stronger ordering is needed.

## 7. Mistakes you will make

- **Splitting attention heads without their KV heads.** With grouped-query attention, a query head needs the KV head of its group on the same device.
- **Expecting identical bits.** The sums are added in a different order on different device counts; results differ in the last digits, and greedy decoding can, rarely, diverge.
- **Collectives in a different order on different devices.** Every device must call the same collectives in the same order, or two devices wait for each other forever (or, worse, add the wrong buffers).
- **Pipelining a single request.** It is no faster; only several sequences in flight fill the stages.
- **Balancing stages by layer count.** The embedding and the LM head make the first and last stages heavier.
- **Tensor parallelism across slow links.** Two latency-bound all-reduces per layer per token: keep it within NVLink.
- **Ignoring expert load.** The busiest expert sets the pace; without a capacity or balancing, adding devices helps less than expected.
- **Splitting a small model.** If it fits on one device, copies of it (data parallelism) serve more with no communication.

## 8. How the professionals do it

- **Megatron-LM** (Shoeybi et al., 2019) introduced the column/row split used here; it is the tensor parallelism of vLLM (`--tensor-parallel-size`), SGLang, TensorRT-LLM and TGI.
- **NCCL** implements the collectives on NVIDIA GPUs, choosing between ring and tree algorithms and, on NVSwitch systems, in-switch reductions (NVLink SHARP). For the tiny all-reduces of decoding, vLLM and TensorRT-LLM have custom one-shot all-reduce kernels that cut the latency.
- **Pipeline parallelism** was popularized by GPipe (Huang et al., 2019); serving engines support it across nodes (vLLM's `--pipeline-parallel-size`), usually with tensor parallelism inside each node.
- **Mixture of experts:** GShard (Lepikhin et al., 2020) and the Switch Transformer (Fedus et al., 2022) introduced expert capacity and load-balancing losses; Mixtral (2024) and DeepSeek-V3 (2024) are widely deployed MoE models. DeepSeek's DeepEP library implements the all-to-all for expert parallelism over NVLink and RDMA.
- **Long contexts** add sequence or context parallelism: splitting a long sequence's attention across devices, for example with ring attention (Liu et al., 2023), which passes blocks of keys and values around a ring.
- **Disaggregation** (chapter 25) places prefill and decode on different devices, which adds a KV cache transfer between them to the traffic.

## 9. Exercises

1. **Small messages.** Implement *recursive doubling* all-reduce (at step `i`, exchange the whole buffer with device `rank XOR 2^i` and add): `log₂ n` steps, each sending the whole buffer. Compare it with the ring on 4 devices for 2 KB and 16 MB.
2. **Prefill with tensor parallelism.** Time a 512-token prefill on 1 and 3 devices. Is the share of time spent communicating larger or smaller than in decoding, and why?
3. **Balanced stages.** Give the last stage fewer layers to make up for the LM head (for 3 stages, 11, 11 and 8 layers, say), and measure part 3 again.
4. **Load balancing without dropping.** Add a bias per expert to the router's scores, used only to choose the experts (not in the weights), and after each batch lower the bias of experts above the even share and raise it for those below (DeepSeek-V3's "auxiliary-loss-free" balancing). How many batches until the loads are within 10% of even?
5. **Plan a deployment.** A 405B-parameter model in `bf16` on servers of eight 80 GB GPUs. How many GPUs for the weights alone, and how would you combine tensor and pipeline parallelism? Estimate the tensor-parallel communication per token with part 5's model (126 layers, hidden size 16,384).
6. **2D tensor parallelism.** With 6 devices, split SmolLM2's heads 3 ways (as here) and the MLP 6 ways. What extra communication does the mismatch need?

## 10. Check yourself

1. Why can a 70B model not be served by data parallelism on 80 GB devices?
2. How many bytes does each device send in a ring all-reduce of `B` bytes over `n` devices, and how many sequential steps does it take?
3. Why is the ring slower than the naive all-reduce for 2 KB on 4 devices?
4. Which of SmolLM2's matrices are column-parallel, which are row-parallel, and why does attention need no communication in between?
5. How many all-reduces does tensor parallelism need per layer per token?
6. Why does a pipeline not make a single sequence's tokens faster?
7. Why was the last pipeline stage the busiest?
8. What does a capacity factor of 1 trade, and what happened to the dropped tokens?
9. Why is tensor parallelism kept inside a server?

## 11. Recap

- Models are split across devices for memory (a 70B model needs 140 GB) and for speed; data parallelism (copies) adds throughput but does neither.
- Ring all-reduce sends under two buffers per device whatever the device count, in `2 (n − 1)` latency-bound steps; it beat the naive all-reduce 2.0-3.7 times on 16 MB and lost on 2 KB.
- Tensor parallelism splits each layer by heads and MLP units; two all-reduces per layer complete the row-parallel products. On 3 one-core devices, SmolLM2-135M decoded 1.8-2.0 times faster than on one, with 242 messages and about 5 ms of communication per token, and the same tokens.
- Pipeline parallelism splits by layers; one sequence gets no faster (the bubble), three sequences in flight got 1.9-2.2 times the throughput on 3 stages, limited by the last stage, which also computes the LM head.
- Expert parallelism sends tokens to their experts with two all-to-alls; uneven routing (151 to 429 tokens per expert) limited 4 devices to 2.0-2.7 times, and capacities trade dropped tokens (13.6% at factor 1) for time.
- For decoding, tensor parallelism's all-reduces are latency-bound: about 2 ms per token for a 70B model on 8 NVLink-class devices by the α-β model, many times more over slower links. Tensor parallelism stays within a server, pipelines span servers.

## Answers

**Exercises**

1. Recursive doubling takes `log₂ 4 = 2` steps instead of 6, each sending the whole buffer: 2 buffers per device instead of 1.5. For 2 KB, fewer steps win (latency-bound); for 16 MB, the ring's smaller volume wins (bandwidth-bound). NCCL's choice between ring and tree is the same trade.
2. Smaller. The all-reduces carry 512 tokens' hidden states each, but there are still only 60 of them per forward pass, so the latency is paid once per 512 tokens; the computation grows 512 times. Prefill is where tensor parallelism is most efficient; decoding, with one token per all-reduce, is where communication hurts most.
3. The last stage's extra work is the LM head (about 21% of the model's weights, roughly 6 layers' worth on SmolLM2-135M); with 11, 11 and 8 layers the three stages carry similar work, the last stage's busy share drops below 99%, and the three-sequence throughput rises towards 3 times one stage. The best split is found by measuring each stage's time.
4. The bias only changes which experts are chosen, not how much each contributes. Experts above the even share become slightly less likely to be chosen, those below slightly more. With a step of a few percent of the score's spread per batch, loads typically settle within 10% of even after tens of batches; the step size trades speed of balancing against oscillation.
5. 405B × 2 bytes = 810 GB: at least 11 GPUs for the weights, before the KV cache; in practice two servers of 8 (16 GPUs, 1,280 GB). Tensor parallelism of 8 within each server and pipeline parallelism of 2 across them. Per token: 2 × 126 = 252 all-reduces of 32 KB over 8 devices; with α = 1 µs and 400 GB/s, each is 14 × (1 µs + 4 KB / 400 GB/s) ≈ 14.1 µs, about 3.6 ms per token, plus one transfer of a hidden state between the two servers per token.
6. Attention is split 3 ways (each pair of devices holds the same heads) and the MLP 6 ways, so the attention output must be complete on all 6 devices before the MLP: an all-reduce among the devices that hold different heads, plus agreement between the pairs that hold the same ones (or each pair computes the same attention twice). Real systems avoid such mismatches by choosing device counts that divide both.

**Check yourself**

1. Data parallelism keeps a whole copy of the model on each device: 140 GB of weights do not fit in 80 GB, however many devices there are.
2. `2 (n − 1) / n · B` bytes in `2 (n − 1)` steps.
3. A 2 KB transfer costs almost nothing in bandwidth; the time is the fixed cost per step, and the ring has 6 steps against the naive version's 2 hops (in and out of device 0).
4. Column-parallel: `wq`, `wk`, `wv`, `w_gate`, `w_up` (split by output: heads, MLP units). Row-parallel: `wo`, `w_down` (split by input: the heads' outputs, the MLP units). Attention is computed per head, and each device has whole heads with their KV heads, so it needs nothing from the others until `wo`.
5. Two: one after the attention block's `wo`, one after the MLP's `w_down`.
6. Each token must pass through every stage in order, and the next token cannot start until the previous one has come out of the last stage; while one stage works, the others have nothing to do.
7. It computes its share of the layers and also the final norm and the LM head, a 49,152 × 576 matrix (21% of SmolLM2-135M's weights).
8. It bounds each expert's work per batch at an even share, so the slowest device finishes sooner, at the cost of dropping 13.6% of (token, expert) pairs; those tokens lose that expert's contribution and continue through the residual connection with their other expert's output, or none.
9. It needs two all-reduces per layer per token, latency-bound for decoding; NVLink's latency and bandwidth inside a server keep that cost acceptable, slower links would make communication exceed computation.

## Further reading

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", 2019.
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", NeurIPS 2019.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", 2020.
- Fedus, Zoph and Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022.
- DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024 (expert parallelism and auxiliary-loss-free load balancing).
- Patarasuk and Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations", 2009 (the ring all-reduce).
- Liu, Zaharia and Abbeel, "Ring Attention with Blockwise Transformers for Near-Infinite Context", 2023.
- Next: [Chapter 29: GPUs](../29-gpu/README.md). How the devices themselves run inference, and CUDA kernels written in Rust.
