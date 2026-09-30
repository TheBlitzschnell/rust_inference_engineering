# Chapter 18: Quantization I: int8

> **In one sentence:** quantization stores each weight as a small integer plus a shared scale (`w ≈ scale × q`), which cuts the bytes a memory-bound decode step must read and lets the arithmetic run on integers; the craft is choosing how many values share a scale (outliers make that choice matter), measuring the quality you lose with perplexity and KL divergence instead of by reading outputs, and measuring the speed you gain instead of assuming it from the byte count.

**Where this fits:** chapter 14 showed that decode is limited by the bytes of weights read per token, and chapter 17 found decode running close to this machine's memory bandwidth. The remaining lever is fewer bytes. This chapter halves them with int8 and builds the evaluation tools (perplexity, KL divergence, top-1 agreement) that chapter 19 uses to go down to 4 bits.

**You need:** chapter 2 (number formats), chapter 6 (SIMD), chapter 14 (the `Matrix` trait), chapter 16 (the real model), chapter 17 (A/B comparisons).

**You will build:** 64-value int8 blocks with a scale, weight quantization per tensor, per row or per block, dynamic activation quantization, a W8A32 kernel (int8 weights, `f32` activations) and a W8A8 kernel (both int8, using AVX-512 VNNI integer dot products, with an AVX2 fallback), an int8 `Matrix` for the engine, and a quality evaluation against the `bf16` model.

---

## 1. The intuition

Imagine recording the heights of everything in a room with a ruler that has only 255 marks. If the ruler must cover the tallest object, say a 2.5 m bookshelf, the marks are 2 cm apart, and a 3 cm pencil is recorded as "2 marks, 4 cm". Now give each shelf of the bookcase its own ruler, sized to the tallest thing on that shelf. The pencil's shelf gets a ruler with marks a millimetre apart, and its height comes out almost exact.

That is quantization with a scale per block: 255 levels (−127 to 127) per block of 64 weights, each block's ruler sized to its largest value. The one bookshelf in the room (an **outlier** weight) no longer ruins everyone else's measurement.

**Where the analogy breaks:** measurement errors in a room stay put. In a model they travel: each layer's output, slightly wrong, is the next layer's input, through 30 layers. Whether the final prediction survives depends on how those errors add up, and the only way to know is to run the model and compare its predictions with the original's, position by position.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Quantization** | Storing numbers with fewer bits by mapping them to a small set of levels. |
| **Scale** | The step between levels: a value is reconstructed as `scale × q`. |
| **Symmetric quantization** | Levels spread evenly around zero (−127..127); no offset stored. |
| **Zero point** | The offset used by asymmetric quantization (levels `0..255` mapped to `[min, max]`). |
| **Granularity** | How many values share a scale: per tensor, per row (output channel), per block (group). |
| **Outlier** | A value much larger than its neighbours; it inflates the scale for everything sharing it. |
| **W8A32 (weight-only)** | Int8 weights, full-precision activations. Saves memory traffic; math stays in floating point. (On GPUs usually called W8A16.) |
| **W8A8** | Int8 weights *and* activations; products computed on integers. |
| **Dynamic quantization** | Choosing activation scales at run time, from the actual values. |
| **VNNI** | "Vector Neural Network Instructions": x86 instructions that multiply 8-bit integers and accumulate in 32 bits (`vpdpbusd`). |
| **Perplexity** | `exp(mean negative log-likelihood)` of real text: how surprised the model is, on average. |
| **KL divergence** | How far one probability distribution is from another; here, how far the quantized model's predictions are from the original's. |

## 3. The concepts in depth

### 3.1 The arithmetic

Symmetric int8 quantization of a group of values:

```text
scale = max |w| / 127
q     = round(w / scale), clamped to [-127, 127]      (stored: one signed byte)
ŵ     = scale × q                                      (what the kernel uses)
|w − ŵ| ≤ scale / 2
```

A worked example, with four values `[0.5, -1.2, 0.03, 2.54]`: the largest magnitude is 2.54, so `scale = 0.02`, and `q = [25, -60, 2, 127]` (0.03 / 0.02 = 1.5, rounded to 2). Back: `[0.5, -1.2, 0.04, 2.54]`. The largest value is exact; the smallest is off by 33%. **The error is fixed in absolute terms (at most half a step) and therefore large, relatively, for small values.** How big the step is depends entirely on the largest value sharing the scale.

The level −128 is left out on purpose. With −127..127 the grid is symmetric, so `−w` quantizes to `−q` exactly, and (section 3.3) the integer kernels need to negate values without overflowing.

### 3.2 Granularity and outliers

Part 1 of the demo quantizes every matrix of SmolLM2 three ways and measures the error, `‖W − Ŵ‖ / ‖W‖`:

```text
== 1. int8 weights: error by granularity (211 matrices)
   134479872 weights, RMS 0.1821, largest magnitude 9.25 (51 times the RMS)
   per tensor       relative error  3.596% overall, 11.699% in the worst matrix   (8 bits per weight)
   per row          relative error  0.903% overall,  1.206% in the worst matrix   (8 bits per weight)
   per block of 64  relative error  0.653% overall,  0.852% in the worst matrix   (8.5 bits per weight)
```

The largest weight is 51 times the typical one. With one scale per tensor, a matrix that contains such an outlier gets a step of 9.25 / 127 ≈ 0.073, a third of the typical weight's size: 11.7% error in the worst matrix. One scale per row confines each outlier to its row; one per block of 64 confines it to 64 values. Finer granularity costs storage (a 32-bit scale per 64 values is half a bit per weight) and a little arithmetic (applying a scale per block), and buys accuracy.

This chapter's block layout, `BlockQ8`, is 72 bytes for 64 weights: the scale, the 64 values, and their sum (used by the W8A8 kernel, section 3.3). That is 9 bits per weight: SmolLM2's weights shrink from 269 MB in `bf16` to 151 MB.

### 3.3 Two ways to use int8 weights

**W8A32: dequantize in registers.** The weights stay int8 in memory and are widened to `f32` as they are loaded: sign-extend 16 bytes to 16 32-bit integers, convert to `f32`, multiply-add with the activations, and apply the block's scale once per block. Memory traffic shrinks by 44%; the arithmetic is floating point, as before. This is how chapter 16's `bf16` weights already worked, with one more widening step.

**W8A8: integer products.** Quantize the activations too, at run time (one scale per block of 64, computed from the actual values), and compute `Σ q_w · q_x` on integers:

```text
Σ w·x  ≈  Σ_blocks  scale_w × scale_x × Σ_i q_w[i] · q_x[i]
                                         └──── int32 ────┘
```

AVX-512 VNNI's `vpdpbusd` multiplies 64 byte pairs and adds each group of four products into one of 16 32-bit lanes: 64 multiply-adds in one instruction, against 16 for an `f32` fused multiply-add. It has a quirk: one operand must be *unsigned* bytes. The kernel shifts each activation by 128 (flipping its top bit maps −127..127 to 1..255), and removes the extra term with the block's precomputed sum of weights:

```text
Σ (q_x + 128) · q_w  =  Σ q_x · q_w  +  128 · Σ q_w
```

AVX2 has no VNNI, but `vpmaddubsw` (also unsigned × signed, into 16 bits) works with the "sign trick": multiply `|q_x|` by `q_w` with `q_x`'s sign moved onto it. Pairs of products reach at most 2 × 127 × 127 = 32,258, which fits in 16 bits because no value is −128.

### 3.4 What it buys, measured

Part 2 measures the kernels on one thread, first on a matrix that fits in the cache, then streamed from memory:

```text
== 2. kernels, one thread (matrix-vector product)
   in cache: 512 x 1536           bf16      21.9 GB/s of weights,   10.9 G multiply-adds/s
   in cache: 512 x 1536           W8A32     12.2 GB/s of weights,   10.8 G multiply-adds/s
   in cache: 512 x 1536           W8A8      16.2 GB/s of weights,   14.4 G multiply-adds/s
   from memory: 256 MiB of bf16   bf16       8.4 GB/s of weights,    4.2 G multiply-adds/s
   from memory: 256 MiB of bf16   W8A32      7.7 GB/s of weights,    6.9 G multiply-adds/s
   from memory: 256 MiB of bf16   W8A8       8.9 GB/s of weights,    7.9 G multiply-adds/s
```

The in-cache numbers varied a lot between runs (W8A8 measured 14-40 G multiply-adds/s, `bf16` 11-18), but from memory, where decode lives, the picture was stable: per weight, int8 is 1.4-1.9 times faster than `bf16` on one core, because the core moves roughly the same bytes per second and int8 needs fewer bytes per weight.

Part 3 runs the whole model, with chapter 17's interleaved comparison:

```text
== 3. SmolLM2-135M, 4 threads
   weights read per token: bf16 269 MB, int8 blocks 151 MB
   decode, bf16 -> W8A32: 201.07ms -> 146.42ms per 16 tokens, speedup 1.43x (80% of pairs: 1.26-1.49x)
   decode, bf16 -> W8A8: 199.94ms -> 151.31ms per 16 tokens, speedup 1.36x (80% of pairs: 1.20-1.44x)
   256-token prefill: bf16 250, bf16 tiled (ch. 17) 358, W8A32 245, W8A8 239 tokens/s
```

Over six runs, int8 decoding was 1.27-1.43 times faster (W8A32) and 1.29-1.36 times (W8A8). Before a restart of this virtual machine (after which its kernel version and its speed both changed), two runs measured 1.88 and 2.00. The machine matters as much as the code.

What should it be? The bytes drop to 56%, so a purely bandwidth-bound step would be 1.78 times faster; the non-matrix work (about 1 ms per token, chapter 17) does not shrink, which lowers the ceiling to about 1.7. The measured 1.3-1.4 falls short of that, and chapter 17's per-matrix timer shows where: with four threads, the `bf16` matrices streamed at 23-24 GB/s and the int8 ones at 20 GB/s; on one thread, 7.8-8.3 against 6.3-7.0 GB/s. Int8 matrices are read more slowly, byte for byte, than `bf16` ones.

I did not find out why, and the chapter says so rather than guessing. Four plausible fixes were measured and changed nothing beyond noise: running the loop over rows inside the kernel (to remove a per-row call), four independent accumulators, a layout with the values 64-byte aligned and the scales in a separate array, and fusing the Q, K, V and gate, up products (chapter 17's idea, which a cost model suggested would matter more for int8; it did not: 0.99x and 1.05x). The next step would be hardware performance counters (cache misses per byte, memory requests in flight), which this virtual machine does not expose. On such a machine, `perf stat` or VTune would be the tool.

Prefill is compute-bound (chapter 14), and here int8 buys nothing: 220-280 tokens/s for the int8 models against 221-250 for `bf16` with the same one-dot-product-per-output structure, while chapter 17's tiled `bf16` kernel reaches 358-419. The VNNI instruction's four-fold advantage in multiply-adds per instruction only shows in a kernel that reuses loaded values, like chapter 17's tile. Without tiling, the int8 prefill is limited by the same loads as before.

### 3.5 What it costs, measured

Looking at a few generated answers tells you almost nothing about a quantized model: most answers are identical, and the damage hides in the probabilities. Part 4 runs the `bf16` model and each int8 variant on the same 2,044 predictions (four windows of 512 tokens of *Pride and Prejudice*, chapter 15's text) and compares them three ways:

- **Perplexity**: `exp` of the mean negative log-probability each model gives to the actual next token. For the `bf16` model it is 17.455: on this text, it is on average as uncertain as a fair choice among 17.5 tokens.
- **KL divergence** from the `bf16` model's distribution to the quantized model's, averaged over positions: `Σ p · (log p − log q)`. It is zero only when the two predict exactly the same probabilities, and it looks at every token, not only the one that came next.
- **Top-1 agreement**: how often both models' most likely next token is the same (what greedy decoding would pick).

```text
== 4. quality on 2044 tokens of Pride and Prejudice (bf16 perplexity 17.455)
   weights            perplexity    KL (nats)     same top-1
   W8A32 per tensor       18.424      0.05385          83.3%
   W8A32 per row          17.492      0.00309          96.6%
   W8A32 per block        17.430      0.00172          97.4%
   W8A8 per block         17.450      0.00609          95.4%
   W8A8 per token         17.877      0.03125          88.6%
```

What the table says:

- **Per-tensor scales hurt**: perplexity up 5.6%, and one prediction in six changes. The outliers of section 3.2 at work.
- **Per-row and per-block weights are close to lossless**: KL of 0.002-0.003 nats, 97% of top choices unchanged.
- **The per-block model's perplexity is *lower* than the original's.** That is not an improvement. The change is 0.14%, and quantization noise can nudge the probability of the actual next token either way; the KL divergence (0.0017, not 0) shows the model did change. Perplexity alone cannot resolve differences this small, which is why KL is reported next to it.
- **Quantizing activations costs more than quantizing weights** (KL 0.006 against 0.0017), and **one scale per token costs much more** (0.031). Activations have outliers too, far larger than the weights': a few channels carry values many times the rest, and with one scale per token they flatten every other channel. Per-block activation scales contain them, like per-block weight scales.

And both int8 models still answer "The capital of France is Paris." token for token (the test in [`tests/reference.rs`](tests/reference.rs) checks it).

## 4. The code

The block and quantization are in [`src/quant.rs`](src/quant.rs), the kernels in [`src/kernels.rs`](src/kernels.rs), the parallel product for block formats in [`src/matmul.rs`](src/matmul.rs), the `Matrix` in [`src/lib.rs`](src/lib.rs), the evaluation in [`src/eval.rs`](src/eval.rs), the demo in [`src/main.rs`](src/main.rs).

### 4.1 A block

<!-- file: src/quant.rs -->
```rust
    pub fn with_scale(values: &[f32], scale: f32) -> Self {
        assert_eq!(values.len(), BLOCK, "a block has 64 values");
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut q = [0i8; BLOCK];
        for (qi, &v) in q.iter_mut().zip(values) {
            *qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        Self {
            scale,
            sum: q.iter().map(|&x| i32::from(x)).sum(),
            q,
        }
    }
```

`BlockQ8` is `#[repr(C)] { scale: f32, sum: i32, q: [i8; 64] }`: 72 bytes, the fields in exactly that order, so a row of blocks is one contiguous run of memory the kernel streams through. The scale is passed in, so the same function quantizes with a per-tensor, per-row or per-block scale: `quantize` computes the right one and calls it per block. A block of zeros gets scale 0 and stays zero, instead of dividing by zero.

### 4.2 W8A32

<!-- file: src/kernels.rs -->
```rust
        for (b, xs) in w.iter().zip(x.chunks_exact(BLOCK)) {
            let mut acc = _mm512_setzero_ps();
            for part in 0..4 {
                // SAFETY: 16 bytes at 16 * part are inside the 64-byte `q`,
                // and 16 floats at 16 * part inside the 64-float `xs`.
                let (bytes, xv) = unsafe {
                    (
                        _mm_loadu_si128(b.q.as_ptr().add(16 * part).cast::<__m128i>()),
                        _mm512_loadu_ps(xs.as_ptr().add(16 * part)),
                    )
                };
                let wf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes));
                acc = _mm512_fmadd_ps(wf, xv, acc);
            }
            total = _mm512_fmadd_ps(acc, _mm512_set1_ps(b.scale), total);
        }
```

Per 16 weights: sign-extend 16 bytes to 16 integers, convert to `f32`, multiply-add. The block's scale multiplies the block's partial sum once, instead of multiplying every weight.

### 4.3 W8A8 with VNNI

<!-- file: src/kernels.rs -->
```rust
            let unsigned_x = _mm512_xor_si512(vx, flip);
            let ints = _mm512_dpbusd_epi32(_mm512_setzero_si512(), unsigned_x, vw);
            let scale = bw.scale * bx.scale;
            total = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ints), _mm512_set1_ps(scale), total);
            correction += 128.0 * bw.sum as f32 * scale;
```

One 64-byte load per operand, one XOR to make the activations unsigned, one `vpdpbusd` for all 64 products, then the 16 integer lane sums are converted to `f32` and scaled by both blocks' scales. The correction for the +128 shift is accumulated as a plain number and subtracted once at the end. The 32-bit lanes cannot overflow: each holds at most 4 × 255 × 127 = 129,540, and `f32` represents every integer below 2²⁴ exactly.

`Q8Matrix::matmul` quantizes the activation rows first (one small allocation per call, a few kilobytes against the matrix's hundreds), then runs `matmul_blocks`, chapter 14's parallel product rewritten so that row lengths come from the slice lengths and the activations can be any type: `f32` values for W8A32, blocks for W8A8.

### 4.4 Comparing two models

<!-- file: src/eval.rs -->
```rust
        for (t, (r, c)) in rows.enumerate().take(tokens.len() - 1) {
            let next = tokens[t + 1] as usize;
            self.reference_nll -= f64::from(r[next]);
            self.candidate_nll -= f64::from(c[next]);
            // KL(p ‖ q) = Σ p · (log p − log q), with p the reference.
            self.kl += r
                .iter()
                .zip(c)
                .map(|(&lp, &lq)| f64::from(lp.exp()) * f64::from(lp - lq))
                .sum::<f64>();
            if argmax(r) == argmax(c) {
                self.same_top1 += 1;
            }
            self.tokens += 1;
        }
```

`log_probs` runs a window through the model with chapter 14's `forward_all` (logits for every position) and applies chapter 8's `log_softmax` to each row. Row `t` predicts token `t + 1`, so a window of 512 tokens gives 511 predictions. Sums are accumulated in `f64`: 2,044 predictions × 49,152 terms each is 100 million additions, too many to trust to `f32`.

## 5. Run it

```bash
cargo test -p ch18-int8
cargo run --release -p ch18-int8                  # all four parts, about 2 minutes
cargo run --release -p ch18-int8 -- quality       # or: weights, kernels, speed
```

The output is shown in section 3. The quality numbers are deterministic; the speed numbers move between runs and between machines (section 3.4).

## 6. The Rust behind it

**`#[repr(C)]`** fixes a struct's field order and padding to C's rules. Without it, Rust may reorder fields. For a block that kernels read with pointer arithmetic, and that you may one day write to a file, the layout must be pinned. `size_of::<BlockQ8>()` is 72.

**Float-to-integer casts saturate.** `x as i8` for an `f32` clamps to −128..127 and maps NaN to 0 (since Rust 1.45; before that it was undefined behaviour). The code clamps to −127..127 itself anyway, because −128 must never appear (section 3.1).

**Lossless conversions with `From`.** `i32::from(q)` and `f64::from(x)` compile only when no value can be lost; `as` compiles for any pair of number types and silently truncates. Using `From` wherever it applies means every remaining `as` is a place where truncation was considered.

**Several target features at once.** `#[target_feature(enable = "avx512f,avx512vnni")]` compiles one function for a CPU with both. The runtime check must test all of them (`Kernels::is_available`), and there is a separate AVX2 kernel and a portable one, selected once and cached in a `OnceLock`.

**One generic product for every format.** `matmul_blocks<B, X, D>` is generic over the weight block type, the activation type and the kernel. Chapter 19's 4-bit blocks reuse it unchanged.

## 7. Mistakes you will make

- **One scale per tensor.** One outlier ruins the whole matrix (section 3.2).
- **Allowing −128.** The grid becomes asymmetric, and negating −128 in 8 bits overflows, which breaks the sign trick.
- **Forgetting the +128 correction** in the VNNI kernel. Every dot product gains `128 · Σ w`, and the model produces fluent nonsense.
- **Judging quantization by reading a few outputs.** Measure KL divergence and top-1 agreement against the original on real text; perplexity alone is too coarse for small differences.
- **Quantizing activations per token.** Activation outliers are worse than weight outliers (KL 0.031 against 0.006 here).
- **Predicting the speedup from the byte count.** Measure it; here the bytes promised 1.78x and the machine delivered 1.3-1.4x.
- **Accumulating integer products in 16 bits**, or summing log-probabilities in `f32`.

## 8. How the professionals do it

- **llama.cpp's Q8_0** stores 32 values with one `f16` scale (34 bytes, 8.5 bits per weight) and computes int8 dot products against activations quantized to the same blocks, like this chapter's W8A8. Its k-quants (chapter 19) go lower.
- **LLM.int8()** (bitsandbytes) found that large models have a few activation channels with huge values, and computes those channels in 16-bit while quantizing the rest. **SmoothQuant** moves the difficulty from activations to weights by rescaling channels (mathematically equivalent, easier to quantize), enabling per-tensor W8A8.
- **GPUs:** H100-class GPUs run FP8 (chapter 2's e4m3) matrix products at twice the `bf16` rate, usually with per-tensor or per-channel scales; vLLM and TensorRT-LLM ship FP8 and INT8 W8A8 kernels. Weight-only int8 there is called W8A16, since activations stay in 16 bits.
- **Intel CPUs** add AMX, tile instructions that multiply whole 16 × 64 int8 or bf16 matrices at once; oneDNN and PyTorch use them. They are not yet in stable Rust.
- **Evaluation** in practice: perplexity on standard corpora (WikiText-2, C4) plus task benchmarks, and for small changes KL divergence against the unquantized model, which llama.cpp's `llama-perplexity` tool reports.

## 9. Exercises

1. **By hand.** Quantize `[0.5, -1.2, 0.03, 2.54]` with one absmax scale, then with two scales (one for the first two values, one for the last two). Which value's error improves?
2. **Bits per weight.** What do this chapter's blocks cost in bits per weight? What would dropping `sum`, and storing the scale as `f16`, give? Compare with llama.cpp's Q8_0.
3. **Negative 128.** Suppose weights could be −128. Find values of `q_w` and `q_x` for which the AVX2 sign trick gives a wrong product.
4. **KL or perplexity.** Why can quantization lower the perplexity on some text, and why can it never lower the KL divergence below zero?
5. **Your machine.** Run `cargo run --release -p ch18-int8 -- speed` three times. How does your int8 decode speedup compare with the 1.78x the bytes predict?

## 10. Check yourself

1. What does the scale of a symmetric int8 block equal, and what bounds the rounding error?
2. Why do outliers make per-tensor quantization bad, and what fixes it?
3. What is the difference between W8A32 and W8A8, and which one uses integer instructions?
4. Why does the VNNI kernel add 128 to activations, and how is that undone?
5. Why doesn't int8 speed up prefill in this chapter?
6. Why report KL divergence and top-1 agreement next to perplexity?

## 11. Recap

- `w ≈ scale × q`, `q` in −127..127, `scale = max|w| / 127`, error at most half a step, large for small values relative to their size.
- SmolLM2's largest weight is 51 times the typical one. Per-tensor scales: 3.6% weight error and 5.6% higher perplexity; per-row or per-64-block scales: under 1% error, KL 0.002-0.003, near lossless.
- W8A32 widens int8 weights to `f32` in registers; W8A8 quantizes activations per block and uses integer dot products (VNNI: 64 multiply-adds per instruction, with a +128 correction).
- Decode: 151 MB of weights instead of 269, 1.3-1.4x faster on this machine (1.9-2.0x before the VM's restart); int8 matrices stream more slowly per byte, for a reason this VM's missing counters could not reveal.
- Prefill: no gain without a tiled int8 kernel.
- Measure quality with KL divergence and top-1 agreement, not only perplexity, and never by reading samples.

## Answers

**Exercises**

1. One scale (0.02): `[0.5, -1.2, 0.04, 2.54]`, only 0.03 is off (by 0.01). Two scales: the first pair's scale is 1.2 / 127 ≈ 0.00945, giving `q = [53, -127]` and back `[0.501, -1.2]`; the second pair's scale is 0.02 as before, so 0.03 is still 0.04. The first value got slightly worse, not better: the error depends on which values share a scale with the small ones. To save 0.03, it would need a block without 2.54.
2. 72 bytes per 64 weights = 9 bits. Without `sum`: 68 bytes, 8.5 bits. With an `f16` scale as well: 66 bytes, 8.25 bits. Q8_0 is 34 bytes per 32 weights = 8.5 bits: its blocks are half as large, which halves the values an outlier can spoil but doubles the scale overhead.
3. `vpsignb` negates `q_w` when `q_x` is negative; negating −128 in 8 bits gives −128 again. With `q_w = -128` and `q_x = -1`, the trick computes `|−1| × (−128) = −128` instead of the correct +128.
4. Perplexity only looks at the probability of the actual next token; quantization noise can raise it at some positions and lower it at others, and on a finite text the balance can come out slightly in the quantized model's favour. KL divergence compares the whole distribution with the reference's and is zero only when they are identical: any change raises it.
5. On the reference machine: 1.27-1.43x (W8A32) and 1.29-1.36x (W8A8) in six runs, and 1.88-2.00x before the VM was restarted. If yours is well below 1.78x, measure the per-matrix GB/s with chapter 17's `instrument` to see whether the int8 matrices stream slower than the `bf16` ones, as they did here.

**Check yourself**

1. The largest magnitude in the block divided by 127. The rounding error of each value is at most half a step, `scale / 2`.
2. One large value sets the step for every value sharing its scale, so small values are rounded coarsely. Finer granularity (per row, per block) limits each outlier's reach.
3. W8A32 keeps activations in `f32` and widens the int8 weights to `f32` before multiplying; W8A8 also quantizes the activations and multiplies integers (VNNI's `vpdpbusd`, or `vpmaddubsw` on AVX2).
4. `vpdpbusd` needs one unsigned operand, so each activation is shifted into 1..255 (by flipping its top bit). The shift adds `128 × Σ q_w` to each block's sum, which the kernel subtracts using the precomputed `sum` of the block's weights.
5. Prefill is compute-bound, and this chapter's int8 kernels compute one dot product per output, loading each weight row once per token, just like the untiled `bf16` kernel; the loads, not the multiplies, limit them. Chapter 17's tile kernel is what made `bf16` prefill faster.
6. Perplexity measures only the probability of the observed tokens, and can move either way by chance for small changes. KL divergence compares whole distributions against the reference, and top-1 agreement shows what greedy decoding would do differently.

## Further reading

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2018.
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022.
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2023.
- Intel, "Intel 64 and IA-32 Architectures Software Developer's Manual": `VPDPBUSD`, `VPMADDUBSW`.
- Next: [Chapter 19: Quantization II: 4-bit](../19-4bit/README.md). Half the bytes again, and a real quality cost.
