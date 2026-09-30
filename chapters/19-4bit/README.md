# Chapter 19: Quantization II: 4-bit

> **In one sentence:** at 4 bits a weight has only 16 possible values, so quantization stops being nearly free: how you choose each group's range decides whether the model degrades a little or falls apart, and the choice that works best is the one that minimizes the error *where the activations are large*, measured on sample text.

**Where this fits:** chapter 18 showed int8 is close to lossless. This chapter halves the bytes again, measures the real quality cost of 4 bits on SmolLM2, and builds the calibration step (recording activation statistics on sample text) that production 4-bit methods rely on.

**You need:** chapter 18 (blocks, kernels, the evaluation).

**You will build:** 4-bit blocks with two values per byte, three ways of choosing a block's range (symmetric, min-max, and searched versions of both), a `Recorder` that measures activation importance while the model runs sample text, importance-weighted quantization, a W4A8 kernel using AVX-512 VNNI, and a comparison of every variant's size, quality and speed.

---

## 1. The intuition

Back to chapter 18's rulers. At 8 bits each ruler had 255 marks; at 4 bits it has 16. A ruler with 16 marks cannot measure everything well, so you must decide what to measure well.

Suppose the thing you care about is not the heights themselves, but a weighted sum of them, where some heights are multiplied by large numbers and some by small ones. Then an error on a heavily weighted height hurts a lot and an error on a lightly weighted one barely matters. The sensible ruler places its marks to be accurate on the heavily weighted values, even if that makes it worse on the others.

That is importance-weighted quantization. A weight's error is multiplied by the activation it meets, so the error that matters is `(w − ŵ) × x`. Measure how large each input channel's activations typically are, and choose each block's range to minimize the error weighted by that.

**Where the analogy breaks:** the multipliers (activations) change with every token. The importance is an average over sample text, and a model quantized with it is tuned for text that looks like the sample. That is why the evaluation below runs on different text than the calibration.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Nibble** | Half a byte, 4 bits: one 4-bit code. Two per byte. |
| **Code** | The stored 4-bit integer, 0..15. The weight is `scale × code + min`. |
| **Symmetric** | Range centred on zero: codes stand for −7..7 times the scale. |
| **Min-max (asymmetric)** | The 16 levels span the block's own minimum to maximum. |
| **Clipping** | Choosing a range slightly smaller than the extremes: the extremes are rounded to the edge, everything else gets finer steps. |
| **Calibration** | Running the full-precision model on sample text to measure statistics used by quantization. |
| **Importance** | Here, the mean square of the activations entering each input column of a matrix. |
| **GPTQ / AWQ** | Widely used 4-bit methods that use calibration data (Frantar et al.; Lin et al.). |

## 3. The concepts in depth

### 3.1 Packing two weights per byte

A block holds 64 weights: an `f32` scale, an `f32` offset, and 32 bytes of codes (40 bytes, 5 bits per weight). Byte `j` stores weight `j` in its low 4 bits and weight `j + 32` in its high 4 bits:

```text
byte j:   [ code(j + 32) | code(j) ]
           high nibble     low nibble
```

This layout makes unpacking two vector instructions: `bytes & 0x0F` gives weights 0..32 in order, `(bytes >> 4) & 0x0F` gives weights 32..64. Storing neighbours in the same byte would need a shuffle to restore the order.

The whole model shrinks to 84 MB, from 269 MB in `bf16` and 151 MB in chapter 18's int8 blocks.

### 3.2 Sixteen levels are not enough everywhere

Part 1 quantizes all of SmolLM2's weights with each scheme and measures the relative error, as chapter 18 did for int8 (0.65% with blocks of 64):

```text
== 1. 4-bit weights: error and size (134479872 weights)
   stored as 40 bytes per 64 weights = 5.0 bits per weight, 84 MB (bf16: 269 MB, int8 blocks: 151 MB)
   symmetric, per row             relative error  16.34%
   symmetric, 192                 relative error  14.14%
   symmetric, 64                  relative error  11.84%
   symmetric searched, 64         relative error  10.39%
   min-max, 64                    relative error   9.51%
   min-max searched, 64           relative error   8.72%
   symmetric searched, 64, imp.   relative error  10.88% (without importance 10.39%); weighted by importance 6.29% (without 9.04%)
   min-max searched, 64, imp.     relative error   9.16% (without importance 8.72%); weighted by importance 3.85% (without 6.58%)
```

Errors are 13-25 times larger than int8's, and three choices shrink them:

- **Smaller groups** (per row → 192 → 64 values per scale): outliers affect fewer values.
- **Min-max instead of symmetric**: a block's weights are rarely centred exactly on zero. A symmetric range wastes levels on values that do not occur (and uses only 15 of the 16 codes, so that zero is exact); min-max spends all 16 on the block's actual span.
- **Searching the range**: trying slightly narrower ranges (clipping the extremes by up to 30% for symmetric, up to 10% at each end for min-max) and keeping the one with the smallest squared error. Clipping a few extreme values makes every step finer for all the rest.

The last two rows are importance-weighted (section 3.4): they make the plain error slightly *worse* (8.72% → 9.16%) and the error weighted by activation size much better (6.58% → 3.85%).

### 3.3 What 4 bits cost

Part 2 evaluates every variant with chapter 18's method: 2,044 predictions on the first 2,048 tokens of *Pride and Prejudice*, against the `bf16` model:

```text
== 2. quality on 2044 tokens of Pride and Prejudice (bf16 perplexity 17.455)
   weights                                perplexity   KL (nats)   same top-1
   W8A8 per block (chapter 18)                17.450      0.0061        95.4%
   W4A8 symmetric, per row                    67.435      1.4019        36.8%
   W4A8 symmetric, 192                        35.963      0.7677        50.0%
   W4A8 symmetric, 64                         27.272      0.4941        60.5%
   W4A8 symmetric searched, 64                27.432      0.4980        59.9%
   W4A8 min-max, 64                           25.579      0.3725        61.5%
   W4A8 min-max searched, 64                  23.340      0.3278        64.2%
   W4A8 symmetric searched, 64, imp.          23.021      0.3030        67.7%
   W4A8 min-max searched, 64, imp.            19.546      0.1317        79.0%
   W4A32 min-max searched, 64                 23.234      0.3232        64.8%
```

Read it from the top:

- **4 bits are not free on this model.** Even the best variant raises perplexity by 12% (17.5 → 19.5); int8 raised it by nothing measurable. Small models are known to suffer more from low-bit quantization than large ones: with fewer parameters, each carries more of the model's knowledge. Treat SmolLM2-135M's numbers as a hard case, not a typical one.
- **One scale per row breaks the model**: perplexity 67, and in part 4 it answers in a loop.
- **Reducing weight error does not always help.** The symmetric search lowered weight error from 11.84% to 10.39% and changed quality not at all (KL 0.494 → 0.498). Squared weight error is the wrong objective: an error on a weight that meets tiny activations costs nothing.
- **Weighting the error by importance helps a lot.** Min-max searched: KL 0.328 without importance, 0.132 with it; top-1 agreement 64% → 79%; perplexity 23.3 → 19.5. The same weights, the same format, the same speed; only the choice of each block's range changed, using 2,048 tokens of *different* text (tokens 4,096-6,144 of the novel) to measure the importance.
- **Activation quantization is not the problem here.** W4A32 (activations in `f32`) is no better than W4A8 (KL 0.323 against 0.328): 4-bit weight error dwarfs 8-bit activation error.

### 3.4 Calibration and importance

For one output of a linear layer, the error caused by quantizing its weights is

```text
Σ_j (w_j − ŵ_j) · x_j
```

The weight errors are multiplied by the activations. Their typical size, measured over many tokens, is roughly proportional to `Σ_j (w_j − ŵ_j)² · E[x_j²]`: each squared weight error weighted by the mean square of the activation in its input column. So:

1. **Calibrate**: wrap every matrix in a `Recorder` and run the full-precision model on sample text. Each recorder adds up `x_j²` for every column `j` of every input it multiplies.
2. **Quantize with weights**: when searching each block's range, minimize `Σ_j importance_j · (w_j − ŵ_j)²` instead of the plain squared error.

In SmolLM2, as in most transformers, a few input channels carry activations far larger than the rest (the same activation outliers that made per-token int8 activations bad in chapter 18). The weights in those columns matter most, and the weighted search spends the 16 levels to get them right.

This is the idea behind llama.cpp's "importance matrix" quantization; AWQ ("activation-aware weight quantization") uses the same statistics to rescale channels before quantizing, and GPTQ goes further, adjusting the remaining weights of a row to compensate for the error of each one it quantizes. All three need calibration data; none needs retraining.

### 3.5 Speed

The W4A8 kernel mirrors chapter 18's VNNI kernel. The 4-bit codes are unsigned, which is exactly the operand `vpdpbusd` wants unsigned, so there is no +128 trick: unpack the 32 bytes into 64 codes, multiply by the 64 int8 activations, and add the offset term using the activation block's precomputed sum:

```text
Σ w·x = Σ (scale · c + min) · x = scale · Σ c·x + min · Σ x
```

Part 3 measures decode with chapter 17's interleaved comparison:

```text
== 3. decode speed, 4 threads
   weights read per token: bf16 269 MB, int8 151 MB, 4-bit 84 MB
   bf16 -> 4-bit: 214.40ms -> 119.59ms per 16 tokens, speedup 1.71x (80% of pairs: 1.57-1.95x)
   int8 -> 4-bit: 157.93ms -> 125.16ms per 16 tokens, speedup 1.27x (80% of pairs: 1.11-1.36x)
```

Over two runs: 1.68-1.71x faster than `bf16`, 1.27-1.31x faster than int8. The bytes shrink by 3.2x and 1.8x respectively; as in chapter 18, the speedup is well below the byte ratio, because the rest of a decode step does not shrink and, on this machine, smaller formats are read at fewer bytes per second.

### 3.6 What the answers look like

Part 4 asks each model the same question, greedily:

```text
== 4. greedy answers to "Explain in two sentences why the sky is blue."
   bf16:
     The sky appears blue because the Earth's atmosphere scatters sunlight in all directions, including blue light, which is scattered more than other colors by large molecules like water droplets in clouds. This scattering effect is known as Rayleigh scattering. As a result, blue light is scattered in all directions, while other
   int8 (W8A8 per block):
     The sky appears blue because the Earth's atmosphere scatters sunlight in all directions, but primarily in the blue and violet parts of the spectrum. This scattering occurs because the atmosphere is composed of molecules with different wavelengths of light, such as blue and violet, which scatter more light than other wavelengths, like red
   4-bit symmetric, per row:
     The sky is blue because the Earth and the moon, and the Earth and the moon, and the Earth and the moon, and the Earth and the moon and the Earth and the moon and the Earth and the moon and the Earth and the Earth and the Earth and the Earth and the Earth and the
   4-bit min-max searched, 64:
     The sky is blue because the sun's rays, primarily blue, are scattered by the Earth's atmosphere, scattering the shorter wavelengths, such as blue, more efficiently than the longer wavelengths, such as red, orange, and yellow, which are scattered more efficiently than green, a color that is not present
   4-bit min-max searched, 64, imp.:
     The sky is blue because the sun's rays bounce off the surface of the Earth and reflect off the water and clouds in the sky. The blue color is due to the scattering of light by the water molecules in the atmosphere, which scatter shorter wavelengths of light, such as blue and violet, more than
```

Every quantized model answers differently from the original after a few words, even int8, whose KL divergence is tiny: greedy decoding amplifies any change in the first token where two choices were close. All but the per-row model answer fluently, and all of them (the original included) get some physics wrong. You could not rank these models by reading their answers; the table in section 3.3 ranks them clearly.

## 4. The code

The block and quantization are in [`src/quant.rs`](src/quant.rs), the kernels in [`src/kernels.rs`](src/kernels.rs), `Q4Matrix` and `Recorder` in [`src/lib.rs`](src/lib.rs), the measurements in [`src/main.rs`](src/main.rs).

### 4.1 Packing and unpacking

<!-- file: src/quant.rs -->
```rust
    pub fn pack(codes: &[u8; BLOCK], scale: f32, min: f32) -> Self {
        let mut packed = [0u8; BLOCK / 2];
        for (j, p) in packed.iter_mut().enumerate() {
            *p = (codes[j] & 0x0F) | (codes[j + 32] << 4);
        }
        Self { scale, min, packed }
    }
```

`BlockQ4` is `#[repr(C)] { scale: f32, min: f32, packed: [u8; 32] }`. Every scheme, symmetric included, stores a `min`: for symmetric blocks it is `−8 × scale`, so code 8 is exactly zero. One formula, `scale × code + min`, then serves all schemes, and the kernels need no special cases.

### 4.2 Searching a range

<!-- file: src/quant.rs -->
```rust
        Scheme::MinMaxSearch => {
            let range = hi - lo;
            let mut candidates = Vec::with_capacity(36);
            for a in 0..=5u8 {
                for b in 0..=5u8 {
                    let (l, h) = (
                        lo + 0.02 * f32::from(a) * range,
                        hi - 0.02 * f32::from(b) * range,
                    );
                    candidates.push(((h - l) / 15.0, l));
                }
            }
            best(candidates)
        }
```

36 candidate ranges, each end moved inwards by 0-10% independently, and `best` keeps the one with the smallest error as measured by `weighted_error`:

<!-- file: src/quant.rs -->
```rust
fn weighted_error(values: &[f32], importance: Option<&[f32]>, (scale, min): (f32, f32)) -> f32 {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let back = scale * f32::from(code(v, scale, min)) + min;
            importance.map_or(1.0, |imp| imp[i]) * (v - back) * (v - back)
        })
        .sum()
}
```

Without importance every value counts 1, the plain squared error. This search is brute force: 36 candidates × 64 values per block, about 5 billion evaluations for the whole model. Real quantizers use smarter searches, but the objective is the part that matters.

### 4.3 Recording importance

<!-- file: src/lib.rs -->
```rust
            let (sums, rows) = &mut *self.stats.lock().expect("stats lock");
            for row in x.chunks_exact(self.inner.cols()) {
                for (s, &v) in sums.iter_mut().zip(row) {
                    *s += f64::from(v) * f64::from(v);
                }
            }
            *rows += m as u64;
        }
        self.inner.matmul(pool, x, y, m, scratch);
```

`Recorder<W>` is a `Matrix` wrapper, like chapter 17's timer: it adds each input row's squares to a per-column sum, then multiplies as usual. The statistics sit behind a `Mutex`, because `matmul` takes `&self` (a vector of sums cannot be updated with atomics the way chapter 17's counters were). Calibration runs the wrapped model with `forward_all`, so that the LM head, which in normal prefill only sees the last token, records all 2,048 positions too.

### 4.4 The W4A8 kernel

<!-- file: src/kernels.rs -->
```rust
            let codes = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            let ints = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, xq);
            let scale = bw.scale * bx.scale;
            total = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ints), _mm512_set1_ps(scale), total);
            offsets += bw.min * bx.scale * bx.sum as f32;
```

`unpack` (AVX2) turns the 32 packed bytes into two vectors of 32 codes; `_mm512_inserti64x4` joins them into the 64 codes of the block, in order, matching the 64 activations. The offset term uses `bx.sum`, which chapter 18's activation blocks already carry.

## 5. Run it

```bash
cargo test -p ch19-4bit
cargo run --release -p ch19-4bit                  # all parts: over 15 minutes (many models to build and evaluate)
cargo run --release -p ch19-4bit -- quality       # or: weights, speed, answers
```

The weights, quality and answers parts are deterministic; the speed part varies between runs.

## 6. The Rust behind it

**Bit manipulation on `u8`.** `(a & 0x0F) | (b << 4)` packs, `byte & 0x0F` and `byte >> 4` unpack. In Rust, `<<` on a `u8` that shifts bits out simply loses them (in debug builds only an out-of-range shift *amount* panics), and `>>` on unsigned types shifts in zeros, so no masking is needed after `>> 4`.

**`Option<&[f32]>` for optional data.** `quantize` and `quantize_weighted` share one implementation that takes `Option<&[f32]>`; `importance.map_or(1.0, |imp| imp[i])` reads "the importance if there is one, else 1". No allocation of a vector of ones for the unweighted case.

**`Mutex` inside a `Sync` type.** `Recorder<W>` is shared by the pool's threads through `&self`. `Mutex<(Vec<f64>, u64)>` makes it `Sync` and lets `matmul` update the sums; the engine calls one matrix at a time, so the lock is never contended.

**Closures that capture an iterator.** `q4_weighted` in the demo passes `map_matrices` a closure that pulls the next importance vector from `imp.iter()` each time it is called. It relies on `map_matrices` visiting the matrices in the same order as `calibrate` recorded them, which the code documents in both places.

## 7. Mistakes you will make

- **One scale per row at 4 bits.** Fine at 8 bits (chapter 18), catastrophic at 4 (perplexity 67).
- **Optimizing the wrong error.** Minimizing squared weight error left quality unchanged; minimizing activation-weighted error improved it a lot.
- **Calibrating and evaluating on the same text.** The quality then flatters the method. Keep them apart, as the demo does.
- **Judging a 4-bit model by a few answers.** Every quantized model here gave a different, fluent answer.
- **Getting the nibble order wrong** between the packer and the kernel: the model still runs, and its output is garbage. The kernels are tested against the portable dequantizing version for every scheme.
- **Expecting the byte ratio as speedup.** 3.2x fewer bytes than `bf16` gave 1.7x.

## 8. How the professionals do it

- **llama.cpp's k-quants** (`Q4_K`, `Q5_K`, `Q6_K`...) use "super-blocks" of 256 weights split into sub-blocks of 32, with the sub-blocks' scales and minimums themselves quantized to 6 bits, reaching about 4.5 bits per weight for Q4_K. Their scales are chosen by an error-minimizing search, optionally weighted by an importance matrix from `llama-imatrix`, the method of this chapter.
- **GPTQ** quantizes a matrix one column at a time and updates the not-yet-quantized columns to absorb each column's error, using second-order information from calibration activations. **AWQ** scales important input channels up (and the matching activations down) before quantizing, so their weights get finer steps. Both are standard ways to publish 4-bit models (for example, the many `-GPTQ` and `-AWQ` checkpoints on Hugging Face).
- **GPU kernels** for 4-bit weights (Marlin, Machete, and those in TensorRT-LLM and vLLM) unpack nibbles in registers and feed tensor cores in 16-bit or 8-bit, reaching close to the memory bandwidth limit for batch-1 decode.
- **Hardware formats:** NVIDIA's Blackwell GPUs and the OCP "microscaling" (MX) formats support 4-bit floating point (FP4, e2m1) with a shared scale per block of 16 or 32 values in hardware: the block-scaling idea of this chapter, built into the chip.

## 9. Exercises

1. **Bits per weight.** This chapter's block is 40 bytes per 64 weights. How many bits per weight would the block cost with `f16` scale and offset? With one scale and offset per 256 weights, as in llama.cpp's super-blocks?
2. **Why min-max wins.** A block's weights range from −0.02 to 0.10. Compute the step size for the symmetric scheme and for min-max. How many of the 16 codes does the symmetric scheme leave unused?
3. **Importance from other text.** The demo calibrates on the same novel it evaluates on (different passages). Why could calibration on very different text (code, say) help less?
4. **Bigger groups, stored once.** This chapter's format repeats the scale and offset in every block even when 192 values share them. How many bits per weight would a format that stores them once per group of 192 need? Is the quality of "symmetric, 192" worth that saving?

## 10. Check yourself

1. How are two 4-bit codes stored in a byte here, and why that order?
2. Why does min-max use the 16 levels better than symmetric?
3. Why can clipping the largest values reduce the total error?
4. What does the `Recorder` measure, and why does it matter for the error?
5. Why did reducing the plain weight error not improve the model, while reducing the importance-weighted error did?
6. Why doesn't the W4A8 VNNI kernel need chapter 18's +128 correction?

## 11. Recap

- 4 bits = 16 levels; a block of 64 weights costs 40 bytes here (5 bits per weight), SmolLM2 shrinks to 84 MB.
- Group size, range choice and search matter far more than at 8 bits: per-row symmetric gives perplexity 67, min-max per 64 gives 25.6, with a range search 23.3.
- The right objective is error weighted by activation size. Calibrating on 2,048 tokens and weighting the search by per-column importance brought perplexity to 19.5 (bf16: 17.5), KL from 0.33 to 0.13, top-1 agreement from 64% to 79%, at no cost in size or speed.
- On this small model, 4 bits still cost real quality; larger models usually lose less.
- Decode: 1.7x faster than `bf16`, 1.3x faster than int8.

## Answers

**Exercises**

1. With `f16` scale and offset: 4 + 32 = 36 bytes per 64 weights, 4.5 bits. With one `f32` scale and offset per 256 weights: 128 bytes of codes plus 8 bytes, 136 bytes per 256 weights, 4.25 bits (with `f16`: 4.125). llama.cpp's Q4_K lands at 4.5 bits because it adds small quantized scales per 32-weight sub-block, trading a little size for accuracy.
2. Symmetric: `scale = 0.10 / 7 ≈ 0.0143`, levels from −0.114 to 0.100; the block's values (−0.02 to 0.10) use only the codes for −1 to 7, so codes for −7 to −2 (six codes) are wasted, plus the one code (−8) the scheme never uses: seven of 16. Min-max: `scale = 0.12 / 15 = 0.008`, all 16 codes inside the block's range, a step 1.8 times finer.
3. The importance measures which input channels carry large activations. Different kinds of text activate different channels to different degrees; importance measured on prose describes prose. Channels that matter for code but are quiet in prose get little weight, and their weights are quantized less carefully. Calibration text should resemble the traffic the model will serve.
4. Three blocks' codes (96 bytes) plus one `f32` scale and offset (8 bytes): 104 bytes per 192 weights, 4.33 bits instead of 5. Not worth it here: "symmetric, 192" has a perplexity of 36.0 against 27.3 for 64-value groups and 19.5 for the best 64-value variant. Spend bits on smaller groups and a better range before saving them on scales.

**Check yourself**

1. Byte `j` holds code `j` in its low nibble and code `j + 32` in its high nibble, so masking gives codes 0..32 in order and shifting gives codes 32..64 in order, without any shuffling.
2. Weights in a block are rarely centred on zero. Symmetric levels cover `[−max|w|, max|w|]`, part of which contains no weights; min-max levels cover exactly `[min, max]`, so the step is smaller.
3. Rounding error is about half a step for every value, and the step is set by the range. Narrowing the range a little makes a few extreme values err more (they are clamped) and every other value err less, which can lower the total.
4. The mean square of the activations entering each input column. A weight's error is multiplied by its column's activation, so errors in columns with large activations change the output more.
5. The model's output error is the weight error multiplied by activations. Squared weight error treats all columns alike, so reducing it can move error into columns that matter; the weighted error measures what the output feels.
6. Its unsigned operand is the 4-bit codes, which are already unsigned (0..15); the activations are the signed operand, as `vpdpbusd` requires. Chapter 18 needed the shift because both of its operands were signed.

## Further reading

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2023.
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2024.
- llama.cpp: the `ggml-quants.c` source (k-quants) and the `llama-imatrix` tool.
- Rouhani et al., "Microscaling Data Formats for Deep Learning", 2023 (the OCP MX formats).
- Next: [Chapter 20: FlashAttention](../20-flash-attention/README.md). Attention in one pass over the cache.
