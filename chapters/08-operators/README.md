# Chapter 8: Neural network operators

> **In one sentence:** between the matrix multiplications of a model sit a few small operators (softmax, normalization, activations, residual adds, embedding lookups) that cost little time but are easy to get numerically wrong, so each must be written to be stable, allocation-free and tested against a precise reference.

**Where this fits:** chapters 5-7 made the linear layers fast. A transformer layer is linear layers glued together by the operators in this chapter. Chapter 12 builds attention from softmax, chapter 13 assembles the full model from all of them, and chapter 15 uses softmax again to turn the model's output into probabilities.

**You need:** chapter 2 (floating point, overflow, accumulation) and chapter 6 (vectorization).

**You will build:** ReLU, SiLU, GELU (two ways), SwiGLU, a naive and a stable softmax, a fast vectorizable `exp`, log-softmax, RMSNorm, LayerNorm, embedding lookup and residual add. Each is tested against an `f64` reference, and the demo measures stability, speed and the cost of allocating in the hot path.

---

## 1. The intuition

A factory line (the model) has a few huge machines doing the heavy work: the presses, which are the matrix multiplications. Between the presses are small stations: one that trims every part to a standard size (normalization), one that bends parts past a threshold (activation functions), one that sorts parts into bins by likelihood (softmax). Each small station takes little time, but if one of them is miscalibrated, every part after it is wrong, and the defect is hard to trace back.

**Where the analogy breaks:** in a factory, a miscalibrated station usually produces obviously bad parts. A numerically unstable operator in a model often produces answers that look fine for most inputs and then, for some rare input (a very long prompt, an unusual token), produces NaN, which then spreads through every later layer. You cannot find these bugs by looking at typical outputs; you find them by testing edge cases against a precise reference.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Operator** | One step of a model's computation: a matmul, a softmax, a norm... |
| **Element-wise** | Applied to each number independently (activations, residual add). |
| **Row-wise** | Needs a whole row at once (softmax, normalization), because it uses a sum or maximum. |
| **Activation function** | A non-linear function applied element-wise between linear layers. |
| **Logits** | Raw, unnormalized scores. Softmax turns them into probabilities. |
| **Softmax** | `e^xᵢ / Σⱼ e^xⱼ`: turns a vector of scores into probabilities that sum to 1. |
| **Numerically stable** | Gives accurate results over the whole input range, without overflow or catastrophic cancellation. |
| **Normalization** | Rescaling a vector to a standard size (RMSNorm, LayerNorm). |
| **Residual connection** | Adding a layer's input back to its output: `x = x + f(x)`. |
| **Embedding** | A table with one learned vector per token; lookup is by token ID. |
| **ulp** | Unit in the last place: the gap between adjacent floats (chapter 2). |
| **Minimax polynomial** | The polynomial of a given degree whose worst-case error over a range is smallest. |

## 3. The concepts in depth

### 3.1 Where these operators sit in a transformer

This is one layer of a Llama-style model (chapter 13 builds it in full). Everything that is not a matmul is from this chapter:

```text
 x ──┬─► RMSNorm ─► Q,K,V matmuls ─► attention (uses softmax) ─► output matmul ─► (+) ──┬─► ...
     └──────────────────────────── residual ──────────────────────────────────────────┘ │
                                                                                         │
 ... ┬─► RMSNorm ─► gate matmul ─┐                                                       │
     │              up matmul ───┴─► SwiGLU ─► down matmul ─► (+) ──► next layer          │
     └─────────────────────────── residual ────────────────┘                             
```

In FLOPs, these operators are a rounding error: RMSNorm on a 576-wide vector is about 2,000 operations, while the matmuls next to it are hundreds of thousands. In correctness they are critical, and in time they are not always free: chapter 17's profiler will show softmax over a 49,152-token vocabulary costing a noticeable slice of each token when done naively.

### 3.2 Activation functions

Without a non-linear function between them, stacked linear layers collapse into a single linear layer. Activations are that non-linearity:

- **ReLU**: `max(0, x)`. Old, simple, rarely used in LLMs now.
- **GELU**: `x · Φ(x)`, where Φ is the standard normal cumulative distribution, `Φ(x) = ½(1 + erf(x/√2))`. Used by GPT-2, BERT and many others. A `tanh` approximation is common because `erf` used to be slow. The demo measures the difference between the two versions at up to 4.7 × 10⁻⁴: small, but not zero. **A model must be run with the same variant it was trained with**; mixing them is a classic source of "the numbers are slightly off from the reference".
- **SiLU** (or swish): `x · σ(x)`, with σ the logistic sigmoid `1/(1 + e⁻ˣ)`. Used by Llama and most recent models.
- **SwiGLU**: Llama's feed-forward block computes two projections of the input, `gate` and `up`, and combines them as `silu(gate) · up`. This "gated" form works better in practice than a plain activation and is why Llama's MLP has three weight matrices (gate, up, down) instead of two.

### 3.3 Softmax, and why the naive version breaks

Softmax turns scores into probabilities:

```text
softmax(x)ᵢ = e^xᵢ / Σⱼ e^xⱼ
```

The naive implementation computes `e^xᵢ` directly. `f32` overflows to infinity for `e^x` with x above about 88.7. Attention scores and logits can easily exceed that, and then the result is `inf / inf = NaN`. The demo shows `softmax_naive([1000, 999, 998])` returning `[NaN, NaN, NaN]`.

The fix is one line of algebra. For any constant c:

```text
e^xᵢ / Σⱼ e^xⱼ  =  e^(xᵢ − c) / Σⱼ e^(xⱼ − c)
```

The `e^(−c)` factor cancels. Choosing c = max(x) makes every exponent ≤ 0, so every `e^(xᵢ − c)` is between 0 and 1 and nothing can overflow. At least one term (the maximum itself) is exactly `e^0 = 1`, so the denominator is at least 1 and cannot underflow to zero either. The stable version returns `[0.665, 0.245, 0.090]`.

Every real implementation of softmax subtracts the maximum. Chapter 20 extends the idea to an *online* softmax that updates the maximum as it goes, which is the core of FlashAttention.

### 3.4 Log-softmax

Often you want `log(softmax(x))`: log-probabilities of tokens, for perplexity (chapter 18) or for returning logprobs from an API. Computing softmax and then taking the log fails for very unlikely tokens: their probability underflows to 0 and `log(0) = −∞`. The stable form never builds the probabilities:

```text
log softmax(x)ᵢ = xᵢ − max − log Σⱼ e^(xⱼ − max)
```

The subtraction stays accurate for any logit value.

### 3.5 Normalization

Deep networks keep the size of their activation vectors under control by normalizing them before each block.

**LayerNorm** (GPT-2, BERT): subtract the mean, divide by the standard deviation, then apply a learned per-element scale and shift.

```text
out = (x − mean(x)) / sqrt(var(x) + ε) · weight + bias
```

**RMSNorm** (Llama, Mistral, Qwen, SmolLM2): skip the mean, divide by the root mean square, apply a learned scale.

```text
out = x / sqrt(mean(x²) + ε) · weight
```

RMSNorm is cheaper (one reduction instead of two) and works as well in practice, which is why recent models use it.

Two numerical details:

- **ε** (epsilon, typically 10⁻⁵ or 10⁻⁶) prevents division by zero for an all-zero vector. Its value is part of the model's definition; SmolLM2's config says `rms_norm_eps: 1e-05`. Using a different ε gives slightly different outputs.
- **Two-pass variance.** Variance can be computed in one pass as `mean(x²) − mean(x)²`. When the mean is large compared with the spread (the test uses values around 100 with a spread of 1), the two terms are nearly equal and their difference loses most of its significant digits: catastrophic cancellation. Our `layer_norm` first computes the mean, then the average squared distance from it. Two passes over a vector that is in L1 cache anyway cost almost nothing.

### 3.6 The price of `exp`

Softmax needs one `exp` per element: 49,152 of them per generated token when turning SmolLM2's logits into probabilities, and one per (query, key) pair in attention. The standard `f32::exp` calls a C library function for each element, which is accurate and reasonably fast, but a function call per element prevents the compiler from vectorizing the loop.

`exp_fast` computes `e^x` with plain arithmetic the compiler can vectorize:

1. **Range reduction.** Write `x = n · ln 2 + r`, with `n` a whole number and `|r| ≤ ln 2 / 2 ≈ 0.35`. Then `e^x = 2^n · e^r`.
2. **`2^n` for free.** A float's exponent field holds a power of two, so `2^n` is just the bit pattern with `n + 127` in the exponent field.
3. **`e^r` by polynomial.** Because `r` is small, a degree-7 polynomial approximates `e^r` very well.

Two engineering details turned out to matter, and both were found by measuring:

- **Which polynomial.** A first version used the Taylor series `1 + r + r²/2! + ... + r⁶/6!`. Its worst relative error over [−87, 88] was 4.7 × 10⁻⁷. The same number of terms with *minimax* coefficients (from the Cephes math library, fitted to minimize the worst-case error over the range instead of being exact at 0) gave 8.1 × 10⁻⁸, better than 1 ulp and about as accurate as the standard library (6 × 10⁻⁸). Same cost, 6x more accurate.
- **How to build `2^n`.** The first version computed `n as i32`. In Rust, float-to-integer `as` *saturates* (out-of-range values become `i32::MAX`/`MIN`, NaN becomes 0), which needs extra compare-and-select instructions in every vector lane. Getting `n + 127` from integer arithmetic on the float's bits (section 4.3) avoids the cast. On the reference machine this took the element-wise cost from 2.4 to about 1.0 ns (default target).

### 3.7 Operators must not allocate

Every function in this crate writes into a buffer the caller provides, or works in place. Section 5 measures what an allocation costs: an RMSNorm that returns a fresh `Vec` took about 76-79 ns longer per call than one writing into a reused buffer (measured in two runs), an 18-19% slowdown for this small operator. A 30-layer model calls RMSNorm 61 times per token, plus dozens of other operators; allocating in each would add microseconds per token, fragment the heap over millions of tokens, and make latency less predictable (an allocator occasionally has to ask the OS for memory, which can take much longer than usual).

The standard design, which the engine from chapter 14 on follows: **allocate every buffer the forward pass needs once, when the model is loaded or a request starts, and reuse them for every token.**

### 3.8 Embedding lookup and residual add

An embedding table is `vocab_size × hidden_size` numbers. "Looking up" token 42 means taking row 42. In Rust that is a slice of the table: a pointer and a length, no copying. Chapter 3's views, in their simplest form.

A residual connection `x = x + f(x)` is an in-place element-wise add. It is the reason the model can have 30 layers: each layer only has to learn a *correction* to its input, and the original signal passes through unchanged if the layer contributes nothing.

## 4. The code

Everything is in [`src/lib.rs`](src/lib.rs); the demo is [`src/main.rs`](src/main.rs).

### 4.1 Activations

<!-- file: src/lib.rs -->
```rust
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (also called swish): `x · sigmoid(x)`. Used by Llama-style models.
pub fn silu(x: &mut [f32]) {
    for v in x {
        *v *= sigmoid(*v);
    }
}
```

`sigmoid` is stable as written for all `x`: for very negative `x`, `(-x).exp()` overflows to infinity and `1 / (1 + inf)` is exactly 0, which is the correct limit. For very positive `x`, `exp` underflows to 0 and the result is 1.

`for v in x` iterates over `&mut f32` references because `x` is `&mut [f32]`, so `*v *= ...` updates the slice in place.

<!-- file: src/lib.rs -->
```rust
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = g * sigmoid(g) * u;
    }
}
```

Doing `silu(gate)` and then multiplying by `up` in a separate loop would read and write `gate` twice. Combining both steps in one loop (**fusing** them) reads each input once and writes the output once. With vectors of 1,536 floats that all fit in L1 the gain here is small, but on a GPU, where each separate operator is a separate kernel launch that goes through slow memory, fusion is one of the most important optimizations (chapters 20 and 29).

### 4.2 Stable softmax

<!-- file: src/lib.rs -->
```rust
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}
```

Three passes: find the maximum, exponentiate and sum, scale. `fold(f32::NEG_INFINITY, f32::max)` starts from −∞ so that any real value replaces it. Multiplying by `inv` instead of dividing by `sum` replaces n divisions (slow) with one division and n multiplications (fast).

A detail the tests caught: this version adds all n exponentials into a **single** running sum. Over a 49,152-element vocabulary its worst relative error against an `f64` reference was 2.8 × 10⁻⁵, dominated by that sum (chapter 2's accumulation error). The fast version below keeps eight running sums and came out *more accurate*, 4.2 × 10⁻⁶, even though its `exp` is approximate. Both are far more accurate than the model needs, but it is a good example of why "optimized" and "accurate" are not opposites.

### 4.3 The fast `exp`

<!-- file: src/lib.rs -->
```rust
#[inline]
pub fn exp_fast(x: f32) -> f32 {
    // Outside this range the result would underflow or overflow; clamping
    // keeps n in the range where 2^n is a normal f32.
    let x = x.clamp(-87.0, 88.0);
    // After adding ROUNDER, the integer n = round(x·log2 e) sits in the low
    // mantissa bits of `shifted`: its bit pattern is ROUNDER's plus n.
    let shifted = x * LOG2_E + ROUNDER;
    let n = shifted - ROUNDER;
    let r = x - n * LN2_HI - n * LN2_LO;
```

- `#[inline]` lets other crates inline this function. Without it, a non-generic function from another crate is compiled once, in its own crate, and called through a normal function call: the calling loop cannot be vectorized. This also cost us a round of debugging: the first version of the demo (a separate crate from the library) measured `exp_fast` as slower than `f32::exp` because every call was a real call.
- `clamp(-87.0, 88.0)`: `e^88` is close to the largest `f32` and `e^-87` close to the smallest normal one. Clamping keeps `n` in the range where the exponent trick produces a valid float. For softmax, where all inputs are ≤ 0, only the lower bound ever matters, and there the result is ~10⁻³⁸ instead of something even smaller, which is a meaningless difference for a probability.
- **The rounding trick.** `ROUNDER` is 1.5 × 2²³ = 12,582,912. An `f32` has 23 mantissa bits, so numbers between 2²³ and 2²⁴ are spaced exactly 1 apart: adding `x · log2(e)` to `ROUNDER` rounds it to a whole number as a side effect of normal float addition. Subtracting `ROUNDER` again gives `n` as a float. No `round()` call (which is a library call on the default x86-64 target).
- **Cody-Waite reduction.** `r = x − n · ln 2` subtracts two nearly equal numbers, so any error in `n · ln 2` becomes a large relative error in `r`. Splitting ln 2 into `LN2_HI` (which has enough trailing zero bits that `n · LN2_HI` is exact for |n| < 512) and a small correction `LN2_LO` keeps `r` accurate.

<!-- file: src/lib.rs -->
```rust
    // e^r ≈ 1 + r + r²·P(r), P evaluated in Horner form.
    let mut p = 1.987_569_1e-4_f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    let e_r = p * r * r + r + 1.0;
```

The polynomial is evaluated in **Horner form**, `((a·r + b)·r + c)·r + ...`: one multiply and one add per coefficient, and no powers computed separately. The coefficients are close to the Taylor ones (1/2, 1/6 ≈ 0.16667, 1/24 ≈ 0.041667, 1/120 ≈ 0.0083333, ...) but nudged to spread the error evenly across the range.

<!-- file: src/lib.rs -->
```rust
    let two_to_n = f32::from_bits(shifted.to_bits().wrapping_sub(ROUNDER.to_bits() - 127) << 23);
    e_r * two_to_n
```

`shifted`'s bit pattern is `ROUNDER`'s bit pattern plus `n` (because at that magnitude, one unit of the float is one unit of the mantissa). Subtracting `ROUNDER.to_bits() − 127` leaves `n + 127` as an integer. Shifting it left by 23 puts it in the exponent field with a zero mantissa: exactly `2^n`. Everything here is integer arithmetic on `u32`, which vectorizes into a couple of instructions.

### 4.4 RMSNorm and LayerNorm

<!-- file: src/lib.rs -->
```rust
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}
```

One reduction (the mean of squares), one square root, one division, then a single element-wise pass. `x` and `out` are separate slices: the engine keeps the un-normalized `x` for the residual connection and passes the normalized copy to the next matmul, so in-place would be wrong here.

<!-- file: src/lib.rs -->
```rust
    let mean = x.iter().sum::<f32>() / n;
    // Two passes (mean first, then variance around it) is more accurate
    // than the one-pass formula E[x²] - E[x]², which can cancel badly.
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
```

The two-pass variance of section 3.5. The test `layer_norm_output_has_zero_mean_and_unit_variance` feeds values around 100 with a spread of about 0.7, exactly the case where the one-pass formula would lose most of its precision.

### 4.5 Testing against `f64`

<!-- file: src/lib.rs -->
```rust
    #[test]
    fn exp_fast_is_accurate() {
        let mut worst = 0.0f64;
        for x in sweep(-87.0, 88.0, 2_000_000) {
            let want = f64::from(x).exp();
            let rel = ((f64::from(exp_fast(x)) - want) / want).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 1.5e-7, "worst relative error {worst:e}");
```

Two million evenly spaced points across the whole valid range, each compared with `f64::exp` (which is accurate to far better than `f32` precision). The threshold 1.5 × 10⁻⁷ is about one `f32` ulp; the measured worst case is 8.1 × 10⁻⁸. Numeric tests should say *what* accuracy they guarantee, in numbers, and fail if the implementation gets worse.

## 5. Run it

```bash
cargo test -p ch08-operators
cargo run --release -p ch08-operators
```

On the reference machine:

```text
== 1. softmax of [1000, 999, 998]
   naive:  [NaN, NaN, NaN]
   stable: [0.66524094, 0.24472848, 0.09003057]

== 2. exp over 1,000,000 values
   f32::exp:   2.38ms  (2.38 ns each)
   exp_fast:   1.02ms  (1.02 ns each)

== 3. softmax speed
   attention row, 1024 keys   softmax    3.06µs   softmax_fast    1.50µs   (2.0x)
   vocabulary, 49152 tokens   softmax  154.51µs   softmax_fast   82.74µs   (1.9x)

== 4. RMSNorm of a 576-wide vector: allocating vs reusing the output buffer
   into a reused buffer: 423.00ns
   into a new Vec:       499.00ns   (+76 ns per call)

== 5. GELU: tanh approximation vs exact (erf)
   largest difference 4.73e-4 at x = 2.691
```

- **The naive softmax fails completely** on inputs that a real model produces routinely.
- **`exp_fast` is 2.3x faster** than the standard library on the default target, and softmax built on it is about 2x faster. With `-C target-cpu=native` (wider vectors), the gap grew to about 3.5x for the vocabulary softmax in a separate measurement on this machine. 80-150 µs per token for the vocabulary softmax sounds small, but at 50 tokens/s it is 0.5% of all time; in chapter 15, sampling does several passes over the vocabulary, and those add up.
- **Allocating the output costs about 76 ns per call** (18% of the operator's time). Small per call, avoidable, and it adds up over thousands of calls per token.
- **The two GELU variants differ by up to 4.7 × 10⁻⁴**, near x = 2.7. Use the variant the model was trained with.

## 6. The Rust behind it

**In-place and out-parameter signatures.** `fn softmax(x: &mut [f32])` and `fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32])` make allocation impossible inside the operator and visible at the call site. The borrow checker also guarantees `out` does not alias `x`: you cannot pass the same slice as both `&[f32]` and `&mut [f32]`. In C, forgetting that `out` and `x` might overlap (and the compiler therefore having to assume they might) is both a correctness risk and a performance cost, because it blocks vectorization. Rust's aliasing rules give the compiler this guarantee for free.

**`#[inline]` across crates.** Generic functions are always available for inlining in other crates (they are compiled where they are used). Non-generic ones are not, unless marked `#[inline]` or the build uses link-time optimization (LTO). For tiny, hot functions like `exp_fast`, `#[inline]` is the difference between a vector loop and a function call per element.

**Float-to-int `as` is saturating.** `f32::NAN as i32` is 0 and `1e10f32 as i32` is `i32::MAX`. This makes `as` safe (no undefined behaviour, unlike C), but it is not free inside a hot vector loop. When you know the value is in range, integer bit manipulation, or the `unsafe` `to_int_unchecked`, avoids the checks.

**`clamp`, `copysign`, `fold`.** Small standard-library functions that express intent directly: `x.clamp(lo, hi)`, `y.copysign(x)` (use `x`'s sign on `y`, as `erf` does for negative inputs), `fold(init, f32::max)` for a maximum that handles an empty slice by returning −∞.

**No stable `erf`.** Rust's standard library does not yet have `erf` for floats (it is available on nightly). We implemented the classic Abramowitz-Stegun approximation; the `libm` crate provides a full-precision one.

## 7. Mistakes you will make

- **Forgetting to subtract the maximum** in any softmax you write by hand: in attention, in sampling, in a custom loss. Everything works on test inputs until the day logits exceed 88.
- **The wrong epsilon, or putting it in the wrong place.** `x / (sqrt(mean) + eps)` is not `x / sqrt(mean + eps)`. Match the reference implementation exactly.
- **Mixing up GELU variants** or SiLU and GELU. The model runs and produces plausible but slightly worse text.
- **Normalizing in place when the residual needs the original.** The model then adds the normalized vector back instead of the raw one; the outputs are wrong but not obviously broken.
- **Taking `log` of a softmax** for log-probabilities. Use log-softmax.
- **Allocating inside the per-token loop** for convenience. Every `Vec::new()`, `to_vec()` and `collect()` in the forward pass is a candidate for a preallocated buffer.

## 8. How the professionals do it

- **Fused kernels.** Production engines combine RMSNorm with the following matmul's input quantization, SwiGLU with the up/gate matmul, the residual add with the next norm, and softmax with the attention matmuls (FlashAttention, chapter 20). On GPUs each separate operator is a kernel launch and a round trip through memory, so fusion is often worth more than any single kernel's speed.
- **Vectorized transcendental functions.** llama.cpp has SIMD `exp`, `silu` and `gelu` for each ISA; GPU libraries use hardware approximations (CUDA's `__expf`), sometimes with explicit accuracy trade-offs. GELU is sometimes implemented with a lookup table.
- **Mixed precision.** Norms and softmax are usually computed in `f32` even when weights and activations are stored in `bf16` or 8-bit formats, because the reductions (sums of squares, sums of exponentials) are exactly where low precision hurts (chapter 2).
- **Reference testing.** Every serious engine compares each operator (and each whole layer) against a reference implementation, usually PyTorch, with explicit tolerances. Chapter 16 does this for the whole model and chapter 30 for each layer.

## 9. Exercises

1. **Temperature.** Chapter 15 divides logits by a temperature T before softmax. Write `softmax_with_temperature(x, t)`. What happens as T → 0 and as T → ∞? Is dividing by T before or after subtracting the maximum more accurate?
2. **One-pass variance.** Implement LayerNorm with the one-pass formula `var = mean(x²) − mean(x)²` and run `layer_norm_output_has_zero_mean_and_unit_variance` against it. What happens? Now shift the test input to values around 10,000.
3. **Fast SiLU.** Write `silu_fast` using `exp_fast`. Measure it against `silu` on a 1,536-long vector (the size of SmolLM2's MLP), and measure its worst error over [−20, 20].
4. **Taylor versus minimax.** Replace the Cephes coefficients with 1/720, 1/120, 1/24, 1/6, 1/2 (the Taylor coefficients, one fewer degree). Run `exp_fast_is_accurate`. How much worse is the worst-case error?
5. **Where is the error of plain softmax?** Change `softmax` to use eight running sums (like `softmax_fast`) but keep `f32::exp`. Does its worst relative error on the vocabulary test approach `softmax_fast`'s? What does that tell you?
6. **Count the allocations.** SmolLM2 has 30 layers. Count how many RMSNorm, SwiGLU and residual-add calls one token needs. If each allocated its output (about 76-79 ns extra, as measured for RMSNorm), how much time per token would that add?

## 10. Check yourself

1. Why does subtracting the maximum before exponentiating not change softmax's result?
2. Why is log-softmax computed directly instead of as `log(softmax(x))`?
3. What is the difference between LayerNorm and RMSNorm?
4. Why can't the compiler vectorize a loop that calls `f32::exp`, and how does `exp_fast` get around it?
5. Why does SwiGLU need three weight matrices?
6. Why should operators write into caller-provided buffers?

## 11. Recap

- The operators between matmuls are cheap in FLOPs but easy to get numerically wrong. Test each against an `f64` reference over the whole input range.
- Softmax must subtract the maximum; log-probabilities must use log-softmax.
- RMSNorm is LayerNorm without the mean. Epsilon and the exact formula are part of the model's definition.
- Llama-style MLPs use SwiGLU: `silu(gate) · up`, fused into one pass.
- A vectorizable `exp` (range reduction, exponent bits, minimax polynomial) is about 2.3x faster than the library call and just as accurate. Mark small hot functions `#[inline]`, and avoid saturating `as` casts in vector loops.
- Allocate buffers once and reuse them; allocation in the forward pass costs time and predictability.

## Answers

**Exercises**

1. Divide every logit by T, then apply the stable softmax. As T → 0 the largest logit dominates and softmax approaches a one-hot vector (greedy choice); as T → ∞ all probabilities approach 1/n (uniform). Dividing first and then subtracting the maximum is the standard order; since dividing by T scales the maximum by the same factor, both orders are mathematically equal, and the difference in rounding is negligible. What matters is that the maximum is subtracted *after* any operation that changes the values' range, so the largest exponent is 0. Chapter 15 implements this.
2. With the one-pass formula on values around 100 (squares around 10,000), `mean(x²)` and `mean(x)²` agree in their first four or five digits, and their difference keeps only two or three significant digits of `f32`'s seven. The variance comes out noticeably wrong, the output's variance is not 1, and the test fails. Around 10,000 (squares around 10⁸) the difference can even come out negative, giving `sqrt` of a negative number: NaN.
3. `silu_fast(x) = x / (1 + exp_fast(-x))`. Measured on the reference machine for a 1,536-long vector: 1.84 µs against 4.00 µs for `silu`, 2.2x faster. Its worst relative error over [−20, 20] was 1.3 × 10⁻⁷ (excluding points where SiLU is within 10⁻⁶ of zero, where relative error is meaningless).
4. Measured on the reference machine while developing this chapter: the degree-6 Taylor polynomial had a worst-case relative error of 4.7 × 10⁻⁷ over [−87, 88], against 8.1 × 10⁻⁸ for the Cephes coefficients: about 6x worse for the same cost.
5. Measured: with eight running sums and the standard `f32::exp`, the worst relative error is 4.17 × 10⁻⁶, identical to `softmax_fast`'s. The whole gap came from the single running sum, not from `exp`. Accuracy problems in reductions are usually about *how you add*, not about the function being summed.
6. Per layer: 2 RMSNorms, 1 SwiGLU, 2 residual adds = 5 operators, so 150 for 30 layers, plus the final norm: 151 calls. At ~80 ns each, about 12 µs per token. Small against a token that takes ~10 ms, but pure waste, and allocation cost grows with vector size and heap fragmentation. The same reasoning applies to every intermediate buffer in the forward pass (Q, K, V, attention scores, MLP hidden states), where the vectors are larger.

**Check yourself**

1. Subtracting c from every input multiplies every `e^xᵢ` by the same factor `e^(−c)`, in the numerator and in the denominator, so it cancels.
2. Because the probabilities of unlikely tokens underflow to 0, and `log(0)` is −∞. Log-softmax works with the logits directly and stays finite and accurate.
3. LayerNorm subtracts the mean and divides by the standard deviation, then scales and shifts. RMSNorm divides by the root mean square (no mean subtraction, no shift), which is one reduction cheaper.
4. Each `f32::exp` call is a call into a library function, which the compiler cannot turn into vector instructions. `exp_fast` uses only multiplications, additions and integer bit operations, which have vector equivalents, and it is `#[inline]` so the compiler sees its body inside the loop.
5. Two of them (gate and up) project the input to the hidden size; SwiGLU combines them as `silu(gate) · up`; the third (down) projects the result back to the model size.
6. So the forward pass never allocates: no time spent in the allocator, no heap fragmentation over millions of tokens, predictable latency, and the caller decides where memory lives and how long it is reused.

## Further reading

- Stephen Moshier, the Cephes mathematical library (`expf.c`): the source of the `exp` coefficients, and a model for implementing transcendental functions.
- Zhang and Sennrich, "Root Mean Square Layer Normalization", 2019.
- Noam Shazeer, "GLU Variants Improve Transformer", 2020. Where SwiGLU comes from.
- Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)", 2016.
- Next: [Chapter 9: Weights on disk](../09-safetensors/README.md). Our operators are ready; now we need real weights to feed them.
