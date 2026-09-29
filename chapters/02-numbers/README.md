# Chapter 2: Numbers inside a model

> **In one sentence:** a model's weights can be stored in 32, 16, 8 or even 4 bits per number, and that choice sets how much memory the model needs, how fast it runs, and how accurate it stays.

**Where this fits:** chapter 1 showed that speed is mostly about how many bytes of weights we read. This chapter is about what those bytes *are*. Every later chapter depends on it: the weight loader (chapter 9) reads `bf16` from disk, the real model (chapter 16) computes with it, and quantization (chapters 18-19) pushes the same ideas down to 8 and 4 bits.

**You need:** chapter 1. Comfort with binary and hexadecimal helps, but the lesson explains every bit it uses.

**You will build:** the `bf16`, `f16` and `fp8` formats from scratch on top of plain integers, with rounding that matches PyTorch bit for bit, plus experiments on rounding error, overflow, and the right and wrong way to add numbers up.

---

## 1. The intuition

Think of a ruler.

A ruler with millimetre marks lets you measure anything from 1 mm to 30 cm. Now imagine a strange ruler whose marks get further apart as you move right: 1 mm apart near zero, 1 cm apart at 10 cm, 10 cm apart at 1 m, and so on out to kilometres. You can measure both a grain of sand and a road with the same ruler, and your error is always about the same *fraction* of what you are measuring (say 1%), not the same number of millimetres.

That is a **floating-point number**. The marks are the values the format can represent exactly. Anything between two marks gets rounded to the nearest mark.

The formats in this chapter are rulers with different numbers of marks:

- `f32` has about 4 billion marks. Fine spacing, huge range.
- `bf16` has 65,536 marks spread over the *same* huge range, so they are 65,536 times further apart.
- `f16` also has 65,536 marks, but packs them into a much *shorter* range, so they are closer together, and the ruler ends at 65,504.
- `fp8` has 256 marks. Coarse, and very short.

Fewer marks means fewer bits per number, which means fewer bytes to read, which (chapter 1) means faster inference.

**Where the analogy breaks:** a real ruler's marks are evenly spaced within each stretch and the stretches double in length. Floating-point marks do exactly that: they are evenly spaced between 1 and 2, twice as far apart between 2 and 4, and so on. Near zero there is a special region ("subnormal" numbers) where the spacing stops shrinking. We will meet it below.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Bit pattern** | The raw 0s and 1s a number is stored as. |
| **Sign bit** | One bit: 0 for positive, 1 for negative. |
| **Exponent** | Which power of two the number sits near. Sets the *range*. |
| **Mantissa** (or significand, fraction) | The digits after the leading 1. Sets the *precision*. |
| **Bias** | A constant subtracted from the stored exponent so it can be negative. 127 for `f32`, 15 for `f16`. |
| **ULP** | "Unit in the last place": the gap between one representable number and the next. |
| **Machine epsilon** | The ULP of 1.0. For `f32` it is 2⁻²³ ≈ 1.19 × 10⁻⁷. |
| **Rounding mode** | The rule for picking a representable value when the exact one falls between two marks. |
| **Round half to even** | Round to nearest; exact ties go to the value whose last bit is 0. |
| **Overflow** | A result too large for the format: becomes infinity (or saturates to the maximum). |
| **Underflow** | A result too small: loses precision, then becomes zero. |
| **Subnormal** | Tiny numbers near zero with reduced precision, filling the gap down to 0. |
| **NaN** | "Not a number": the result of 0/0, ∞−∞ and similar. |
| **Accumulator** | The running total in a sum or dot product. |

## 3. The concepts in depth

### 3.1 Why the format matters so much for inference

Three things depend on the bytes per number:

1. **Memory.** A 8-billion-parameter model is 32 GB in `f32`, 16 GB in `bf16`, 8 GB in 8-bit, 4 GB in 4-bit. The difference between "fits on this GPU" and "does not".
2. **Speed.** Chapter 1's rule: decode time per token ≈ weight bytes ÷ memory bandwidth. Halve the bytes, nearly double the tokens per second.
3. **Quality.** Fewer bits means bigger rounding errors. Whether the model still gives good answers is something you must *measure*, never assume.

Every production system trades these against each other. You cannot make those trade-offs without understanding what the bits mean.

### 3.2 How an `f32` is laid out

A 32-bit float is three fields packed into 32 bits:

```text
 bit 31   bits 30..23        bits 22..0
┌──────┬──────────────┬─────────────────────────┐
│ sign │ exponent (8) │     mantissa (23)       │
└──────┴──────────────┴─────────────────────────┘

value = (−1)^sign × 2^(exponent − 127) × 1.mantissa   (in binary)
```

For "normal" numbers the mantissa has an **implicit leading 1** that is not stored. The 23 stored bits are the fraction after the binary point.

Worked example, `−2.5`:

- −2.5 = −1 × 1.25 × 2¹. In binary 1.25 is `1.01`.
- sign = 1.
- exponent: we need 2¹, and the stored exponent is the real one plus the bias 127, so 128 = `10000000`.
- mantissa: the bits after "1." are `01`, then zeros: `01000000000000000000000`.

The demo prints exactly this: `-2.5e0  f32 1|10000000|01000000000000000000000`.

Worked example, `0.1`:

- 0.1 in binary is `0.000110011001100110011...` repeating forever, like 1/3 in decimal.
- 23 bits cannot hold an infinite pattern, so it is rounded. The stored value is 0.100000001490116...

This is why `0.1 + 0.2 != 0.3` in every language that uses binary floats. It is not a bug; the marks on the ruler simply do not include 0.1.

**Range and precision are separate knobs.** The exponent field decides how big and how small numbers can get (range). The mantissa field decides how many significant digits you keep (precision). With 23 mantissa bits, the gap between neighbouring `f32` values is about 2⁻²³ ≈ 1.2 × 10⁻⁷ of the value itself, so `f32` keeps about 7 significant decimal digits.

**Special patterns:**

- Exponent all zeros, mantissa zero: **±0**. (Yes, there is a negative zero.)
- Exponent all zeros, mantissa non-zero: **subnormal** numbers. No implicit 1, so precision drops as they get smaller. They fill the gap between the smallest normal number and zero.
- Exponent all ones, mantissa zero: **±infinity**.
- Exponent all ones, mantissa non-zero: **NaN**.

### 3.3 `bf16`: the top half of an `f32`

`bf16` ("brain float 16", invented at Google Brain) keeps the sign and the full 8-bit exponent of `f32`, and only the top 7 mantissa bits:

```text
f32:   s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm
bf16:  s eeeeeeee mmmmmmm                  ← the first 16 bits, nothing else
```

Consequences:

- **Same range as `f32`**: up to about 3.4 × 10³⁸. A value that fits in `f32` (almost) never overflows in `bf16`.
- **Much less precision**: 7 mantissa bits, so a relative gap of 2⁻⁷ ≈ 0.8% between neighbours. Only 2-3 significant decimal digits.
- **Converting to `f32` is free**: shift the 16 bits left by 16. No arithmetic, no branches, exact.
- **Converting from `f32` is almost free**: round, then keep the top 16 bits.

This is why `bf16` became the default format for training and serving large language models. Neural networks tolerate imprecise weights well (they are statistical objects to begin with), but they break badly when a value overflows to infinity. `bf16` gives up precision, which networks can afford, to keep range, which they cannot do without.

### 3.4 `f16`: more precision, less range

IEEE half precision splits its 16 bits differently: 5 exponent bits and 10 mantissa bits.

```text
f16:   s eeeee mmmmmmmmmm
```

- **Precision** is 8 times better than `bf16` (10 mantissa bits versus 7).
- **Range** is tiny: the largest finite value is 65,504. `300 × 300 = 90,000` does not fit.
- The smallest normal value is 2⁻¹⁴ ≈ 6.1 × 10⁻⁵. Below that, subnormals go down to 2⁻²⁴ ≈ 6 × 10⁻⁸ with shrinking precision, then zero.

`f16` works well for weights (which are small) and poorly for some activations (the intermediate values inside a network, which occasionally spike into the thousands). Models trained in `bf16` sometimes produce infinities when run in `f16`. If you ever see a model output NaN in `f16` but work in `bf16`, overflow is the first suspect.

### 3.5 `fp8`: 256 values

Recent accelerators (NVIDIA H100 and later, AMD MI300) compute natively in 8-bit floats. There are two common layouts:

- **E4M3**: 4 exponent bits, 3 mantissa bits. Largest value 448. No infinity. Used for weights and activations.
- **E5M2**: 5 exponent bits, 2 mantissa bits. Largest value 57,344. It is literally the top byte of an `f16`. Used where range matters more.

Look at the demo's rounding table: 9,982 of the 1,000,000 test weights in [−0.1, 0.1) were too small for E4M3 and became exactly zero, so the maximum relative error is 100%. 8-bit floats are never used "raw". They always come with a **scale factor**: multiply the tensor by a number that stretches its values to fill the format's range before converting, and divide by it afterwards. Exercise 5 does this. Chapter 18 builds the same idea for integers.

### 3.6 The formats side by side

| Format | Bits | Exponent | Mantissa | Largest value | Relative precision | Typical use |
|---|---|---|---|---|---|---|
| `f64` | 64 | 11 | 52 | 1.8 × 10³⁰⁸ | 2.2 × 10⁻¹⁶ | Reference checks in tests |
| `f32` | 32 | 8 | 23 | 3.4 × 10³⁸ | 1.2 × 10⁻⁷ | Accumulators, CPU inference, reference runs |
| `bf16` | 16 | 8 | 7 | 3.4 × 10³⁸ | 7.8 × 10⁻³ | Weights and activations of most LLMs |
| `f16` | 16 | 5 | 10 | 65,504 | 9.8 × 10⁻⁴ | Weights; activations on older GPUs |
| `fp8` E4M3 | 8 | 4 | 3 | 448 | 1.25 × 10⁻¹ | Weights and activations on newest GPUs, with scales |
| `fp8` E5M2 | 8 | 5 | 2 | 57,344 | 2.5 × 10⁻¹ | Gradients, some activations |

"Relative precision" here is the machine epsilon, the gap just above 1.0: 2 to the power of minus the mantissa bits.

### 3.7 Rounding: truncate or round to nearest

To turn an `f32` into a `bf16` you must throw away 16 mantissa bits. The lazy way is to just drop them (truncation). That always rounds towards zero, so every value gets slightly smaller in magnitude. The error is up to one full ULP and it is *biased*: across millions of weights, everything shrinks a little in the same direction.

**Round to nearest** looks at the dropped bits. More than half: round up. Less than half: round down. Exactly half: a tie.

For ties, **round half to even** picks whichever neighbour has a 0 as its last bit. Half the ties go up and half go down, so no systematic bias builds up. This is the IEEE 754 default, what CPUs do for `f32` arithmetic, and what PyTorch does when you call `.to(torch.bfloat16)`.

The demo measures the difference on a million weights: truncation has mean relative error 2.83 × 10⁻³, rounding 1.42 × 10⁻³. Rounding halves the error for the same 16 bits.

### 3.8 Accumulate in high precision

Here is the most important practical rule in this chapter:

> **Store numbers in a small format if you like. Add them up in a big one.**

A dot product of two 4,096-long vectors adds 4,096 products into one running total, the **accumulator**. If the accumulator is a `bf16`, something ugly happens. `bf16` has 8 significant bits (7 stored plus the implicit 1). The representable numbers around 256 are 254, 256, 258: the gap is 2. So `256 + 1 = 257` is exactly halfway, rounds to the even neighbour 256, and the total never moves again.

The demo shows it: adding 1.0 a thousand times gives 1000 in `f32` and **256** in `bf16`. Adding ten million values between 0 and 1 also gives 256 in a `bf16` accumulator: a 100% error.

This is why every serious kernel reads `bf16` or 8-bit inputs but keeps its accumulators in `f32`. GPU tensor cores do the same: `bf16` inputs, `f32` accumulate. When you write your own kernels (chapters 5, 6, 18, 19) you will do it too.

Even `f32` accumulators lose precision over long sums. Once the running total is 5,000,000, the gap between neighbouring `f32` values is 0.5, so each small addition is rounded to the nearest 0.5. Two classic fixes:

- **Pairwise summation**: split the list in half, sum each half recursively, add the halves. Totals being added are always of similar size, and the error grows like log(n) instead of n.
- **Kahan summation**: keep a second variable that captures what was lost in each addition and feed it back into the next one.

Measured on 10 million values: plain loop relative error 7.3 × 10⁻⁶, pairwise and Kahan 4.5 × 10⁻⁸. In practice, SIMD and multi-threaded reductions (chapters 6 and 7) are naturally pairwise-ish, because they sum several partial totals and combine them at the end. That is one reason fast code is often *more* accurate than a naive loop.

### 3.9 Addition is not associative, and why that matters for testing

In real numbers, (a + b) + c = a + (b + c). In floating point, not always:

```text
(1e8 + 1) − 1e8 = 0     because 1e8 + 1 rounds back to 1e8 in f32
(1e8 − 1e8) + 1 = 1
```

The demo sums the same million numbers forwards and backwards and gets −465.74414 and −465.74377.

Consequences you will run into:

- **Any change in the order of additions changes the result slightly.** Vectorizing a loop, splitting it across threads, changing a tile size, or changing the batch size can all change the last few digits.
- **So tests of numeric code must compare with a tolerance**, not with `==`. The usual pattern: compare against a high-precision reference (often `f64`) and allow an error proportional to the size of the values and the length of the sums.
- **"The same model gives slightly different answers in different runs" usually has this cause.** It is normally harmless, but it can flip a greedy choice between two nearly equal tokens, and from there the generated texts diverge. Chapter 30 returns to this ("batch invariance").

### 3.10 What it all costs

For single-user decode, time per token ≈ weight bytes ÷ bandwidth. At 50 GB/s:

| Model | f32 | bf16 | 8-bit | 4-bit |
|---|---|---|---|---|
| 135M params (SmolLM2, chapter 16) | 0.54 GB, 93 tok/s | 0.27 GB, 185 tok/s | 0.14 GB, 370 tok/s | 0.07 GB, 741 tok/s |
| 8B params | 32 GB, 1.6 tok/s | 16 GB, 3.1 tok/s | 8 GB, 6.2 tok/s | 4 GB, 12.5 tok/s |

These are ceilings, not predictions: real code also spends time on other work. But they tell you which lever is worth pulling.

## 4. The code

All of it is in [`src/lib.rs`](src/lib.rs). The demo is [`src/main.rs`](src/main.rs).

### 4.1 Looking at the bits

<!-- file: src/lib.rs -->
```rust
pub fn f32_fields(x: f32) -> F32Fields {
    let bits = x.to_bits();
    F32Fields {
        sign: bits >> 31,
        exponent: (bits >> 23) & 0xFF,
        mantissa: bits & 0x007F_FFFF,
    }
}
```

- `x.to_bits()` returns the four bytes of the float as a `u32`, **unchanged**. This is a reinterpretation, not a conversion. Compare `x as u32`, which converts the *value*: `1.5f32 as u32` is `1`, while `1.5f32.to_bits()` is `0x3FC0_0000`.
- `bits >> 31` moves the top bit down to position 0: the sign.
- `(bits >> 23) & 0xFF` moves the exponent down and masks off the sign above it. `0xFF` is eight 1-bits.
- `bits & 0x007F_FFFF` keeps the low 23 bits: the mantissa. Rust lets you put `_` in number literals anywhere, which makes masks readable.

### 4.2 The `Bf16` type

<!-- file: src/lib.rs -->
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bf16(u16);
```

This is a **newtype**: a struct with a single field, which exists to give a `u16` a different meaning.

Why not pass `u16` around? Because a `u16` holding bf16 bits and a `u16` holding a token ID or an f16 pattern are completely different things, and a function taking `u16` would accept all of them. With a newtype the compiler refuses to mix them up. You cannot add two `Bf16` values by accident, or pass `F16` bits where `Bf16` bits are expected. The wrapper costs nothing at runtime.

`#[repr(transparent)]` promises that `Bf16` has exactly the same memory layout as `u16`. That matters in chapter 9: a weight file contains raw bf16 bytes, and we want to view them as `&[Bf16]` without copying. Without this attribute the compiler would be free to lay the struct out differently.

The field is private (`Bf16(u16)`, not `Bf16(pub u16)`), so the only ways in and out are `from_bits`/`to_bits` and the conversions. Every place that touches raw bits is explicit.

`#[derive(Clone, Copy, ...)]`: `Copy` makes `Bf16` behave like a number, copied on assignment. It is two bytes, so copying is cheaper than borrowing. `Hash`, `Eq` and `PartialEq` compare *bit patterns*, which is right for a storage type. (Note that `f32` itself does not implement `Eq`, because NaN ≠ NaN.)

### 4.3 `bf16` to `f32`: one shift

<!-- file: src/lib.rs -->
```rust
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }
```

- `u32::from(self.0)` widens the `u16` to a `u32`. We use `From` instead of `as` because `From` only exists for conversions that can never lose information, so the compiler checks our reasoning. `as` would also compile if the types were the other way round and silently truncate.
- `<< 16` moves the bf16 bits into the top half, leaving 16 zero mantissa bits at the bottom.
- `f32::from_bits` reinterprets the result as a float.

Because every `bf16` is an `f32` with the low bits zero, this is exact. It also compiles to one or two instructions and vectorizes well, which is why `bf16` weights are so cheap to use: chapter 16's kernel does this shift in registers, just before the multiply.

### 4.4 `f32` to `bf16`: rounding with one addition

<!-- file: src/lib.rs -->
```rust
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        if x.is_nan() {
            // Truncating a NaN could clear every mantissa bit that survives,
            // which would turn it into infinity. Force a mantissa bit on.
            return Self((bits >> 16) as u16 | 0x0040);
        }
        // We are about to throw away the low 16 bits. Adding 0x7FFF rounds
        // up exactly when those bits are more than half (0x8000). Adding the
        // lowest kept bit as well makes an exact half round up only when
        // that bit is 1, which is what "ties to even" means.
        let lowest_kept_bit = (bits >> 16) & 1;
        let rounded = bits + 0x7FFF + lowest_kept_bit;
        Self((rounded >> 16) as u16)
    }
```

This is the standard trick (PyTorch and TensorFlow use the same one). Walk through it with two ties:

```text
x = 1.00390625 = 1 + 2⁻⁸      bits = 0x3F80_8000
    kept = 0x3F80 (last bit 0), dropped = 0x8000 (exactly half)
    0x3F80_8000 + 0x7FFF + 0 = 0x3F80_FFFF  → top half 0x3F80 = 1.0          (tie, rounded down to even)

x = 1.01171875 = 1 + 3×2⁻⁸    bits = 0x3F81_8000
    kept = 0x3F81 (last bit 1), dropped = 0x8000 (exactly half)
    0x3F81_8000 + 0x7FFF + 1 = 0x3F82_0000  → top half 0x3F82 = 1.015625     (tie, rounded up to even)
```

Any dropped value above `0x8000` carries into the kept bits (round up); anything below does not (round down). If the carry ripples all the way through the mantissa, it correctly increments the exponent: that is the beauty of the IEEE layout, where the bit patterns of positive floats are in the same order as their values. The largest finite `f32` values round up to infinity (`0x7F80`), which is the correct IEEE result.

- The NaN branch exists because a NaN whose only set mantissa bits are in the low half would become `0x7F80` after truncation, which is **infinity**. Setting bit `0x0040` keeps it a NaN.
- `bits + 0x7FFF + lowest_kept_bit` cannot overflow a `u32` for any non-NaN input (the largest is −infinity, `0xFF80_0000`). In debug builds Rust checks arithmetic overflow and would panic if we were wrong; the exhaustive tests run in debug mode, so this reasoning is checked.
- `(rounded >> 16) as u16`: here `as` is the right tool. After the shift the value fits in 16 bits, and `as` between integer types keeps the low bits.

### 4.5 `f16` to `f32`

<!-- file: src/lib.rs -->
```rust
        match exponent {
            // Zero or subnormal: value = mantissa × 2^-24. Multiplying a
            // small integer by a power of two is exact in f32.
            0 => {
                let magnitude = mantissa as f32 * f32::from_bits(0x3380_0000); // 2^-24
                if sign == 0 { magnitude } else { -magnitude }
            }
            // Infinity (mantissa 0) or NaN (mantissa non-zero).
            0x1F => f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13)),
            // Normal: rebias the exponent from 15 to 127, widen the mantissa.
            _ => f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)),
        }
```

`f16` has a different exponent width, so we cannot just shift. There are three cases:

- **Normal numbers** (the `_` arm): the value is 2^(e−15) × 1.m. In `f32` the same value has stored exponent e − 15 + 127. The 10 mantissa bits move to the top of the 23-bit `f32` mantissa (`<< 13`).
- **Subnormals** (exponent 0): the value is m × 2⁻²⁴ with no implicit 1. `f32` has enough range to represent these as *normal* numbers, so the simplest exact route is arithmetic: convert the 10-bit integer to `f32` (exact) and multiply by 2⁻²⁴ (exact, because multiplying by a power of two only changes the exponent). `f32::from_bits(0x3380_0000)` is 2⁻²⁴ written as bits: stored exponent 103 = 127 − 24, mantissa zero.
- **Infinity and NaN** (exponent all ones): map to the `f32` versions, keeping the NaN payload bits.

### 4.6 `f32` to `f16`

The reverse direction has four cases. The interesting ones:

<!-- file: src/lib.rs -->
```rust
        let e = exponent - 127; // the real exponent
        if e > 15 {
            return Self(sign | 0x7C00); // too big for f16
        }
        if e >= -14 {
            // A normal f16. Keep the top 10 of 23 mantissa bits and round
            // on the 13 we drop. If rounding carries out of the mantissa it
            // correctly bumps the exponent (and 65520 and up become infinity).
            let mut half = (((e + 15) as u32) << 10) | (mantissa >> 13);
            let dropped = mantissa & 0x1FFF;
            if round_up(dropped, 0x1000, half) {
                half += 1;
            }
            return Self(sign | half as u16);
        }
```

- If the real exponent is above 15, the value is at least 2¹⁶ = 65,536, beyond `f16`'s range: infinity.
- For normal `f16` results, build exponent and mantissa, then round on the 13 dropped bits. `0x1000` is the value of exactly half a unit in the last kept place.
- A subtle case the tests check: 65,519 rounds down to 65,504 (the max), but 65,520 is exactly halfway between 65,504 and the next step (65,536, which does not exist), and ties-to-even carries it into the exponent, giving infinity. That is the IEEE-correct answer.

<!-- file: src/lib.rs -->
```rust
        if e < -25 {
            return Self(sign); // below half the smallest subnormal: zero
        }
        // Subnormal f16: count units of 2^-24. Put the implicit leading 1
        // back, then shift right until one unit is worth 2^-24.
        let significand = mantissa | 0x0080_0000;
        let shift = (-e - 1) as u32; // between 14 and 24
        let mut half = significand >> shift;
        let dropped = significand & ((1 << shift) - 1);
        if round_up(dropped, 1 << (shift - 1), half) {
            half += 1;
        }
        Self(sign | half as u16)
```

For tiny values we compute how many units of 2⁻²⁴ (the smallest `f16` step) the value is worth. The `f32` value is `significand × 2^(e − 23)` where `significand` includes the implicit 1 (`| 0x0080_0000`). Dividing by 2⁻²⁴ gives `significand × 2^(e + 1)`, which is a right shift by `−e − 1`. The same `round_up` helper rounds on the bits shifted out. If rounding carries into bit 10, the result becomes the smallest *normal* `f16`, which is again correct.

<!-- file: src/lib.rs -->
```rust
fn round_up(dropped: u32, halfway: u32, kept: u32) -> bool {
    dropped > halfway || (dropped == halfway && kept & 1 == 1)
}
```

One definition of the rounding rule, shared by both paths, so the rule is written (and can be wrong) in only one place.

### 4.7 `fp8`: when there are few enough values, search

<!-- file: src/lib.rs -->
```rust
        let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
        let magnitude = x.abs().min(Self::MAX);
        // Positive patterns 0x00..=0x7E are sorted by value, so find the
        // first one that is >= magnitude and compare it with the one below.
        let upper = (0u8..=0x7E)
            .find(|&b| Self(b).to_f32() >= magnitude)
            .expect("magnitude is clamped to MAX, which is 0x7E");
```

With only 127 positive values, we skip the bit tricks: scan the sorted positive values for the first one at or above the input, then pick the nearer of it and the one below (ties to the even pattern). This is slow compared to bit manipulation, and it does not matter: converting weights to fp8 happens once, offline. Formats with 16 or fewer values (4-bit formats like NF4, chapter 19) are *always* implemented as lookup tables, even on GPUs.

`x.abs().min(Self::MAX)` **saturates**: values beyond ±448 become ±448. E4M3 has no infinity, and silently producing NaN from a large activation would be worse than clipping it.

### 4.8 Four ways to add numbers

<!-- file: src/lib.rs -->
```rust
pub fn sum_bf16_accumulator(xs: &[f32]) -> f32 {
    let mut total = Bf16::ZERO;
    for &x in xs {
        total = Bf16::from_f32(total.to_f32() + x);
    }
    total.to_f32()
}
```

This simulates hardware or a kernel that keeps its running total in `bf16`: after each addition, the total is rounded back to `bf16`. It is here to demonstrate the failure.

<!-- file: src/lib.rs -->
```rust
pub fn sum_kahan(xs: &[f32]) -> f32 {
    let mut total = 0.0f32;
    let mut compensation = 0.0f32;
    for &x in xs {
        let y = x - compensation;
        let t = total + y;
        compensation = (t - total) - y;
        total = t;
    }
    total
}
```

Kahan's trick, line by line: `t = total + y` is the rounded new total. `(t - total)` is what *actually* got added after rounding, and subtracting `y` (what we *wanted* to add) leaves the rounding error, negated. That error is subtracted from the next input, so it is not lost. Four floating-point operations per element instead of one, so it is rarely used in hot loops, but it is a good tool for reference computations.

<!-- file: src/lib.rs -->
```rust
pub fn sum_pairwise(xs: &[f32]) -> f32 {
    if xs.len() <= 8 {
        return sum_f32(xs);
    }
    let (left, right) = xs.split_at(xs.len() / 2);
    sum_pairwise(left) + sum_pairwise(right)
}
```

`split_at` divides a slice into two borrowed halves without copying. The recursion depth is log₂(n), about 21 levels for ten million elements, so the stack is not a concern.

### 4.9 The tests: exhaustive where possible

With 16-bit formats there are only 65,536 bit patterns, so we test all of them:

<!-- file: src/lib.rs -->
```rust
    #[test]
    fn bf16_round_trips_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let b = Bf16::from_bits(bits);
            let back = Bf16::from_f32(b.to_f32());
            if b.to_f32().is_nan() {
                assert!(back.to_f32().is_nan());
            } else {
                assert_eq!(back, b, "bits {bits:#06x}");
            }
        }
    }
```

Converting any `bf16` to `f32` and back must return the same bits (NaNs just have to stay NaN). This takes milliseconds and covers every case, including subnormals, both zeros and infinities. Whenever your input space is small enough to enumerate, enumerate it. It beats any amount of cleverness in choosing test cases.

The other direction (`f32` to 16 bits) has 4 billion inputs, so the tests take a sample of about a million bit patterns spread across the whole range and compare against a **brute-force reference**. The reference builds a sorted table of every positive value the format can represent, binary-searches for the two neighbours of the input, and picks the nearer one, ties to the even bit pattern. It is slow and obviously correct; the real implementation is fast and clever. Checking the clever one against the obvious one is a pattern you will use again and again in this course.

The tests module is marked `#[expect(clippy::float_cmp, reason = "...")]`. Clippy normally warns about comparing floats with `==`, because (section 3.9) floating-point results depend on operation order. Here the conversions are *supposed* to be exact, so exact comparison is the correct test, and the attribute records that decision.

Finally, the conversions were cross-checked against PyTorch: 401,006 values (normal, tiny, huge and boundary cases) converted with `tensor.to(torch.bfloat16)` and `tensor.to(torch.float16)` gave **zero mismatches** in bit patterns. This check is not part of `cargo test` because it needs Python, but it means the weights our engine reads in chapter 16 are interpreted exactly as PyTorch interprets them.

## 5. Run it

```bash
cargo test -p ch02-numbers
cargo run --release -p ch02-numbers
```

Output on the reference machine (parts 1 and 6 are exact; the timings in part 7 vary run to run):

```text
== 1. bit layouts (sign | exponent | mantissa)
            1e0  f32 0|01111111|00000000000000000000000
                 bf16 0|01111111|0000000         = 1e0
                 f16  0|01111|0000000000         = 1e0
           1e-1  f32 0|01111011|10011001100110011001101
                 bf16 0|01111011|1001101         = 1.00097656e-1
                 f16  0|01011|1001100110         = 9.9975586e-2
...
       6.5504e4  f32 0|10001110|11111111110000000000000
                 bf16 0|10001111|0000000         = 6.5536e4
                 f16  0|11110|1111111111         = 6.5504e4
           1e-6  f32 0|01101011|00001100011011110111101
                 bf16 0|01101011|0000110         = 9.983778e-7
                 f16  0|00000|0000010001         = 1.013279e-6

== 2. rounding 1,000,000 weights in [-0.1, 0.1)
   format          | mean rel. error | max rel. error
   bf16 (truncate) |         2.83e-3 |        7.75e-3
   bf16 (round)    |         1.42e-3 |        3.89e-3
   f16             |         1.78e-4 |        5.26e-2
   fp8 e4m3        |         4.33e-2 |         1.00e0

== 3. range: squaring an activation of 300
   f32:  90000
   f16:  inf   (largest f16 is 65504)
   bf16: 90112

== 4. accumulation
   1000 x 1.0 : f32 total 1000, bf16 total 256
   10,000,000 values in [0, 1): exact sum (f64) = 4999808.3
   plain f32 loop         4999772.0  relative error 7.3e-6
   pairwise f32           4999808.5  relative error 4.5e-8
   Kahan f32              4999808.5  relative error 4.5e-8
   bf16 accumulator           256.0  relative error 1.0e0

== 5. order of addition
   (1e8 + 1) - 1e8 = 0
   (1e8 - 1e8) + 1 = 1
   same 1,000,000 numbers, forward -465.74414, backward -465.74377

== 7. converting 16M values to f32
   bf16 -> f32: 10.22ms  (1.6 G values/s)
   f16  -> f32: 20.80ms  (0.8 G values/s)
```

Things to notice:

- **0.1 is not 0.1** in any format. `bf16` stores 0.100097656, `f16` stores 0.099975586.
- **65,504 in `bf16` becomes 65,536.** `bf16` has only 8 significant bits, and 65,504 needs 11.
- **`1e-6` in `f16` is a subnormal** (exponent field all zeros) and comes back as 1.013 × 10⁻⁶, a 1.3% error. `f16`'s maximum relative error in part 2 (5.26 × 10⁻²) comes from weights this small. `bf16` does not have this problem because its exponent range is huge.
- **`f16` is 8x more precise than `bf16`** on average (1.78 × 10⁻⁴ versus 1.42 × 10⁻³), exactly as 3 extra mantissa bits predict.
- **Squaring 300 overflows `f16`** but is fine in `bf16` (90,112 is the nearest `bf16` to 90,000).
- **A `bf16` accumulator is useless**, whatever the input format. An `f32` accumulator is fine for most purposes; pairwise and Kahan are better.
- **`bf16` → `f32` is twice as fast as `f16` → `f32`** here: a shift versus a branchy function. Both are limited by memory traffic at this size (reading 32 MB, writing 64 MB). Some CPUs have a dedicated `f16` conversion instruction (F16C on x86), and GPUs convert both for free.

## 6. The Rust behind it

**`to_bits`/`from_bits` versus `as`.** `as` between a float and an integer converts the *value* (rounding towards zero and saturating: `300.7f32 as u8` is 255). `to_bits`/`from_bits` reinterpret the *bits*. Mixing them up is a classic bug when implementing number formats: `(x as u32) >> 16` compiles fine and produces garbage.

**Newtypes cost nothing and prevent mix-ups.** `Bf16(u16)`, `F16(u16)` and `Fp8E4M3(u8)` are distinct types to the compiler and plain integers to the machine. A model loader that returns `&[Bf16]` cannot be accidentally fed to a function expecting `&[F16]`.

**`#[repr(transparent)]` enables zero-copy views.** Weight files contain raw little-endian bytes. Because `Bf16` is guaranteed to be laid out exactly like `u16`, chapter 9 can turn a `&[u8]` from a memory-mapped file into a `&[Bf16]` with no copying (after checking alignment).

**`From` for lossless conversions, `as` when you mean truncation.** `u32::from(u16)` compiles only because it can never lose data. Prefer it, and keep `as` for the places where you deliberately drop bits.

**Rust has no stable `f16` type yet.** A primitive `f16` exists on nightly Rust. In stable Rust, production code uses the [`half`](https://crates.io/crates/half) crate, which provides `bf16` and `f16` types very similar to ours (with hardware-accelerated conversions where available). We wrote our own so that nothing is hidden. From here on, the course uses these types from this crate.

**Debug builds check integer overflow.** `bits + 0x7FFF + lowest_kept_bit` would panic in a debug build if it could overflow. Running the exhaustive tests in debug mode checks our overflow reasoning for free. In release builds overflow wraps silently, so the debug-mode test run is the one that matters for this.

## 7. Mistakes you will make

- **Truncating instead of rounding** when converting to `bf16`. The model still works, just a little worse, and nobody notices until a careful comparison against the reference shows a small but consistent drift.
- **Accumulating in the storage format.** Any hand-written kernel that keeps its running sum in `bf16` or `f16` will give wrong answers for long vectors.
- **Assuming a model that works in `bf16` works in `f16`.** Activation overflow produces infinities and then NaNs.
- **Comparing floats with `==` in tests** of anything that involves sums. Use a tolerance based on the magnitude of the values.
- **Forgetting that `-0.0 == 0.0` is true** but their bit patterns differ. If you hash floats (for caching), hash the bits deliberately and decide what to do about −0 and NaN.
- **Reading `bf16` bytes in the wrong byte order.** Weight files are little-endian. On the machines you will use this matches the CPU, but code that reads bytes should say so explicitly (chapter 9 does).

## 8. How the professionals do it

- **PyTorch, JAX, and every GPU library** round to nearest even when converting to `bf16` and `f16`, as we do. Many older CPU code paths truncated `bf16`; mismatches between "reference" and "production" numerics sometimes come from that.
- **Tensor cores and matrix engines** (NVIDIA, AMD, Intel AMX, Apple AMX) take `bf16`/`f16`/`fp8` inputs and accumulate in `f32`. The CPU on the reference machine has AMX and AVX-512 BF16 instructions that do exactly this.
- **FP8 inference** (vLLM, TensorRT-LLM, SGLang on H100 and later) stores weights in E4M3 with a scale per tensor or per channel, and often keeps a few sensitive layers in `bf16`.
- **llama.cpp and GGUF files** mostly store weights in integer block formats (chapters 18-19), with `f16` scales.
- **Newer formats** keep pushing down: MXFP8, MXFP4 and NVFP4 are 8- and 4-bit floats that share one scale per small block of values. The ideas are the ones in this chapter plus the block scaling of chapter 19.

## 9. Exercises

1. **E5M2.** Implement `Fp8E5M2` (1 sign, 5 exponent, 2 mantissa bits, with infinity and NaN like `f16`). Hint: its bit patterns are exactly the top byte of an `f16`. Write an exhaustive round-trip test.
2. **Exact integers.** What is the largest N such that every integer from 0 to N is exactly representable in `bf16`? In `f16`? In `f32`? Write a test that finds each by search.
3. **Where does the error come from?** Change the part 4 experiment so the *inputs* are rounded to `bf16` but the accumulator stays `f32`. How big is the error now, compared with the `f32` inputs? What does that tell you about where precision matters?
4. **The smallest non-zero `f16`.** Find the smallest positive `f32` that converts to a non-zero `f16`. Explain the exact boundary.
5. **Scaled FP8.** Before converting the part 2 weights to E4M3, multiply them by `448 / max|w|`, and divide by it after converting back. Measure mean and max relative error, and count how many weights became zero, before and after scaling.
6. **Load-time cost.** SmolLM2-135M has about 135 million weights in `bf16`. Using the conversion speed you measured in part 7, how long would it take to convert all of them to `f32` at load time, and how much extra memory would the `f32` copy need? Is it worth it?

## 10. Check yourself

1. Which field of a float sets its range, and which sets its precision?
2. Why is converting `bf16` to `f32` exact and nearly free, while `f16` to `f32` needs branches?
3. Why do machine learning systems usually prefer `bf16` over `f16`, even though `f16` is more precise?
4. What does "round half to even" do with an exact tie, and why is that better than always rounding ties up?
5. Why does adding 1.0 to 256.0 in `bf16` give 256.0?
6. You parallelize a sum over 4 threads and the answer changes in the 7th digit. Is that a bug?
7. Why does an FP8 format need a scale factor, and what happens without one?

## 11. Recap

- A float is sign, exponent (range) and mantissa (precision). Neighbouring values are a fixed *fraction* apart, not a fixed distance.
- `bf16` = top 16 bits of `f32`: full range, 2-3 digits of precision, trivially convertible. The standard format for LLM weights.
- `f16` has 8x more precision but overflows at 65,504. `fp8` has 256 values and needs scale factors.
- Round to nearest, ties to even. Truncation doubles the error and biases it.
- Store small, accumulate in `f32`. A low-precision accumulator silently destroys long sums.
- Float addition is not associative: reordering changes results slightly, so numeric tests need tolerances.
- In Rust: `to_bits`/`from_bits` to reinterpret, newtypes with `#[repr(transparent)]` for zero-cost, type-safe formats, and exhaustive tests whenever the input space allows.

## Answers

**Exercises**

1. Since E5M2 is the top byte of an `f16`, convert `f32` → `F16` with `F16::from_f32`, then round the `f16` bits to their top 8 bits with the same trick as `bf16` (add `0x7F` plus the lowest kept bit, shift right by 8), handling NaN first. Beware double rounding: rounding `f32` → `f16` → E5M2 can differ from rounding `f32` → E5M2 directly in rare tie cases. A fully correct version rounds once, from the `f32` bits. Going back is `F16::from_bits(u16::from(b) << 8).to_f32()`.
2. `bf16`: 256 (257 needs 9 significant bits; `bf16` has 8). `f16`: 2,048 (11 significant bits). `f32`: 16,777,216 = 2²⁴. In general 2^(mantissa bits + 1). This matters in practice: never store token IDs or positions in a 16-bit float.
3. Measured on the reference machine: with `bf16`-rounded inputs, the exact sum of the rounded values differs from the original exact sum by only about 1 part in 5 million (4,999,809.3 versus 4,999,808.3), because individual rounding errors are unbiased and mostly cancel. Summing the rounded inputs with Kahan gives relative error 2.5 × 10⁻⁷, while a plain `f32` loop gives 7.5 × 10⁻⁵. Rounding the *inputs* costs little. The *accumulation* is where precision is lost.
4. Any value just above 2⁻²⁵ ≈ 2.98 × 10⁻⁸. The smallest `f16` subnormal is 2⁻²⁴, so 2⁻²⁵ is exactly halfway between 0 and it. The tie goes to the even pattern, which is 0. The next `f32` above 2⁻²⁵ (bits `0x3300_0001`) rounds up to 2⁻²⁴ (bits `0x0001`). The test `f16_rounding_matches_brute_force` covers this region.
5. Measured: without scaling, 9,982 of 1,000,000 weights become zero and the mean relative error is 4.33 × 10⁻². With scaling, no weight becomes zero and the mean relative error drops to 2.24 × 10⁻². The maximum relative error is still large (0.93) for the tiniest weights, because a weight of 0.0000001 gets rounded to the nearest of 256 levels spread over [−0.1, 0.1]. Relative error on near-zero weights is a misleading metric: what matters is the error relative to the tensor's scale, which chapter 18 uses.
6. At 1.6 billion values per second, 135 million weights take about 85 ms, which is acceptable at startup. But the `f32` copy needs 540 MB instead of 270 MB, and every decode step then reads twice the bytes, which (chapter 1) roughly halves decode speed. Chapter 16 keeps the weights in `bf16` and converts in registers inside the dot product instead.

**Check yourself**

1. The exponent sets the range; the mantissa sets the precision.
2. `bf16` has the same exponent layout as `f32`, so the conversion is just a shift. `f16` has a 5-bit exponent with a different bias, so it needs rebiasing and special handling of subnormals, infinity and NaN.
3. Range matters more than precision for neural networks: an occasional large activation that overflows `f16` produces infinity and then NaN, while `bf16`'s imprecision is absorbed by the network's statistical tolerance. And `bf16` converts to and from `f32` trivially.
4. It picks the neighbour whose last bit is 0. Always rounding ties up would bias every tie in the same direction, and over millions of values that bias accumulates.
5. `bf16` has 8 significant bits, so near 256 the representable values are 254, 256, 258. 257 is a tie between 256 and 258, and ties go to the even pattern, 256.
6. Almost certainly not. Four partial sums added at the end is a different order of additions, and floating-point addition is not associative. It is only a bug if the difference is far larger than the expected rounding error.
7. `fp8` has so few values that small numbers underflow to zero and large ones saturate. A scale factor stretches the tensor's values to fill the format's range. Without it, the demo turned about 1% of typical weights into zeros.

## Further reading

- David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", 1991. The classic, still accurate.
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", 2019.
- Micikevicius et al., "FP8 Formats for Deep Learning", 2022. Defines E4M3 and E5M2.
- Next: [Chapter 3: Tensors, strides and views](../03-tensors/README.md). How a pile of numbers becomes a matrix, and how to reshape it without copying a byte.
