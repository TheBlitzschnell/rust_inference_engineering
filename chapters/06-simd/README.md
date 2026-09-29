# Chapter 6: SIMD

> **In one sentence:** modern CPUs can apply one instruction to 4, 8 or 16 numbers at once, and using those instructions deliberately (with runtime detection so the program still runs everywhere) makes in-cache kernels several times faster.

**Where this fits:** chapters 1-5 relied on the compiler to vectorize loops, and it only did so when we restructured the code to let it. This chapter writes the vector instructions directly, for x86 (AVX2 and AVX-512) and for ARM (NEON), and adds the `bf16` kernel that the real model in chapter 16 runs on. It is also the first chapter with `unsafe` code, so it is where we set the rules for writing it.

**You need:** chapters 2 (floating point, `bf16`), 4 (roofline) and 5 (why dot products matter).

**You will build:** dot-product kernels for five instruction sets with runtime dispatch; an experiment showing how many independent accumulators the hardware needs; a `bf16 · f32` kernel that widens weights inside registers; a 64-byte-aligned buffer type; and measurements of when SIMD matters (data in cache) and when it does not (data in DRAM).

---

## 1. The intuition

Think of a cookie cutter. Cutting cookies one at a time with a single-cookie cutter takes a motion per cookie. A tray-sized cutter with 8 shapes cuts 8 cookies in one motion. Same dough, same shapes, one eighth of the motions.

SIMD (Single Instruction, Multiple Data) is the tray-sized cutter. A normal (scalar) instruction adds two numbers. A SIMD instruction adds two *vectors* of 8 numbers (AVX2) or 16 numbers (AVX-512), lane by lane, in about the same time.

**Where the analogy breaks:** the tray only helps if the dough is already laid out in a tray-shaped sheet. SIMD needs its inputs side by side in memory (contiguous), and every lane must do the same operation. Code that branches differently per element, or that gathers numbers from scattered addresses, gets little from SIMD. And if the dough arrives slowly (data coming from DRAM), a faster cutter does not help: you wait for dough. That is the roofline again.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **SIMD** | Single Instruction, Multiple Data: one instruction operating on a vector of values. |
| **Vector register** | A wide register: 128-bit (SSE, NEON), 256-bit (AVX2) or 512-bit (AVX-512). |
| **Lane** | One element position in a vector register. A 256-bit register holds 8 `f32` lanes. |
| **ISA** | Instruction set architecture: SSE2, AVX2, AVX-512, NEON... |
| **Intrinsic** | A function that maps directly to one CPU instruction, like `_mm256_fmadd_ps`. |
| **FMA** | Fused multiply-add: `a × b + c` in one instruction, with a single rounding. |
| **Latency** | Cycles from an instruction starting until its result can be used. |
| **Throughput** | How many such instructions can start per cycle. |
| **Horizontal sum** | Adding the lanes of one vector together. Needed at the end of a dot product. |
| **Tail** | The leftover elements when the length is not a multiple of the vector width. |
| **Target feature** | A CPU capability the compiler may assume (`avx2`, `fma`, `avx512f`, `neon`). |
| **Runtime dispatch** | Detecting the CPU's features when the program runs, then calling the best kernel. |
| **Alignment** | Whether an address is a multiple of some power of two (here, 64 bytes). |

## 3. The concepts in depth

### 3.1 The vector instruction sets you will meet

| ISA | Width | f32 lanes | Where |
|---|---|---|---|
| SSE2 | 128 bit | 4 | Every x86-64 CPU. This is all Rust assumes by default. |
| AVX2 + FMA | 256 bit | 8 | Intel since 2013 (Haswell), AMD since 2015. Nearly every x86 server today. |
| AVX-512 | 512 bit | 16 | Intel Xeon (incl. the reference machine), AMD Zen 4 and later. |
| NEON | 128 bit | 4 | Every 64-bit ARM CPU: Apple M-series, AWS Graviton, phones. |
| SVE / SVE2 | 128-2048 bit | varies | Newer ARM servers (Graviton 3/4, Grace). |

Beyond these, many chips now have **matrix units**: Intel AMX, Apple's AMX/SME, ARM SME. They multiply small tiles of matrices in one instruction. They are what the fastest CPU GEMM libraries use, and they are out of scope here.

### 3.2 Why the compiler did not vectorize our first dot product

```rust
a.iter().zip(b).map(|(x, y)| x * y).sum()
```

To vectorize this, the compiler would have to compute eight partial sums (lanes 0, 8, 16... in one, lanes 1, 9, 17... in another) and add them at the end. That changes the order of the additions. Chapter 2 showed that floating-point addition is not associative, so a different order gives a (slightly) different answer. Rust, like C without `-ffast-math`, is not allowed to change your program's answer, so it keeps the one running sum, and each addition waits for the previous one.

When we write eight running sums ourselves (chapter 1's `dot`), we have *chosen* the new order. The compiler can then map the eight sums onto a vector register. This is the single most important trick for fast reductions, and it works in any language.

### 3.3 Latency, throughput, and the number of accumulators

An FMA instruction on this CPU has a **latency** of about 4 cycles (you wait 4 cycles for the result) and a **throughput** of 2 per cycle (two can start every cycle). To keep the FMA units busy, you need about 4 × 2 = 8 independent FMAs in flight at any moment. With one accumulator, each FMA depends on the previous one, so you get one FMA every 4 cycles: one-eighth of the peak.

Part 3 of the demo measures exactly this with the AVX2 kernel and *k* independent 8-wide accumulators:

```text
   k =  1:   14.4 GFLOP/s
   k =  2:   27.8 GFLOP/s
   k =  4:   51.5 GFLOP/s
   k =  8:   50.7 GFLOP/s
   k = 16:   44.2 GFLOP/s
```

Doubling from 1 to 2 to 4 doubles the speed: the latency was the limit. From 4 up, it stops improving, before the 8 that FMA latency alone would suggest. The reason is loads: a dot product needs two vector loads per FMA (one from each input), and the core can only do two or three loads per cycle. So the dot product is **load-bound** at about one FMA per cycle, which 4 accumulators already achieve. At 16 accumulators the speed drops slightly, because 16 accumulators plus the loaded values no longer fit comfortably in the 16 AVX2 registers.

That is why every kernel in this chapter uses 4 accumulators.

### 3.4 FMA: one instruction, one rounding

`_mm256_fmadd_ps(a, b, c)` computes `a × b + c` for 8 lanes in one instruction. It counts as 2 FLOPs per lane, so it doubles the arithmetic rate compared with a separate multiply and add. It also rounds only once (the multiply is exact inside the instruction), so it is slightly *more* accurate.

Rust never turns `a * b + c` into an FMA on its own, because that would change the rounding. You get FMA only by asking: through intrinsics, or `f32::mul_add` (which becomes a single instruction only when the `fma` target feature is enabled, and a slow library call otherwise).

### 3.5 Compile-time features versus runtime detection

The compiler generates code for a **target**. For `x86_64-unknown-linux-gnu` the default target only guarantees SSE2, because that is what every x86-64 CPU has. There are two ways to use more:

1. **Compile for a specific CPU**: `RUSTFLAGS="-C target-cpu=native"`. The whole program may use every instruction the build machine has. Simple and fast, but the binary crashes with "illegal instruction" on a CPU without those features. Fine for code you build and run on the same machine; wrong for anything you distribute.
2. **Runtime dispatch**: compile several versions of the hot kernels, each marked with the features it needs, and pick one at startup after asking the CPU what it supports. The rest of the program stays portable. This is what production libraries do, and what this chapter does.

Rust supports option 2 directly:

- `#[target_feature(enable = "avx2,fma")]` on a function lets the compiler use those instructions inside it.
- `is_x86_feature_detected!("avx2")` asks the CPU at runtime (via the `CPUID` instruction; the result is cached).
- `#[cfg(target_arch = "x86_64")]` includes code only when compiling for that architecture, so the x86 kernels simply do not exist in an ARM build, and vice versa.

### 3.6 Alignment: when a vector load crosses a cache line

Part 2 of the demo times the same kernels on data starting 0, 16 or 32 bytes past a 64-byte boundary:

```text
   offset  0 bytes:  Avx2Fma  50.3 GFLOP/s  Avx512  70.9 GFLOP/s
   offset 16 bytes:  Avx2Fma  37.1 GFLOP/s  Avx512  39.4 GFLOP/s
   offset 32 bytes:  Avx2Fma  51.4 GFLOP/s  Avx512  39.4 GFLOP/s
```

A 64-byte AVX-512 load from an address that is not a multiple of 64 spans *two* cache lines, and the core has to fetch and merge both. When every load splits, AVX-512 falls from 71 to 39 GFLOP/s, slower than AVX2. A 32-byte AVX2 load splits only when it straddles a line boundary: never at offset 32, every other load at offset 16.

This bit us while writing the chapter. The first version of the demo used ordinary `Vec<f32>` buffers, and AVX-512 came out *slower* than AVX2 (28 against 52 GFLOP/s). `Vec<f32>` only guarantees 4-byte alignment, and the allocator happened to return addresses 32 and 48 bytes past a line boundary. The fix is a buffer type that guarantees 64-byte alignment, `AlignedVec`, described in section 4.6.

For inference this matters in two places:

- **Weights you allocate yourself** (after quantizing, converting, or copying) should be 64-byte aligned.
- **Weights you memory-map from a file** (chapter 9) are aligned however the file lays them out. Safetensors pads its header to a multiple of 8 bytes, so tensors are only guaranteed 8-byte alignment. GGUF (llama.cpp's format) aligns tensors to 32 bytes by default. Whether to copy into aligned memory at load time is a real trade-off, and chapter 16 measures it.

### 3.7 `bf16` weights, `f32` arithmetic

Chapter 2 showed that a `bf16` value becomes an `f32` by shifting its 16 bits into the top half of a 32-bit word. That is cheap in SIMD too:

```text
load 8 × u16 (16 bytes)          → [w0 w1 w2 w3 w4 w5 w6 w7]           (128-bit)
zero-extend each to 32 bits      → [0000w0 0000w1 ... 0000w7]          (256-bit)
shift each lane left by 16       → [w0 0000 w1 0000 ... w7 0000]       = 8 f32 values
FMA with 8 f32 activations
```

Three extra instructions per 8 weights, and the weights never exist as `f32` in memory. For a model stored in `bf16`, this halves the bytes read per token compared with converting everything to `f32` at load time. Part 4 of the demo shows the effect:

```text
   8192x8192 from DRAM | f32 SIMD     |  25.62ms |            10.5 |     5.2
   8192x8192 from DRAM | bf16 SIMD    |  16.09ms |             8.3 |     8.3
```

Same matrix, same arithmetic, 37% less time, because half as many bytes come from DRAM.

### 3.8 SIMD and the roofline

Compare the three matrix sizes in part 4:

| Weights | Portable | SIMD | SIMD speed-up |
|---|---|---|---|
| 512 × 512 (in L2) | 30.2 µs | 11.8 µs | 2.6x |
| 4096 × 4096 (in L3) | 3.33 ms | 2.67 ms | 1.25x |
| 8192 × 8192 (from DRAM) | 31.6 ms | 25.6 ms | 1.23x |

In L2, SIMD more than doubles the speed: the data arrives fast enough that arithmetic was the limit. From DRAM, the portable loop was already waiting on memory most of the time, and SIMD helps by only 23%. This is the roofline from chapter 4: SIMD raises the flat roof, and does nothing for the slanted one. For LLM decode, whose weights live in DRAM, SIMD is necessary but not sufficient; reading fewer bytes (`bf16`, then 8 and 4 bits) and using more cores (chapter 7) matter more.

## 4. The code

The dispatcher and portable kernels are in [`src/lib.rs`](src/lib.rs), the aligned buffer in [`src/aligned.rs`](src/aligned.rs), and the demo in [`src/main.rs`](src/main.rs).

### 4.1 Which instruction sets exist, and which one to use

<!-- file: src/lib.rs -->
```rust
impl Isa {
    /// Every variant, in order of preference (best first).
    pub const ALL: [Isa; 4] = [Isa::Avx512, Isa::Avx2Fma, Isa::Neon, Isa::Portable];

    /// Can this CPU run kernels for this instruction set?
    pub fn is_available(self) -> bool {
        match self {
            Isa::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2Fma => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            // NEON is part of the baseline of every 64-bit ARM CPU.
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => true,
            // Kernels for another architecture are never available.
            _ => false,
        }
    }
}
```

The enum lists every instruction set the crate has kernels for, on every architecture. `#[cfg(target_arch = ...)]` on individual match arms means that on an ARM build the x86 arms disappear (and `is_x86_feature_detected!`, which only exists on x86, is never compiled), and the final `_` arm reports "not available". So the same enum and the same calling code work everywhere.

AVX2 and FMA are separate CPU features. Every AVX2 CPU in practice also has FMA, but we check both because the kernel uses both.

<!-- file: src/lib.rs -->
```rust
pub fn best_isa() -> Isa {
    static BEST: OnceLock<Isa> = OnceLock::new();
    *BEST.get_or_init(|| {
        Isa::ALL
            .into_iter()
            .find(|isa| isa.is_available())
            .unwrap_or(Isa::Portable)
    })
}
```

`OnceLock` is a thread-safe "compute once, then read forever" cell. The first call detects the best instruction set; every later call, from any thread, reads the cached answer. Detection costs a few hundred nanoseconds; after that, choosing a kernel is one load and a well-predicted branch.

### 4.2 Dispatch: the one place `unsafe` meets detection

<!-- file: src/lib.rs -->
```rust
    Some(match isa {
        // SAFETY (all three arms): `is_available` just confirmed that this
        // CPU supports the instructions the kernel was compiled with.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { x86::dot_avx512(a, b) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma => unsafe { x86::dot_avx2(a, b) },
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { arm::dot_neon(a, b) },
        _ => dot_unrolled(a, b),
    })
```

The SIMD kernels are ordinary safe functions with `#[target_feature]`. Calling one from code compiled *without* those features is `unsafe`, because if the CPU lacks the feature the program executes an instruction the CPU does not have. On x86 that is an "illegal instruction" crash at best; in general, running code built for features the CPU lacks is undefined behaviour. The `unsafe` block is the caller saying "I checked", and the `SAFETY` comment says where.

The rule this crate follows, and that you should follow for any `unsafe`: **every `unsafe` block has a `// SAFETY:` comment explaining why the requirements are met, and the check that justifies it is as close to the block as possible.**

### 4.3 The AVX2 kernel

<!-- file: src/lib.rs -->
```rust
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let (a32, a_rest) = a.as_chunks::<32>();
        let (b32, b_rest) = b.as_chunks::<32>();
        let mut acc = [_mm256_setzero_ps(); 4];
        for (ca, cb) in a32.iter().zip(b32) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: `ca` and `cb` each hold exactly 32 floats, so the 8
                // floats starting at 8*j (j < 4) are inside them. `loadu`
                // does not require any particular alignment.
                let (va, vb) = unsafe {
                    (
                        _mm256_loadu_ps(ca.as_ptr().add(8 * j)),
                        _mm256_loadu_ps(cb.as_ptr().add(8 * j)),
                    )
                };
                *acc_j = _mm256_fmadd_ps(va, vb, *acc_j);
            }
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let mut total = horizontal_sum(sum);
        for (x, y) in a_rest.iter().zip(b_rest) {
            total += x * y;
        }
        total
    }
```

Line by line:

- `#[target_feature(enable = "avx2,fma")]`: inside this function the compiler may emit AVX2 and FMA instructions. Since Rust 1.86 such a function can be a safe `fn`; calling it from non-AVX2 code needs `unsafe`, as above.
- `as_chunks::<32>()` splits each input into 32-float blocks (4 vectors of 8) and a tail of up to 31 floats. Working in fixed-size blocks means the loop body never needs bounds checks.
- `__m256` is the type of a 256-bit register holding 8 floats. `_mm256_setzero_ps()` makes a vector of zeros. Four of them are the four accumulators from section 3.3.
- `_mm256_loadu_ps(ptr)` loads 8 floats from `ptr`. The `u` means **unaligned**: it works at any address (the aligned variant `_mm256_load_ps` crashes if the address is not a multiple of 32). Loading through a raw pointer is `unsafe` because the compiler cannot check the 8 floats are in bounds; the `SAFETY` comment explains why they are.
- `_mm256_fmadd_ps(va, vb, acc)` computes `va × vb + acc` in each of the 8 lanes. Since Rust 1.87, arithmetic intrinsics like this are safe to call inside a function that has the right `target_feature`, so only the loads need `unsafe`.
- After the loop, the four accumulators are added into one, the 8 lanes of that one are added into a single number (`horizontal_sum`), and the tail is handled one element at a time.

<!-- file: src/lib.rs -->
```rust
    fn horizontal_sum(v: __m256) -> f32 {
        let low = _mm256_castps256_ps128(v); // lanes 0-3
        let high = _mm256_extractf128_ps::<1>(v); // lanes 4-7
        let four = _mm_add_ps(low, high); // 4 partial sums
        let two = _mm_add_ps(four, _mm_movehl_ps(four, four)); // lanes 0+2, 1+3
        let one = _mm_add_ss(two, _mm_shuffle_ps::<0b01>(two, two)); // lane 0 + lane 1
        _mm_cvtss_f32(one)
    }
```

Adding lanes *across* a vector is awkward in SIMD, because the instructions are designed to work lane-by-lane. The standard sequence halves the vector repeatedly: add the upper 128 bits to the lower 128, then the upper two lanes to the lower two, then lane 1 to lane 0. It runs once per dot product, not once per element, so its cost does not matter for long vectors. The `::<1>` and `::<0b01>` are const-generic immediates: those instructions take a constant encoded in the instruction itself, and Rust enforces that it is known at compile time.

### 4.4 The AVX-512 kernel

<!-- file: src/lib.rs -->
```rust
    #[target_feature(enable = "avx512f")]
    pub fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        let (a64, a_rest) = a.as_chunks::<64>();
        let (b64, b_rest) = b.as_chunks::<64>();
        let mut acc = [_mm512_setzero_ps(); 4];
```

The same structure with 16-float vectors (`__m512`) and 64-float blocks. AVX-512 also provides `_mm512_reduce_add_ps`, which does the horizontal sum in one call. AVX-512 intrinsics became available in stable Rust in version 1.89.

### 4.5 The `bf16` kernel

<!-- file: src/lib.rs -->
```rust
    #[target_feature(enable = "avx2,fma")]
    fn bf16x8_to_f32(bits: __m128i) -> __m256 {
        let widened = _mm256_cvtepu16_epi32(bits);
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(widened))
    }
```

- `_mm256_cvtepu16_epi32` takes 8 unsigned 16-bit integers from a 128-bit register and zero-extends each to 32 bits, filling a 256-bit register.
- `_mm256_slli_epi32::<16>` shifts every 32-bit lane left by 16 bits: the `bf16` bits move into the top half, zeros fill the bottom.
- `_mm256_castsi256_ps` reinterprets the integer register as 8 floats. It generates no instruction; it only changes the type. This is the vector version of chapter 2's `f32::from_bits(u32::from(bits) << 16)`.

<!-- file: src/lib.rs -->
```rust
                // SAFETY: `cw` holds 32 `Bf16` (64 bytes) and `Bf16` is
                // `repr(transparent)` over `u16`, so the 16 bytes at element
                // 8*j are in bounds; `cx` holds 32 floats. Unaligned loads.
                let (vw, vx) = unsafe {
                    (
                        _mm_loadu_si128(cw.as_ptr().add(8 * j).cast::<__m128i>()),
                        _mm256_loadu_ps(cx.as_ptr().add(8 * j)),
                    )
                };
                *acc_j = _mm256_fmadd_ps(bf16x8_to_f32(vw), vx, *acc_j);
```

Here chapter 2's `#[repr(transparent)]` pays off: because `Bf16` is guaranteed to be laid out exactly like `u16`, a pointer to 8 `Bf16` values can be read as 16 raw bytes. Without that attribute the cast would be unsound.

Clippy warns that casting a `*const Bf16` (2-byte alignment) to `*const __m128i` (16-byte alignment) looks dangerous. It would be, for an *aligned* load. `_mm_loadu_si128` is an unaligned load that happens to be declared with a `__m128i` pointer, so the function carries `#[expect(clippy::cast_ptr_alignment, reason = "...")]` with that explanation.

### 4.6 NEON for ARM

<!-- file: src/lib.rs -->
```rust
    #[target_feature(enable = "neon")]
    pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        let (a16, a_rest) = a.as_chunks::<16>();
        let (b16, b_rest) = b.as_chunks::<16>();
        let mut acc = [vdupq_n_f32(0.0); 4];
        for (ca, cb) in a16.iter().zip(b16) {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                // SAFETY: each chunk holds 16 floats; 4 at offset 4*j fit.
                let (va, vb) = unsafe {
                    (
                        vld1q_f32(ca.as_ptr().add(4 * j)),
                        vld1q_f32(cb.as_ptr().add(4 * j)),
                    )
                };
                *acc_j = vfmaq_f32(*acc_j, va, vb);
            }
        }
```

Identical shape with ARM names: `vld1q_f32` loads 4 floats, `vfmaq_f32(acc, a, b)` is the FMA (note the accumulator comes *first* in ARM's argument order), and `vaddvq_f32` adds the 4 lanes horizontally in one instruction. NEON is always present on 64-bit ARM, so detection is trivial. The `bf16` version uses `vmovl_u16` (widen 4 × u16 to 4 × u32) and `vshlq_n_u32::<16>` (shift), the same trick as on x86.

This code was not run on real ARM hardware for this course. It was compiled for `aarch64-unknown-linux-gnu` and the full test suite ran under the `qemu-aarch64` emulator on the reference machine, which executes real NEON instructions (slowly). Every test passed. On an Apple Silicon Mac, `cargo test -p ch06-simd` runs it natively.

### 4.7 A 64-byte-aligned buffer

<!-- file: src/aligned.rs -->
```rust
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
}
```

<!-- file: src/aligned.rs -->
```rust
    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        assert!(size_of::<T>() > 0, "zero-sized types are not supported");
        let layout = Self::layout(len);
        // SAFETY: `layout` has a non-zero size (at least one element of a
        // non-zero-sized type), which is what `alloc` requires.
        let raw = unsafe { alloc(layout) }.cast::<T>();
        let Some(ptr) = NonNull::new(raw) else {
            handle_alloc_error(layout)
        };
        for i in 0..len {
            // SAFETY: `i < len`, so the write stays inside the allocation.
            // `write` does not read or drop the uninitialized old contents.
            // If `f` panics, the buffer leaks, which is safe (T: Copy has
            // nothing to drop).
            unsafe { ptr.as_ptr().add(i).write(f(i)) };
        }
        Self { ptr, len }
    }
```

This is what `Vec` does inside, with one change: the `Layout` asks for 64-byte alignment. The pieces:

- `Layout::from_size_align(bytes, 64)` describes the memory we want. The global allocator must return an address that is a multiple of 64.
- `alloc` returns uninitialized memory, or null on failure. `NonNull::new` turns null into `None`, and `handle_alloc_error` aborts the program, the same behaviour as `Vec` when memory runs out.
- `ptr.add(i).write(value)` initializes element `i` without reading the garbage that was there. (`*ptr = value` would try to drop the old value first, which is wrong for uninitialized memory.)
- `T: Copy` keeps things simple: no element ever needs dropping, so `Drop` only has to free the block.

The remaining pieces (`Deref` to `[T]` via `slice::from_raw_parts`, `Drop` via `dealloc` with the same layout, and the `unsafe impl Send/Sync`) are in the file, each with its `SAFETY` comment. The `Send`/`Sync` impls are needed because the raw pointer inside `NonNull` is neither by default: the compiler cannot know that we own the memory exclusively, so we state it.

The tests for `AlignedVec` were run under **Miri** (`cargo +nightly miri test -p ch06-simd aligned`), an interpreter for Rust's intermediate representation that detects undefined behaviour: out-of-bounds accesses, use of uninitialized memory, use-after-free, invalid alignment and data races. Miri also ran the whole test suite with AVX2 enabled, which checked every raw-pointer load in the AVX2 kernels. Both runs passed. The AVX-512 kernels were checked by the ordinary tests only.

## 5. Run it

```bash
cargo test -p ch06-simd
cargo run --release -p ch06-simd
```

On the reference machine:

```text
== instruction sets on this machine
   Avx512: true
   Avx2Fma: true
   Neon: false
   Portable: true
   selected: Avx512

== 1. dot product of two 4096-float vectors (data in L1, 64-byte aligned)
   naive, 1 running sum        3.1 GFLOP/s
   portable, 2 sums            6.4 GFLOP/s
   portable, 4 sums           12.8 GFLOP/s
   portable, 8 sums           25.0 GFLOP/s
   portable, 16 sums          29.4 GFLOP/s
   Avx2Fma                    50.1 GFLOP/s
   Avx512                     70.9 GFLOP/s

== 2. alignment: start address modulo 64 bytes
   offset  0 bytes:  Avx2Fma  50.3 GFLOP/s  Avx512  70.9 GFLOP/s
   offset 16 bytes:  Avx2Fma  37.1 GFLOP/s  Avx512  39.4 GFLOP/s
   offset 32 bytes:  Avx2Fma  51.4 GFLOP/s  Avx512  39.4 GFLOP/s

== 3. AVX2 dot product with k independent 8-wide accumulators
   k =  1:   14.4 GFLOP/s
   k =  2:   27.8 GFLOP/s
   k =  4:   51.5 GFLOP/s
   k =  8:   50.7 GFLOP/s
   k = 16:   44.2 GFLOP/s

== 4. matrix-vector product y = W x (one core, aligned weights)
   weights             | kernel       |     time | GB/s of weights | GFLOP/s
    512x512  in L2     | f32 portable |  30.19µs |            34.7 |    17.4
    512x512  in L2     | f32 SIMD     |  11.76µs |            89.2 |    44.6
    512x512  in L2     | bf16 SIMD    |  14.16µs |            37.0 |    37.0
   4096x4096 in L3     | f32 portable |   3.33ms |            20.2 |    10.1
   4096x4096 in L3     | f32 SIMD     |   2.67ms |            25.1 |    12.6
   4096x4096 in L3     | bf16 SIMD    |   1.66ms |            20.2 |    20.2
   8192x8192 from DRAM | f32 portable |  31.60ms |             8.5 |     4.2
   8192x8192 from DRAM | f32 SIMD     |  25.62ms |            10.5 |     5.2
   8192x8192 from DRAM | bf16 SIMD    |  16.09ms |             8.3 |     8.3
```

What to take from it:

- **From one running sum to AVX-512 is 23x** (3.1 to 70.9 GFLOP/s) for data in L1. The biggest steps are "more independent sums" (latency) and "wider vectors with FMA" (throughput).
- **The portable version with 8 or 16 sums gets within 2x of hand-written AVX2.** For many kernels that is good enough, and it runs everywhere. Write intrinsics for the few kernels that dominate the profile.
- **Alignment can cost AVX-512 almost half its speed.** Allocate hot buffers aligned to 64 bytes.
- **Once data comes from DRAM, SIMD barely matters** (25.6 against 31.6 ms), while halving the bytes with `bf16` gives a much bigger gain (16.1 ms).
- **For `bf16` data already in L2, the conversion work shows**: 14.2 µs against 11.8 µs for `f32`. In cache, `bf16` saves no time because there is no memory bottleneck to relieve, and the widening instructions cost a little. In DRAM it wins clearly. Which effect dominates depends on where the data lives, which is why you measure at realistic sizes.

## 6. The Rust behind it

**`#[target_feature]` on safe functions (Rust 1.86+).** The function body may use the listed features; the caller must be in a context that has them, or use `unsafe` and promise the CPU does. This moves the dangerous part to one audited place (the dispatcher) and keeps the kernels themselves mostly safe code.

**Safe intrinsics (Rust 1.87+).** Inside a `#[target_feature]` function, intrinsics that only compute on registers (`_mm256_fmadd_ps`, `_mm256_add_ps`) are safe to call. Only those that dereference raw pointers (`_mm256_loadu_ps`) remain `unsafe`, so the `unsafe` blocks shrink to exactly the loads.

**`#[cfg(target_arch)]` removes code, it does not skip it.** The ARM module is not compiled at all on x86, so it cannot even be type-checked there by a normal build. That is why the course checks it separately with `cargo clippy --target aarch64-unknown-linux-gnu` and runs its tests under emulation.

**Explicit imports for intrinsics.** Clippy's pedantic `wildcard_imports` lint rejected `use std::arch::x86_64::*;`. The explicit list is long, but it tells a reader exactly which instructions the module uses.

**`std::simd` is not stable yet.** Rust has a portable SIMD API (`std::simd`, types like `f32x8`) that compiles to the right instructions on each target without intrinsics. As of Rust 1.94 it is nightly-only. On stable, crates such as `wide`, `pulp` and `simd-aligned` fill the gap. The approach in this chapter (portable fallback plus per-ISA intrinsics behind runtime dispatch) is what most production Rust inference code uses today.

**Miri and emulation are how you check `unsafe` code you cannot fully test.** Ordinary tests only show that the code gave the right answer this time; Miri checks that no step of it was undefined behaviour. `qemu-user` lets one machine run another architecture's code. Both are cheap to set up, and both were used for this chapter.

## 7. Mistakes you will make

- **Calling a `target_feature` kernel without checking the CPU.** On a machine without the feature the process dies with `SIGILL` (illegal instruction). With `-C target-cpu=native` builds the same thing happens to the *whole binary* when it is copied to an older machine.
- **Forgetting the tail.** A kernel that only handles multiples of 32 gives wrong answers for every other length. Test every length from 0 to a few hundred, like the tests here do.
- **Assuming alignment.** `_mm256_load_ps` (the aligned load) crashes on unaligned addresses. Use `loadu` unless you *guarantee* alignment, and even then it is usually just as fast.
- **Benchmarking only in cache.** A kernel that is 3x faster on a 4 KB vector may be 1.2x faster on the real weight matrix.
- **Too many accumulators.** Once they no longer fit in registers, the compiler spills them to the stack and the kernel slows down.
- **Denormal inputs.** Operations on subnormal floats (chapter 2) can be 10-100x slower on some CPUs. Inference code sometimes sets the "flush to zero" CPU mode to avoid this.

## 8. How the professionals do it

- **llama.cpp (ggml)** has hand-written kernels for AVX, AVX2, AVX-512 (several variants), NEON, SVE, WASM SIMD and more, chosen at compile time or at runtime depending on the build. Its quantized dot products (chapters 18-19) are where most of that code lives.
- **The `gemm` crate** (used by Hugging Face's `candle`) generates micro-kernels for each ISA and dispatches at runtime, exactly the structure of this chapter scaled up.
- **ONNX Runtime (MLAS)** and **oneDNN** go further and generate machine code at runtime (JIT) for the specific CPU and matrix shape.
- **Apple Accelerate** uses the undocumented AMX matrix unit on M-series chips; it is often several times faster than the best NEON code for matmul.
- **AVX-512 frequency effects:** some older Intel server CPUs (around 2017-2019) lowered their clock speed when running heavy AVX-512 code, so AVX2 could win overall. Recent CPUs, including the one on the reference machine, do this much less. The answer, as always, is to measure.

## 9. Exercises

1. **`axpy`.** Write `axpy(alpha, x, y)` (`y[i] += alpha × x[i]`) with AVX2 intrinsics, a NEON version and a portable fallback, behind the same kind of dispatcher. Attention (chapter 12) uses this operation to accumulate weighted value vectors.
2. **Break the tail.** Delete the tail loop in `dot_avx2` and run the tests. Which lengths fail? Why does the test suite loop over every length from 0 to 200?
3. **ARM, emulated.** Install `qemu-user` and `gcc-aarch64-linux-gnu` (on Debian or Ubuntu) and run `cargo test --target aarch64-unknown-linux-gnu -p ch06-simd`. Then add `println!("{:?}", best_isa())` to a test and run it with `-- --nocapture`. What does it print?
4. **In cache or not.** Explain, using section 3.8, why the `bf16` kernel is slower than the `f32` kernel for the 512 × 512 matrix and faster for the 8192 × 8192 one.
5. **A decode estimate.** SmolLM2-135M has about 270 MB of `bf16` weights. Using the `bf16 SIMD` DRAM bandwidth from part 4, how many tokens per second could one core produce at most? What does this suggest about the next chapter?
6. **Override switch.** Add an environment variable (say `INFER_ISA=avx2`) that `best_isa` honours if that instruction set is available. Why is such a switch useful in production?

## 10. Check yourself

1. Why can't the compiler vectorize `a.iter().zip(b).map(|(x, y)| x * y).sum()` by itself?
2. With FMA latency 4 cycles and throughput 2 per cycle, how many independent accumulators are needed to reach peak? Why did the dot product stop improving at 4?
3. What is the difference between `-C target-cpu=native` and runtime dispatch, and when is each appropriate?
4. Why does an unaligned 64-byte load cost more than an aligned one?
5. How does a `bf16` weight become an `f32` inside a SIMD register, and why is that useful?
6. Why does SIMD help a lot for data in L2 and little for data in DRAM?

## 11. Recap

- SIMD applies one instruction to 4-16 floats at once. The compiler vectorizes reductions only if you give it independent accumulators, because reordering float additions changes results.
- Enough independent chains are needed to hide instruction latency; beyond that, loads or registers become the limit. Four 8-wide accumulators are enough for a dot product here.
- Portable code compiles for the lowest common denominator. Use `#[target_feature]` kernels plus runtime detection to use newer instructions without breaking older CPUs.
- Every `unsafe` block gets a `SAFETY` comment. Check `unsafe` code with Miri and other architectures with emulation.
- Align hot buffers to 64 bytes; split cache-line loads can halve AVX-512 speed.
- `bf16` weights can be widened to `f32` inside registers for a few instructions, halving memory traffic.
- In DRAM-bound kernels, SIMD is necessary but not sufficient: bytes and bandwidth decide the speed.

## Answers

**Exercises**

1. The AVX2 core is `acc = _mm256_fmadd_ps(_mm256_set1_ps(alpha), load(x), load(y))` followed by a store back into `y` with `_mm256_storeu_ps`. `axpy` has no reduction, so no accumulators are needed: every lane is independent. The tail is a scalar loop. The dispatcher and SAFETY reasoning are the same as for `dot`.
2. Every length that is not a multiple of 32 fails (for AVX2), because the last `len % 32` products are never added. Looping over every length from 0 to 200 covers every possible tail length (0-31 for AVX2, 0-63 for AVX-512, 0-15 for NEON) several times, plus lengths shorter than one block. Kernel bugs hide in exactly those edges.
3. It prints `Neon`: `Isa::ALL` checks `Avx512` and `Avx2Fma` first, which are not available on aarch64 (their arms are compiled out), and `Neon` is always available there. On the reference machine under `qemu-aarch64`, the full test suite passed.
4. For 512 × 512 (1 MB in `f32`, 0.5 MB in `bf16`), all data is in L2 and arrives faster than the arithmetic can use it, so the extra widening instructions of the `bf16` kernel make it slower. For 8192 × 8192 the data comes from DRAM, the kernels wait for memory, and the `bf16` kernel reads half as many bytes.
5. About 8.3 GB/s ÷ 0.27 GB ≈ 31 tokens per second on one core, before any other work. Chapter 4 showed that all four cores together reach 30-45 GB/s, so using every core could lift this ceiling to roughly 110-165 tokens/s. That is chapter 7.
6. Read the variable inside the `get_or_init` closure, parse it into an `Isa`, and use it if `is_available()` returns true, falling back to detection otherwise. It lets you test every kernel on one machine, compare them in production, and work around a CPU where the "best" instruction set is slower (the AVX-512 downclocking of section 8) or buggy, without rebuilding.

**Check yourself**

1. Vectorizing requires several partial sums added at the end, which reorders the additions. Floating-point addition is not associative, so that could change the result, and Rust does not allow optimizations that change results.
2. Latency × throughput = 4 × 2 = 8 independent FMAs. The dot product needs two vector loads per FMA and the core can only do two or three loads per cycle, so loads cap it at about one FMA per cycle, which 4 accumulators already reach.
3. `target-cpu=native` compiles the whole program for the build machine's CPU: simplest, fastest, but the binary may crash on other CPUs. Runtime dispatch compiles several kernels and picks one at startup: portable binaries with fast kernels. Use native builds for code you run where you build; use dispatch for anything distributed.
4. It spans two cache lines, so the core must access both and combine the pieces, roughly doubling the cost of the load.
5. Zero-extend each 16-bit value to 32 bits and shift it left by 16; the bits are now a valid `f32`. It lets weights stay in `bf16` in memory (half the bytes) while the arithmetic happens in `f32`.
6. In L2, data arrives faster than scalar code can process it, so arithmetic is the bottleneck and wider instructions help. From DRAM, the processor waits for data whether or not it uses SIMD, so SIMD only shortens the part of the time that was not spent waiting.

## Further reading

- Intel Intrinsics Guide (online) and ARM's NEON Intrinsics Reference: what each intrinsic does, with latencies and throughputs.
- Agner Fog, "Instruction tables" and "Optimizing software in C++": instruction latencies and throughputs for every x86 CPU generation.
- The Rust `std::arch` documentation, and the `target_feature` chapter of the Rust Reference.
- Next: [Chapter 7: Threads](../07-threads/README.md). One core cannot use all the memory bandwidth. Four can, if the cost of coordinating them does not eat the gain.
