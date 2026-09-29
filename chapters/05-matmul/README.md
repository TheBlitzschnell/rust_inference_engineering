# Chapter 5: Matrix multiplication

> **In one sentence:** matrix multiplication is where inference spends most of its time, and the difference between a naive loop and a good one (same arithmetic, different order of memory accesses) is roughly 100x.

**Where this fits:** chapter 4 gave us the roofline. Now we apply it to the operation that dominates every model: each linear layer is a matrix multiplication. The kernels here are single-threaded and use only the vectorization the compiler finds on its own. Chapter 6 adds explicit SIMD, chapter 7 adds threads, and the engine from chapter 14 onward is built on the result.

**You need:** chapters 1-4.

**You will build:** seven matmul kernels (naive, loop-reordered, cache-blocked, dot-product NT, and register-tiled NT micro-kernels of three sizes), tested against an `f64` reference on awkward sizes, and timed on the shapes that decode, batched decode and prefill actually produce.

---

## 1. The intuition

Imagine filling in a big table of results, where each cell needs one row from a stack of cards on your left and one column from a stack of cards on your right, and you have to multiply them pair by pair and add up.

- **The naive way:** for each cell, walk to the left stack and pull out its row, then walk to the right stack and pull out its column, one card at a time from different places in the stack. You pull every row out again for every cell in that row of the table, and every column again for every cell in that column.
- **A better way:** keep one row of the left stack on your desk while you fill in everything that needs it.
- **Better still:** clear a desk-sized area, bring over a *block* of the right stack that fits on it, and do every piece of work that needs that block before putting it back.
- **Best:** work on a small patch of the table at once (say 2 rows × 4 columns of cells), so each card you pick up is used several times before you put it down.

All four methods do exactly the same multiplications. They differ only in how often you walk to the stacks. That walking is memory traffic, and chapter 4 showed it is the expensive part.

**Where the analogy breaks:** on a real desk, picking up a card is picking up a card. In a CPU, "the desk" comes in several sizes (registers, L1, L2, L3), each with its own capacity and speed, so a fast matmul arranges its work in nested blocks, one level per size. We use two levels here (registers and L2); professional libraries use all of them.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Matmul** | Matrix multiplication. `C = A · B` with `A` of shape m×k and `B` of shape k×n gives `C` of shape m×n. |
| **GEMM** | "General matrix multiply", the standard library name (BLAS `sgemm` for `f32`). |
| **GEMV** | Matrix-vector multiply: GEMM with m = 1. The shape of decode. |
| **NN / NT layout** | Whether the second operand is stored normally (k×n) or transposed (n×k). Linear layers are NT. |
| **Reduction dimension** | `k`: the dimension that is summed over. |
| **Loop order** | Which of the three loops (i, j, p) is outermost. Six orders, same result, different speed. |
| **Blocking / tiling** | Cutting a loop into chunks so that the data a chunk uses fits in a cache. |
| **Register tiling** | Computing a small block of outputs at once so each loaded value is used several times from registers. |
| **Micro-kernel** | The innermost, register-tiled piece of a fast matmul. |
| **Register spill** | When a loop needs more values than there are registers, and the compiler parks some in memory. |
| **Reuse** | How many times a value fetched from memory is used before it is evicted. |

## 3. The concepts in depth

### 3.1 Every linear layer is a matmul

Chapter 1's model was one linear layer computing `y = W x` for one input. With several inputs stacked as the rows of a matrix `X`, the layer computes

```text
Y [m × n]  =  X [m × k]  ·  Wᵀ,       W stored as [n × k]  (n outputs, k inputs)

y[i][j] = Σ_p  x[i][p] · w[j][p]        = dot(row i of X, row j of W)
```

`m` is the number of tokens being processed, `k` the layer's input size, `n` its output size. This is the **NT** form: the weight matrix is used transposed, but because PyTorch stores it as `[out_features, in_features]`, "transposed" just means "dot product of two contiguous rows". No actual transpose ever happens.

What changes between the phases of inference is `m`:

| Phase | m | Shape | Character |
|---|---|---|---|
| Decode, one user | 1 | GEMV | Memory-bound (intensity ~0.5 for `f32`) |
| Decode, batch of B users | B | skinny GEMM | Intensity grows with B |
| Prefill of a P-token prompt | P | GEMM | Compute-bound for P in the hundreds |

So an inference engine needs good kernels for *both* extremes: a GEMV that streams weights at full memory bandwidth, and a GEMM that keeps the arithmetic units busy.

### 3.2 Counting the work

The product has m × n outputs, each a sum of k products, so

```text
FLOPs = 2 · m · k · n          (one multiply and one add per term)
bytes = 4 · (m·k + k·n + m·n)  if every number moved exactly once (f32)
```

For a square N × N product, intensity = 2N³ / 12N² = N/6. A 1024 × 1024 matmul has intensity about 170, far above this machine's ridge point of 3-4. So it *should* be compute-bound, and a good kernel should approach the peak of about 31-34 GFLOP/s per core that chapter 4 measured. Whether it does depends entirely on whether every number really moves only once. In a naive kernel, it does not.

The numbers that are re-read are the key. Each element of `A` is used n times (once per output column) and each element of `B` is used m times. A fast matmul arranges the loops so that those repeated uses hit in cache or, better, in registers.

### 3.3 Naive: 1-2 GFLOP/s

```text
for i in 0..m:
  for j in 0..n:
    for p in 0..k:
      c[i][j] += a[i][p] * b[p][j]
```

Two separate problems:

1. **The inner loop walks down a column of `b`.** `b[p][j]` and `b[p+1][j]` are `n` floats apart. Each access fetches a 64-byte line and uses 4 bytes of it (chapter 4, section 3.4).
2. **The inner loop is one long dependency chain.** Every iteration adds into the same `sum`, so each addition waits for the previous one. The compiler may not reorder the additions, because floating-point addition is not associative (chapter 2).

Measured: about 2 GFLOP/s on 512 × 512, and 0.2 GFLOP/s once `b` is too large for the cache. That is under 1% of the core's peak.

### 3.4 Loop order i-k-j: about 10x faster

Swap the two inner loops:

```text
for i in 0..m:
  for p in 0..k:
    a_ip = a[i][p]
    for j in 0..n:
      c[i][j] += a_ip * b[p][j]      ← row p of b, row i of c: both contiguous
```

Now the innermost loop reads row `p` of `b` and updates row `i` of `c`, both contiguous. And each iteration updates a *different* `c[i][j]`, so there is no dependency chain: the compiler is free to process 4 (or 8, or 16) `j`s at once with vector instructions. Measured: 13 GFLOP/s on 512 × 512, six times faster than naive with not a single arithmetic operation changed.

The weakness shows up at 1024 × 1024 (9.8 GFLOP/s): for each row `i` of `a`, the inner two loops sweep through *all* of `b`. At 1024 × 1024, `b` is 4 MB, bigger than the 2 MB L2 cache, so every row of `a` drags all of `b` in from L3 again.

### 3.5 Cache blocking: keep a piece of `b` in L2

The fix is to change *when* each part of `b` is used. Cut `b` into blocks of `KC` rows × `NC` columns and finish all the work involving one block before moving to the next:

```text
for each block of b (rows p0..p1, columns j0..j1):      ← 256 × 512 floats = 512 KB
  for i in 0..m:                                          ← every row of a...
    for p in p0..p1:
      for j in j0..j1:                                    ← ...reuses the block from L2
        c[i][j] += a[i][p] * b[p][j]
```

With `KC = 256` and `NC = 512`, a block is 512 KB, which fits in L2 with room left for the rows of `a` and `c` passing through. All m rows of `a` now reuse that block from L2 instead of refetching it from L3. Measured at 1024 × 1024: 18 GFLOP/s against 9.8 for plain i-k-j.

Choosing block sizes is always about the same arithmetic: what is the working set of the inner loops, and which cache level must it fit in?

### 3.6 The NT form and weights-outer order

In inference we never have `b` in NN layout: the weights are stored as rows (`[n × k]`). The natural kernel is a dot product per output:

```text
for j in 0..n:                    ← each weight row, fetched once...
  for i in 0..m:                  ← ...is reused for every input row
    y[i][j] = dot(x[i], w[j])
```

The loop order matters here too. With weights outermost, each weight row (16 KB at k = 4,096) is fetched from memory once and then reused for all m inputs while it sits in L1/L2. With inputs outermost, every input would sweep all the weights. This is chapter 1's batching insight, now as a kernel design rule: **in decode, put the weights on the outside.**

`dot` uses eight running sums, so it vectorizes. Measured: the best of our kernels for the batch-16 shape (18 GFLOP/s) and for the prefill shape (21 GFLOP/s).

### 3.7 Register tiling: the micro-kernel

A dot product loads two numbers for every multiply-add. The CPU can do about one vector load per multiply-add and still keep up, but not much more, so a dot product can at best just keep pace, and in practice falls short. To go faster, each loaded number must be used several times while it is in a register.

The **micro-kernel** computes a small MR × NR block of outputs at once:

```text
for p in 0..k (8 at a time):
  load x[i0..i0+MR][p..p+8]         ← MR vectors
  for c in 0..NR:
    load w[j0+c][p..p+8]            ← 1 vector, used MR times
    for r in 0..MR:
      acc[r][c] += x[r] * w[c]      ← MR × NR vector multiply-adds
```

With MR = 2 and NR = 4: 6 vector loads feed 8 vector multiply-adds. With 4 × 4: 8 loads feed 16. Bigger tiles mean more reuse, *until you run out of registers*.

The accumulators must stay in registers or the whole point is lost. `acc` holds MR × NR × 8 floats. On the default x86-64 target (SSE2), there are 16 vector registers of 4 floats each, 64 floats in total:

| Tile | Accumulator floats | SSE2 registers needed | Fits? |
|---|---|---|---|
| 1 × 4 | 32 | 8 | Yes, with room for loads |
| 2 × 4 | 64 | 16 | Just: the compiler needs a few more for loads, so it spills a little |
| 4 × 4 | 128 | 32 | No: heavy spilling to memory |

The measurements agree. On 1024 × 1024, 2 × 4 reaches 19.8 GFLOP/s (58% of peak, the best result in this chapter) while 4 × 4 falls to 12.1, because its "registers" are really memory. Recompiled for this CPU's AVX-512 (32 registers of 16 floats), the 4 × 4 tile fits and its speed jumps from 12.3 to 17.1 GFLOP/s (exercise 5). **The right tile size depends on the register file of the machine you compile for.** That is why libraries select micro-kernels at runtime for the CPU they find (chapter 6 shows how).

### 3.8 Where the remaining 40% goes

Our best kernels reach about 55-60% of the single-core peak. Professional BLAS libraries (OpenBLAS, Intel MKL, BLIS, Apple Accelerate) reach 85-95%. The gap comes from:

- **No fused multiply-add.** `acc += x * w` compiles to a multiply and a separate add. FMA does both in one instruction (chapter 6).
- **Narrow vectors.** The default target uses 4-float SSE2 vectors. This CPU has 16-float AVX-512 vectors. Rust will not use them unless told the CPU has them (chapter 6).
- **No packing.** Libraries copy each block of `A` and `B` into a contiguous buffer laid out exactly in the order the micro-kernel reads it, so every load is sequential and no two blocks fight for the same cache sets.
- **Hand-written micro-kernels** in assembly or intrinsics, with software prefetching.
- **More blocking levels**: for registers, L1, L2 and L3, each sized for the machine. The standard structure is the "Goto algorithm" used by BLIS and OpenBLAS: five loops around a micro-kernel.

This course does not chase the last 40% of GEMM, for a reason: during decode, which dominates LLM serving on CPUs, matmul is memory-bound and the micro-kernel's arithmetic speed is not the limit. Chapters 6 and 7 get the decode path to the memory roofline, which is what matters.

## 4. The code

All kernels are in [`src/lib.rs`](src/lib.rs); [`src/main.rs`](src/main.rs) times them.

### 4.1 Naive

<!-- file: src/lib.rs -->
```rust
pub fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(a, b, c, m, k, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
```

All matrices are flat row-major slices (chapter 3) with the dimensions passed alongside. `check_shapes` asserts the three lengths up front, so a wrong call fails with a clear message instead of an out-of-bounds panic deep in the loop. Every index expression is `row * row_length + column`.

### 4.2 Loop order i-k-j

<!-- file: src/lib.rs -->
```rust
pub fn matmul_ikj(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(a, b, c, m, k, n);
    c.fill(0.0);
    for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (&a_ip, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_ip * b_pj;
            }
        }
    }
}
```

- `c.fill(0.0)` first, because this kernel *accumulates* into `c` (every `p` adds a contribution). Forgetting this is a classic bug: the kernel then adds to whatever was in `c` before.
- `a.chunks_exact(k).zip(c.chunks_exact_mut(n))` pairs row `i` of `a` with row `i` of `c`. There is no `i` variable at all.
- `a_row.iter().zip(b.chunks_exact(n))` pairs `a[i][p]` with row `p` of `b`, again with no index.
- The innermost loop zips row `i` of `c` with row `p` of `b`. Both have exactly `n` elements, the compiler can see that, and so it removes all bounds checks and vectorizes the loop.

Writing the loops with `chunks_exact` and `zip` instead of indices is not only style. With `c[i * n + j] += a[i * k + p] * b[p * n + j]`, the compiler must prove each index is in bounds or insert a check per element, and a check inside the inner loop can block vectorization. Iterators carry the proof with them.

### 4.3 Blocked

<!-- file: src/lib.rs -->
```rust
    for p0 in (0..k).step_by(KC) {
        let p1 = (p0 + KC).min(k);
        for j0 in (0..n).step_by(NC) {
            let j1 = (j0 + NC).min(n);
            // The block b[p0..p1][j0..j1] is now reused by every row of a.
            for i in 0..m {
                let c_part = &mut c[i * n + j0..i * n + j1];
                for p in p0..p1 {
                    let a_ip = a[i * k + p];
                    let b_part = &b[p * n + j0..p * n + j1];
                    for (c_ij, &b_pj) in c_part.iter_mut().zip(b_part) {
                        *c_ij += a_ip * b_pj;
                    }
                }
            }
        }
    }
```

The same i-k-j body, wrapped in two block loops. `step_by(KC)` visits block starts 0, 256, 512, ...; `(p0 + KC).min(k)` clips the last block when `k` is not a multiple of 256. The innermost loop again zips two slices of equal length, so it vectorizes exactly like the unblocked version. Blocking changed only which data is live at the same time.

### 4.4 NT with dot products

<!-- file: src/lib.rs -->
```rust
pub fn matmul_nt(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(x, w, y, m, k, n);
    for (j, w_row) in w.chunks_exact(k).enumerate() {
        for (i, x_row) in x.chunks_exact(k).enumerate() {
            y[i * n + j] = dot(x_row, w_row);
        }
    }
}
```

Weight rows outside, input rows inside. The write `y[i * n + j]` jumps between rows of `y`, which is fine: `y` is small (m × n) and the writes are few compared with the k multiply-adds behind each one.

<!-- file: src/lib.rs -->
```rust
pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32]) {
    matmul_nt(x, w, y, 1, x.len(), y.len());
}
```

Decode's matrix-vector product is simply the NT kernel with one input row.

### 4.5 The register-tiled micro-kernel

<!-- file: src/lib.rs -->
```rust
    let mut acc = [[[0.0f32; 8]; NR]; MR];
    let k8 = k - k % 8;
    for p in (0..k8).step_by(8) {
        let xs: [&[f32; 8]; MR] = std::array::from_fn(|r| {
            let start = (i0 + r) * k + p;
            x[start..start + 8].try_into().expect("8 floats")
        });
        for c in 0..NR {
            let start = (j0 + c) * k + p;
            let wv: &[f32; 8] = w[start..start + 8].try_into().expect("8 floats");
            for r in 0..MR {
                for lane in 0..8 {
                    acc[r][c][lane] += xs[r][lane] * wv[lane];
                }
            }
        }
    }
```

Line by line:

- `acc` is a three-dimensional array on the stack: for each of the MR × NR outputs, eight partial sums (one per vector lane). Because `MR` and `NR` are const generics, its size is known at compile time and the compiler can keep it in registers (if they fit).
- `k8` is `k` rounded down to a multiple of 8. The main loop handles 8 values of `p` per step; the leftovers are handled after.
- `std::array::from_fn(|r| ...)` builds a fixed-size array `[&[f32; 8]; MR]` by calling the closure for r = 0..MR: one reference per input row, pointing at the 8 values this step needs.
- `x[start..start + 8].try_into()` converts a slice (length known only at runtime) into a reference to a fixed-size array `&[f32; 8]` (length known at compile time). The conversion checks the length once. After that, `xs[r][lane]` with `lane < 8` needs no bounds check, and the compiler knows it is working with exactly 8 floats, which is what lets it emit vector code.
- For each weight row `c`, load its 8 values once and use them for all MR input rows. For each input row, the 8 values were loaded once and are used for all NR weight rows. That is the reuse.
- The innermost `for lane in 0..8` is written as a scalar loop, and the compiler turns it into one or two vector multiply-and-adds.

<!-- file: src/lib.rs -->
```rust
    for r in 0..MR {
        for c in 0..NR {
            let mut sum: f32 = acc[r][c].iter().sum();
            for p in k8..k {
                sum += x[(i0 + r) * k + p] * w[(j0 + c) * k + p];
            }
            y[(i0 + r) * n + j0 + c] = sum;
        }
    }
```

At the end, each output's eight lane sums are added together ("horizontal sum"), the leftover `k % 8` terms are added one by one, and the result is stored.

The caller runs the micro-kernel on every full tile and falls back to plain dot products for the ragged right and bottom edges:

<!-- file: src/lib.rs -->
```rust
    // The edges that do not fill a whole tile: plain dot products.
    for i in 0..m {
        let cols = if i < m_full { n_full..n } else { 0..n };
        for j in cols {
            y[i * n + j] = dot(row(x, i, k), row(w, j, k));
        }
    }
```

Rows that are covered by full tiles only need their last `n % NR` columns; rows below the last full tile need every column.

### 4.6 The tests: odd sizes and NaN poisoning

<!-- file: src/lib.rs -->
```rust
        // Odd sizes on purpose: they exercise every edge path.
        for (m, k, n) in [
            (1, 1, 1),
            (3, 5, 7),
            (7, 33, 9),
            (8, 64, 8),
            (13, 300, 17),
            (2, 600, 530),
        ] {
```

Tiled and blocked kernels are where edge bugs live: the last partial tile, the last partial block, the `k % 8` tail. So the test sizes are chosen to hit every combination: sizes below one tile, not multiples of 8 or of the tile sizes, and one case (`n = 530`) that crosses the `NC = 512` block boundary.

<!-- file: src/lib.rs -->
```rust
            for (name, f) in nn {
                c.fill(f32::NAN); // any element a kernel forgets stays NaN
                f(&a, &b, &mut c, m, k, n);
                assert_close(name, &c, &want, k);
            }
```

Before each kernel runs, the output is filled with NaN. If a kernel forgets to write some element (a missed edge), that element stays NaN and the comparison fails, even if the "right" answer happened to be sitting in the buffer from the previous kernel's run.

The comparison tolerance grows with √k, because rounding errors in a sum of k terms grow roughly like √k when they are random (chapter 2). The reference is computed in `f64`.

## 5. Run it

```bash
cargo test -p ch05-matmul
cargo run --release -p ch05-matmul
```

On the reference machine:

```text
single-core arithmetic peak measured by chapter 4's probe: 33.9 GFLOP/s

== square: [512 x 512] x [512 x 512]
   kernel            |      time | GFLOP/s | % of peak
   naive (ijk)       |  126.76ms |    2.12 |        6%
   loop order ikj    |   20.10ms |   13.35 |       39%
   blocked ikj       |   15.06ms |   17.82 |       53%
   NT dot products   |   18.07ms |   14.86 |       44%
   NT tiled 1x4      |   27.15ms |    9.89 |       29%
   NT tiled 2x4      |   14.05ms |   19.11 |       56%
   NT tiled 4x4      |   20.49ms |   13.10 |       39%

== square: [1024 x 1024] x [1024 x 1024]
   kernel            |      time | GFLOP/s | % of peak
   loop order ikj    |  219.79ms |    9.77 |       29%
   blocked ikj       |  118.52ms |   18.12 |       53%
   NT dot products   |  245.28ms |    8.76 |       26%
   NT tiled 1x4      |  128.99ms |   16.65 |       49%
   NT tiled 2x4      |  108.61ms |   19.77 |       58%
   NT tiled 4x4      |  177.81ms |   12.08 |       36%

== decode: 1 token: [1 x 4096] x [4096 x 4096]
   kernel            |      time | GFLOP/s | % of peak
   naive (ijk)       |  168.44ms |    0.20 |        1%
   loop order ikj    |    4.66ms |    7.21 |       21%
   blocked ikj       |    5.02ms |    6.69 |       20%
   NT dot products   |    5.84ms |    5.75 |       17%
   NT tiled 1x4      |    2.79ms |   12.01 |       35%
   NT tiled 2x4      |    3.87ms |    8.68 |       26%
   NT tiled 4x4      |    3.80ms |    8.83 |       26%

== decode: batch of 16: [16 x 4096] x [4096 x 4096]
   kernel            |      time | GFLOP/s | % of peak
   naive (ijk)       |     2.54s |    0.21 |        1%
   loop order ikj    |   51.56ms |   10.41 |       31%
   blocked ikj       |   42.99ms |   12.49 |       37%
   NT dot products   |   29.14ms |   18.42 |       54%
   NT tiled 1x4      |   31.90ms |   16.83 |       50%
   NT tiled 2x4      |   30.90ms |   17.37 |       51%
   NT tiled 4x4      |   39.29ms |   13.66 |       40%

== prefill: 128 tokens: [128 x 2048] x [2048 x 2048]
   kernel            |      time | GFLOP/s | % of peak
   naive (ijk)       |     4.96s |    0.22 |        1%
   loop order ikj    |   99.14ms |   10.83 |       32%
   blocked ikj       |   69.97ms |   15.34 |       45%
   NT dot products   |   52.09ms |   20.61 |       61%
   NT tiled 1x4      |   52.65ms |   20.40 |       60%
   NT tiled 2x4      |   55.67ms |   19.29 |       57%
   NT tiled 4x4      |   82.51ms |   13.01 |       38%
```

How to read this:

- **Naive to best is 10x on small matrices and about 100x on large ones.** Once `b` stops fitting in cache, the naive kernel drops to 0.2 GFLOP/s and stays there.
- **No single kernel wins everywhere.** Blocked i-k-j is good for large square NN problems; the 2 × 4 micro-kernel is best for square NT; plain NT dot products win the skinny batch-16 and prefill shapes; 1 × 4 wins decode. Real libraries pick the kernel by shape at runtime, and so will our engine.
- **Decode (m = 1) is a memory problem.** The best time, 2.79 ms for 64 MB of weights, is 24 GB/s. That is only possible because the 64 MB matrix still fits in this VM's share of L3 cache; a larger model would stream from DRAM at 10-12 GB/s per core (chapter 4). No micro-kernel changes that. Only reading fewer bytes (quantization) or using more cores (chapter 7) does. The 1 × 4 tile helps here only because it reads each input vector once per four weight rows instead of once per row.
- **Batch 16 costs only 10x the time of batch 1** (29 ms against 2.8 ms) for 16x the work. This is the roofline effect: batch 16 moves from memory-bound towards compute-bound.
- **The 4 × 4 tile loses everywhere** on this build because of register spills (section 3.7).

## 6. The Rust behind it

**Iterators instead of indices, for speed.** `chunks_exact` and `zip` give the compiler slices whose lengths match by construction, so it drops bounds checks and vectorizes. The same loop written with `c[i * n + j]` indexing is often slower. When you do need indices (the micro-kernel), convert to fixed-size arrays (`&[f32; 8]`) at the boundary, so the inner indexing is provably in range.

**Slices to arrays with `try_into`.** `<&[f32; 8]>::try_from(&slice[a..a + 8])` checks the length once and gives you a type that carries it. Many of the fastest pure-Rust kernels are built from this pattern.

**Const generics select micro-kernels.** `matmul_nt_tiled::<2, 4>` and `matmul_nt_tiled::<4, 4>` are separately compiled functions with their tile sizes baked in, so `acc` has a fixed size and the loops over `r` and `c` can be fully unrolled. The demo stores several instantiations in an array of function pointers and picks one at runtime: the same mechanism a library uses to choose a kernel per shape or per CPU.

**A closure lifetime gotcha.** A first version of this chapter had a helper closure `let row = |buf: &[f32], r: usize| -> &[f32] { ... }`. It does not compile: closures do not get the lifetime elision rules that functions get, so the compiler cannot tell that the returned slice borrows from `buf`. The fix is a tiny named function, `fn row(buf: &[f32], r: usize, k: usize) -> &[f32]`, where elision links the output lifetime to the input. You will hit this whenever a helper returns a borrow of its argument.

**`#[expect]` with a reason.** Clippy's `needless_range_loop` suggests replacing `for c in 0..NR` with an iterator. In the micro-kernel, `r` and `c` index four arrays at once, and the index form mirrors the tile structure, which is the point of the code. We record the decision with `#[expect(clippy::needless_range_loop, reason = "...")]`.

## 7. Mistakes you will make

- **Forgetting to zero the output** of an accumulating kernel. The first test passes (the buffer happened to be zero), later calls give garbage.
- **Testing only nice sizes.** Every tiled kernel works on 64 × 64 × 64. Test sizes like 7 × 33 × 9 and sizes just past a block boundary.
- **Mixing up NN and NT.** Feeding a PyTorch weight (`[out, in]`) to an NN kernel expecting `[in, out]` gives wrong answers for square matrices and a shape panic for others. Name the layout in the function name.
- **Tuning on a matrix that fits in cache.** A 256 × 256 benchmark says nothing about a 4096 × 4096 layer.
- **Bigger tiles are not always better.** When the accumulators stop fitting in registers, speed drops sharply. Measure on the target machine.

## 8. How the professionals do it

- **BLAS libraries** (OpenBLAS, Intel MKL, AMD AOCL-BLIS, Apple Accelerate) provide `sgemm`/`sgemv` tuned per CPU with packing, multi-level blocking and assembly micro-kernels. Apple's Accelerate uses the M-series AMX matrix units, which can make it several times faster than any NEON code on the same chip.
- **In Rust:** the `gemm` crate (used by `candle`) and `matrixmultiply` (used by `ndarray`) are pure-Rust GEMMs built on the same BLIS structure, with runtime selection of AVX2/AVX-512/NEON micro-kernels.
- **llama.cpp's ggml** writes its own GEMV/GEMM kernels because its weights are quantized (chapters 18-19): the dequantization must happen inside the micro-kernel, which standard BLAS cannot do.
- **On GPUs,** cuBLAS and CUTLASS (NVIDIA) and rocBLAS/hipBLASLt (AMD) do the same tiling at three levels: thread-block tiles in shared memory, warp tiles, and per-thread register tiles fed to tensor cores (chapter 29).
- **Inference engines choose kernels by shape.** vLLM and TensorRT-LLM ship separate GEMV and GEMM paths and switch between them on batch size.

## 9. Exercises

1. **Tile sweep.** Add 1 × 8, 2 × 2, 3 × 4 and 4 × 8 tiles to the demo. Which is best on your machine for each shape?
2. **Block sizes.** Try `KC`/`NC` of 64/128, 256/512, 512/1024 and 1024/2048 for the 1024 × 1024 and 2048 × 2048 NN problems. Explain the results with the L2 size.
3. **Blocked NT.** For the prefill shape, block the *weights* (groups of rows of `w` that together fit in L2) and run all input rows against one group before moving on. Does it beat `matmul_nt`?
4. **Predict decode.** For the decode shape, compute the intensity and the roofline bound using chapter 4's numbers (single-core DRAM bandwidth ~11 GB/s). Our best kernel ran at 12 GFLOP/s. Is that consistent? Why?
5. **Native build.** Run with `RUSTFLAGS="-C target-cpu=native"`. How much does the peak change, and how much do the kernels change? Why does 4 × 4 improve more than 2 × 4?
6. **Model FLOPs.** A linear layer with k = 576 inputs and n = 1,536 outputs (one of SmolLM2's MLP layers) runs on a 100-token prompt. How many FLOPs? At 20 GFLOP/s, how long does it take?

## 10. Check yourself

1. Why is `y = W x` for a PyTorch linear layer a dot product of two contiguous rows?
2. What are the two separate reasons the naive ijk loop is slow?
3. What does cache blocking change, if not the arithmetic?
4. How does a register-tiled micro-kernel reduce loads per multiply-add?
5. Why did the 4 × 4 tile run slower than 2 × 4 on the default target, but not on the native one?
6. Why do decode kernels care about memory bandwidth and prefill kernels about arithmetic?

## 11. Recap

- Every linear layer is a matmul: `Y = X · Wᵀ` with weights stored as rows (NT). Decode is GEMV (m = 1), prefill is GEMM.
- FLOPs = 2mkn. Square matmul has intensity N/6: compute-bound if the kernel reuses data well.
- Loop order decides vectorization and cache behaviour: naive ijk 0.2-2 GFLOP/s, i-k-j 10-13.
- Cache blocking keeps a block of the reused operand in L2: 18 GFLOP/s at 1024².
- Register tiling reuses loaded values from registers: best 2 × 4 at 20 GFLOP/s (58% of peak), limited by register count.
- Decode is memory-bound whatever the kernel. The rest of the course attacks it with bytes (quantization), bandwidth (threads) and reuse (batching).

## Answers

**Exercises**

1. Measured on the reference machine (default target), GFLOP/s at 1024³ / decode / prefill: 1 × 8: 12.1 / 9.4 / 16.6; 2 × 2: 16.0 / 8.5 / 23.2; 2 × 4: 18.0 / 8.9 / 22.8; 3 × 4: 13.5 / 8.5 / 13.7; 4 × 8: 10.7 / 9.1 / 10.7; 1 × 4: 16.8 / 12.4 / 20.6. Small-to-medium tiles (2 × 2, 2 × 4, 1 × 4) win; anything whose accumulators exceed the 16 SSE registers loses. Expect a different winner on ARM (32 NEON registers) and with AVX-512.
2. Measured on the reference machine (GFLOP/s at 1024² / 2048²): 64/128 (a 32 KB block): 14.3 / 10.5; 256/512 (512 KB): 16.4 / 13.3; 512/1024 (2 MB): 11.0 / 9.8; 1024/2048 (8 MB): 9.0 / 9.2. The 512 KB block wins at both sizes. A 2 MB block fills the whole L2 by itself, leaving no room for the rows of `a` and `c` streaming through, so it thrashes; an 8 MB block is effectively unblocked. Tiny blocks fit easily but pay more loop overhead and reload `c` more often.
3. Measured for the prefill shape (128 × 2048 × 2048): plain `matmul_nt` 13.9 GFLOP/s in this run; with groups of 16 weight rows (128 KB) 23.0; 64 rows (512 KB) 22.5; 128 rows (1 MB) 20.3; 256 rows (2 MB) 11.4. Blocking the weights helps a lot here: with the plain loop, all 128 input rows (1 MB) are re-read from L2/L3 for every single weight row, while with a group of 16 weight rows kept in L1/L2, each input row is loaded once per group. As in exercise 2, a group that fills all of L2 undoes the benefit.
4. Intensity for `f32` GEMV is 0.5 FLOP/byte. At 11 GB/s from DRAM, the bound is 5.5 GFLOP/s. We measured 12 GFLOP/s, *above* that bound, which means the 64 MB matrix was not coming from DRAM: it was in L3, where chapter 4 measured ~20-23 GB/s for one core, giving a bound of 10-11.5 GFLOP/s. That fits, to within noise. A useful habit: when a measurement beats the roofline, your assumption about where the data lives is wrong.
5. Measured: the single-core peak doubled (29.7 → 59.3 GFLOP/s) but the kernels improved only 0-45%: 2 × 4 stayed at 18.0 on 1024³, 1 × 4 went 16.8 → 24.7, 4 × 4 went 12.3 → 17.1. Wider registers help only when the code is shaped to use them, and our micro-kernel still does separate multiplies and adds on 8-float arrays. 4 × 4 improves most because AVX-512's 32 registers finally hold its accumulators without spilling.
6. 2 × 100 × 576 × 1,536 ≈ 177 MFLOPs. At 20 GFLOP/s: about 8.8 ms.

**Check yourself**

1. PyTorch stores the weight as `[out_features, in_features]`, so output j's weights are row j, contiguous. Output j is the dot product of that row with the input vector.
2. The inner loop walks a column of `b` (one useful float per cache line), and it accumulates into one variable (a dependency chain the compiler may not reorder or vectorize).
3. The order in which data is used, so that a block of the reused operand stays in cache while all its uses happen. Fewer bytes come from slow memory.
4. It loads MR values of `x` and NR values of `w` and performs MR × NR multiply-adds on them from registers, instead of two loads per multiply-add.
5. Its 128 accumulator floats need 32 SSE registers but the default target has 16, so values spill to memory. AVX-512 has 32 registers of 16 floats, enough to hold them.
6. Decode multiplies by one vector, so each weight byte is used for one multiply-add (low intensity, memory-bound). Prefill multiplies by many token vectors, so each weight is reused many times (high intensity, compute-bound).

## Further reading

- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication", ACM TOMS, 2008. The blocking structure every BLAS uses.
- Van Zee and van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality", 2015.
- Siboehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (blog post). The same ideas on a GPU.
- Next: [Chapter 6: SIMD](../06-simd/README.md). We stop hoping the compiler vectorizes and write the vector instructions ourselves.
