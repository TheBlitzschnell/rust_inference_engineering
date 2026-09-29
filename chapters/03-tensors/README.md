# Chapter 3: Tensors, strides and views

> **In one sentence:** a tensor is one flat run of numbers plus a little metadata (shape, strides, offset), and by changing only the metadata you can transpose, slice, split and broadcast it without copying a single number.

**Where this fits:** every model is a pile of tensors: weight matrices, the activations flowing between layers, the attention scores, the KV cache. This chapter is how they sit in memory. Chapter 9 will point views like these straight into a memory-mapped weight file, and chapters 12-14 use the same stride arithmetic to split attention heads.

**You need:** chapters 1-2. Rust references and slices.

**You will build:** an owned `Tensor` and a borrowed `TensorView<'a>` with zero-copy transpose, slice, select, broadcast and reshape; a demonstration, checked by the compiler, of the bugs the borrow checker rules out; and measurements of what a copy costs compared with a view.

---

## 1. The intuition

Think of a long bookshelf holding 24 books in a single row, numbered 0 to 23. Nothing about the shelf says "these are 2 boxes of 3 rows of 4 books". That structure is only in how you *read* the shelf.

If someone gives you the rule "to get to the next box, skip 12 books; to the next row, skip 4; to the next book, skip 1", you can find "box 1, row 2, book 3" instantly: 1×12 + 2×4 + 3×1 = book 23. Those skip distances are the **strides**.

Now someone asks for the collection "read column-wise instead of row-wise". You could take every book off the shelf and put them back in a new order (a **copy**). Or you could hand them a new rule card with the skip distances swapped (a **view**). The shelf does not change. Only the card does.

That is the whole chapter. A tensor is the shelf plus a rule card. Most tensor operations only rewrite the card.

**Where the analogy breaks:** reading books in a scattered order off a real shelf costs the same walking as reading them in order if you have to walk to each one anyway. In a computer it does not. Memory is fetched in chunks of 64 bytes (16 `f32` numbers). Reading in shelf order uses every number in each chunk; reading with big jumps may use one number per chunk and waste the other fifteen. So views are free to *create* but not always free to *use*. Section 5 measures a 6x difference.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Tensor** | A multi-dimensional array of numbers. A vector is a 1-D tensor, a matrix 2-D. |
| **Rank** | The number of dimensions. A matrix has rank 2. |
| **Shape** | The size of each dimension, e.g. `[2, 3, 4]`. |
| **Element** | One number in the tensor. |
| **Row-major** | Layout where the *last* index changes fastest in memory. Used by C, Rust, PyTorch, NumPy, safetensors. |
| **Column-major** | Layout where the *first* index changes fastest. Used by Fortran, MATLAB, BLAS by default. |
| **Stride** | How many elements to skip in the flat buffer to move one step along a dimension. |
| **Offset** | Where in the buffer the view's first element is. |
| **View** | A tensor that borrows another tensor's buffer with its own shape/strides/offset. |
| **Contiguous** | The view's elements, in row-major order, sit side by side in memory with no gaps. |
| **Broadcasting** | Treating a smaller tensor as if it were repeated to match a larger shape. |
| **Cache line** | The 64-byte unit in which memory is moved between RAM and the CPU. |

## 3. The concepts in depth

### 3.1 Everything is a flat buffer

Memory is one-dimensional: addresses 0, 1, 2, ... A 2 × 3 matrix has to be laid out in a line somehow. Row-major order stores row 0, then row 1:

```text
matrix              memory (row-major)
┌───┬───┬───┐       ┌───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │       │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
├───┼───┼───┤       └───┴───┴───┴───┴───┴───┘
│ 3 │ 4 │ 5 │        row 0       row 1
└───┴───┴───┘
```

Everything in this course is row-major, because PyTorch and the safetensors files we load in chapter 9 are. A PyTorch linear layer stores its weight with shape `[out_features, in_features]`, so each output's weights are one contiguous row. That is exactly the layout chapter 1's `dot(row, x)` wanted.

### 3.2 Strides

For row-major shape `[d0, d1, ..., dn]` the strides are computed from the right: the last dimension has stride 1, and each dimension's stride is the product of all the sizes to its right.

```text
shape   [2, 3, 4]
strides [12, 4, 1]      12 = 3×4,  4 = 4,  1

position of element [i, j, k] = offset + i×12 + j×4 + k×1
```

This one formula, `offset + Σ index[d] × stride[d]`, is how *every* element of *every* view is found. PyTorch, NumPy, and the tensors inside GPU kernels all use it.

### 3.3 Views: operations that only touch metadata

**Transpose** swaps two dimensions. In stride terms: swap their sizes and their strides.

```text
original [2, 3]  strides [3, 1]   element [r, c] at r×3 + c
transpose [3, 2] strides [1, 3]   element [c, r] at c×1 + r×3   ← same place
```

The demo prints the transposed view of `[0, 1, 2, 3, 4, 5]` as `[0, 3, 1, 4, 2, 5]`: the columns of the original, read as rows.

**Slice** keeps a sub-range of one dimension. The offset moves forward by `start × stride` and the size shrinks. Rows 1..3 of a 4 × 5 matrix start at offset 5.

**Select** picks one index along a dimension and drops that dimension. Selecting row 3 of a matrix gives a 1-D view whose stride is 1. Selecting column 2 gives a 1-D view whose stride is the row length. This is exactly how an embedding lookup works (chapter 13): "the vector for token 42" is row 42 of the embedding matrix, handed out as a view.

**Broadcast** repeats a tensor along a dimension of size 1 by setting that dimension's stride to **0**. Stepping along a stride-0 dimension lands on the same numbers again and again. Adding a bias vector `[3]` to every row of a `[2, 3]` matrix is "broadcast the bias to `[2, 3]`, then add element by element", and no copy of the bias is ever made.

**Reshape** reinterprets the element count as a different shape, e.g. `[2, 6]` as `[3, 4]`. For a contiguous tensor it is free: compute new row-major strides and you are done. For a non-contiguous view (a transpose), there is no stride assignment that works, and reshape must copy. Our `reshape` returns `None` in that case, so the copy is never hidden.

### 3.4 Contiguity, and when copies are unavoidable

A view is **contiguous** if walking it in row-major order touches consecutive memory addresses with no gaps. Only then can you hand the elements to a kernel as a plain `&[f32]` slice.

Kernels (matmul, SIMD dot products) want contiguous input, because it is what the hardware reads fastest and it makes the code simple. So the life of a tensor in an inference engine is:

1. Create views freely: slice out Q, K and V from one fused projection output, split attention heads, select rows.
2. When a hot kernel needs a particular layout, either choose the layout up front so that the view is already contiguous, or make **one** deliberate contiguous copy.

PyTorch calls the copy `.contiguous()`. Many performance bugs in Python inference code are hidden `.contiguous()` calls inside library functions. In our code, the only copy is a method literally named `to_contiguous`, and it returns a new owned `Tensor`, so a copy is visible in the type as well as the name.

The best engines go further: they choose memory layouts so the hot path never needs a copy at all. For example, chapter 14's KV cache is laid out so that the attention kernel reads each head's keys as one contiguous run.

### 3.5 Where views show up in inference

You will meet every one of these later in the course:

| Operation | View used | Chapter |
|---|---|---|
| Embedding lookup: vector for token *t* | `select(0, t)` on the embedding matrix | 13 |
| Split a fused QKV output into Q, K, V | `slice` along the feature dimension | 13 |
| `[seq, heads × head_dim]` → `[heads, seq, head_dim]` | `reshape` then `transpose` | 12 |
| Add a bias to every token | `broadcast_to` | 10 |
| Read the last token's logits | `select(0, seq_len − 1)` | 14 |
| Weight matrix inside a memory-mapped file | a view with a lifetime tied to the file | 9 |

### 3.6 The cost of access order

The same element can be cheap or expensive to reach depending on what you read just before it. The CPU does not fetch single numbers from RAM; it fetches 64-byte **cache lines** and keeps recently used lines in fast on-chip caches (chapter 4 measures the sizes and speeds).

Summing a matrix row by row reads memory in order: each 64-byte line gives 16 useful numbers. Summing column by column jumps a whole row ahead each time: each line gives 1 useful number, and by the time you come back for its neighbours, the line may have been evicted. Section 5 measures a 6.3x difference for the same additions.

This is why a transposed *view* is free to create but a kernel should not iterate over it element by element. It is also why a transposed *copy* is itself a tricky kernel: one side of it (reading or writing) is always strided. **Tiling** fixes that. Transpose a small square block at a time, small enough that the rows being read and the rows being written all stay in cache while you work on the block. Every cache line is then fully used before it is evicted.

## 4. The code

All in [`src/lib.rs`](src/lib.rs). The demo is in [`src/main.rs`](src/main.rs).

### 4.1 Strides from a shape

<!-- file: src/lib.rs -->
```rust
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut step = 1;
    for (stride, &dim) in strides.iter_mut().zip(shape).rev() {
        *stride = step;
        step *= dim;
    }
    strides
}
```

Walk the dimensions from the last to the first (`.rev()`). The last gets stride 1. Each earlier one gets the running product of the sizes after it. `iter_mut().zip(shape).rev()` pairs each output slot with its dimension size and walks the pairs backwards, with no index arithmetic to get wrong.

### 4.2 Owned versus borrowed

<!-- file: src/lib.rs -->
```rust
#[derive(Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

<!-- file: src/lib.rs -->
```rust
#[derive(Clone, PartialEq)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}
```

The difference is one field: `data: Vec<f32>` versus `data: &'a [f32]`.

- `Tensor` **owns** its numbers. When a `Tensor` is dropped, its buffer is freed. Cloning a `Tensor` copies every number.
- `TensorView<'a>` **borrows** them. It is a pointer, a length, and some metadata. Cloning a `TensorView` copies only the metadata (the pointer and the two small `Vec<usize>`s), never the numbers. Every view method in this crate starts with `self.clone()` and edits the copy's metadata, so a view operation costs a few dozen bytes no matter how big the tensor is.

The lifetime parameter `'a` names "the buffer this view points into". Every view created from a view carries the same `'a`, so the compiler can track, across any chain of `transpose`, `slice` and `select` calls, which buffer each view depends on.

An owned `Tensor` keeps only its shape. Its strides are always the contiguous ones, so storing them would just be a chance for them to be wrong.

### 4.3 From tensor to view

<!-- file: src/lib.rs -->
```rust
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }
```

`&self` in, `TensorView<'_>` out. The `'_` says "the view borrows from `self`". While the returned view exists, `self` counts as borrowed, and the rules in section 6 apply.

### 4.4 Finding an element

<!-- file: src/lib.rs -->
```rust
    pub fn position(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.rank(), "index has the wrong rank");
        let mut pos = self.offset;
        for ((&i, &dim), &stride) in index.iter().zip(&self.shape).zip(&self.strides) {
            assert!(
                i < dim,
                "index {index:?} out of bounds for shape {:?}",
                self.shape
            );
            pos += i * stride;
        }
        pos
    }
```

This is section 3.2's formula, with a bounds check per dimension. Why check each index against its dimension when the final `self.data[pos]` access is bounds-checked anyway? Because an index can be wrong but still land inside the buffer. For a 3 × 4 matrix, index `[0, 5]` computes position 5, which is element `[1, 1]`: in bounds, wrong answer, no crash. Checking every dimension turns silent corruption into a clear panic.

### 4.5 Transpose, slice, select

<!-- file: src/lib.rs -->
```rust
    #[must_use]
    pub fn transpose(&self, a: usize, b: usize) -> TensorView<'a> {
        let mut view = self.clone();
        view.shape.swap(a, b);
        view.strides.swap(a, b);
        view
    }
```

Two swaps. That is a transpose.

`#[must_use]` makes the compiler warn if the caller ignores the result. Someone who writes `v.transpose(0, 1);` expecting `v` itself to change would otherwise get no hint that nothing happened.

The return type is `TensorView<'a>`, not `TensorView<'_>`: the new view borrows from the *original buffer* (lifetime `'a`), not from the view it was created from. So you can create a view, transpose it, drop the first view, and keep the transposed one. It only has to live as long as the buffer.

<!-- file: src/lib.rs -->
```rust
    #[must_use]
    pub fn slice(&self, dim: usize, range: Range<usize>) -> TensorView<'a> {
        assert!(range.start <= range.end && range.end <= self.shape[dim]);
        let mut view = self.clone();
        view.offset += range.start * self.strides[dim];
        view.shape[dim] = range.end - range.start;
        view
    }
```

Move the starting point forward by `start` steps along `dim`, and shrink the size. The strides are unchanged. So a slice of rows stays contiguous, while a slice of columns does not: its rows are now shorter than the stride between them. The test `slicing_moves_the_offset` checks both.

### 4.6 Broadcasting with stride 0

<!-- file: src/lib.rs -->
```rust
        let extra = shape.len() - self.rank();
        let mut strides = vec![0; shape.len()];
        for (d, &target) in shape.iter().enumerate().skip(extra) {
            let (have, stride) = (self.shape[d - extra], self.strides[d - extra]);
            strides[d] = if have == target {
                stride
            } else {
                assert_eq!(have, 1, "cannot broadcast {:?} to {shape:?}", self.shape);
                0
            };
        }
```

The broadcasting rules (the same as NumPy's and PyTorch's):

1. Line the shapes up from the right. Missing leading dimensions count as size 1. The new leading dimensions keep stride 0 from `vec![0; ...]`.
2. Where the sizes match, keep the stride.
3. Where the original size is 1, use stride 0: every index along that dimension reads the same element.
4. Anything else is an error.

A bias of shape `[3]` broadcast to `[2, 3]` gets strides `[0, 1]`. The demo prints exactly that.

### 4.7 Contiguity and the path to a kernel

<!-- file: src/lib.rs -->
```rust
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            // A dimension of size 1 is never stepped along, so its stride
            // does not matter.
            if dim != 1 && stride != expected {
                return false;
            }
            expected *= dim;
        }
        true
    }
```

Walk from the last dimension, checking that each stride equals the product of the sizes to its right. Size-1 dimensions are skipped because you never step along them: `[1, 3]` with strides `[0, 1]` (a broadcast that only added a leading 1) is still contiguous.

<!-- file: src/lib.rs -->
```rust
    pub fn as_slice(&self) -> Option<&'a [f32]> {
        self.is_contiguous()
            .then(|| &self.data[self.offset..self.offset + self.len()])
    }
```

If the view is contiguous, its elements are exactly `data[offset .. offset + len]`, and we can return that slice. This is the zero-cost handoff from the view world to the kernel world. The return type `Option<&'a [f32]>` forces the caller to handle the non-contiguous case: they cannot pretend a transposed view is a flat slice.

`bool::then` runs the closure only when the condition is true, turning it into `Some(...)` or `None`.

### 4.8 Walking any view: the odometer

<!-- file: src/lib.rs -->
```rust
        let mut index = vec![0; self.rank()];
        loop {
            f(self.get(&index));
            // Advance the index like an odometer: bump the last digit, carry left.
            let mut d = self.rank();
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                index[d] += 1;
                if index[d] < self.shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
```

To visit a view with arbitrary strides in row-major order, keep a multi-dimensional index and increment it like a car's odometer: add 1 to the last digit; if it overflows its dimension, reset it to 0 and carry into the digit to its left. When the carry falls off the left end, every element has been visited.

`for_each` checks `as_slice()` first and uses a plain slice iteration when possible. The odometer is the slow, general path, and it is slow: a function call, a rank-sized loop and a bounds check per element. Fine for tests and one-off copies; not something to put in a hot loop.

### 4.9 Deliberate copies

<!-- file: src/lib.rs -->
```rust
    pub fn to_contiguous(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.len());
        self.for_each(|x| data.push(x));
        Tensor::from_vec(data, &self.shape)
    }
```

`Vec::with_capacity` allocates once, up front, so `push` never has to grow the buffer. The result is an owned `Tensor`: a copy you can see in the type.

<!-- file: src/lib.rs -->
```rust
pub fn transpose_copy_tiled(src: &[f32], rows: usize, cols: usize, dst: &mut [f32], tile: usize) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * cols);
    for r0 in (0..rows).step_by(tile) {
        for c0 in (0..cols).step_by(tile) {
            for r in r0..(r0 + tile).min(rows) {
                for c in c0..(c0 + tile).min(cols) {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}
```

The outer two loops walk over `tile × tile` blocks; the inner two transpose one block. Within a block, the reads touch `tile` source rows and the writes touch `tile` destination rows. With `tile = 32`, that is 32 rows × 128 bytes on each side, 8 KB in total, which fits comfortably in the 48 KB level-1 cache of this CPU. `.min(rows)` and `.min(cols)` handle the ragged edge when the matrix size is not a multiple of the tile.

### 4.10 Disjoint mutable pieces

<!-- file: src/lib.rs -->
```rust
    pub fn row_chunks_mut(&mut self, rows_per_chunk: usize) -> std::slice::ChunksMut<'_, f32> {
        assert_eq!(self.shape.len(), 2, "row_chunks_mut needs a 2-D tensor");
        let cols = self.shape[1];
        self.data.chunks_mut(rows_per_chunk * cols)
    }
```

`chunks_mut` hands out several `&mut [f32]` pieces of one buffer at the same time. That looks like it breaks Rust's "only one `&mut` at a time" rule, but it does not: the pieces never overlap, and the standard library's implementation guarantees that. This is the foundation of every parallel kernel in chapter 7: give each thread its own non-overlapping piece of the output, and data races are impossible by construction.

## 5. Run it

```bash
cargo test -p ch03-tensors      # includes the compile_fail examples of section 6
cargo run --release -p ch03-tensors
```

On the reference machine:

```text
== 1. one buffer, many views
   buffer:        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
   as [2, 3]:     shape [2, 3] strides [3, 1] -> Tensor(shape=[2, 3], values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
   transposed:    shape [3, 2] strides [1, 3] -> Tensor(shape=[3, 2], values=[0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
   column 2:      shape [2] strides [3] offset 2 -> Tensor(shape=[2], values=[2.0, 5.0])
   broadcast:     shape [2, 3] strides [0, 1] -> Tensor(shape=[2, 3], values=[100.0, 200.0, 300.0, 100.0, 200.0, 300.0])
   reshape of the transpose possible? false

== 2. transposing a 4096 x 4096 f32 matrix (64 MB)
   view (strides swapped):    64.00ns
   naive copy:               237.20ms
   tiled copy (tile   8):     83.25ms
   tiled copy (tile  32):     84.42ms
   tiled copy (tile 128):     92.69ms

== 3. summing a matrix: row order vs column order
    256 x 256  (  0.3 MB): rows   39.09µs  columns   61.16µs  (1.6x slower)
   1024 x 1024 (  4.2 MB): rows    1.39ms  columns    9.04ms  (6.5x slower)
   4096 x 4096 ( 67.1 MB): rows   25.75ms  columns  162.28ms  (6.3x slower)
```

Reading the results:

- **A view costs 64 nanoseconds; a copy costs 83-237 milliseconds.** That is a factor of more than a million. Most of the 64 ns is allocating the two small `Vec<usize>`s for shape and strides.
- **Tiling makes the copy almost 3x faster** with no change to the arithmetic (there is none: a transpose only moves numbers). For comparison, a straight `memcpy` of the same 64 MB took about 10 ms on this machine, so even the tiled transpose leaves a lot on the table. Production transposes also use SIMD shuffles.
- **Column order is 6.3x slower** for matrices that do not fit in cache. For a 256 × 256 matrix (256 KB, which fits in the 2 MB L2 cache) the penalty is only 1.6x: once everything is in cache, access order matters much less. Chapter 4 maps out exactly where these cliffs are.
- **Row order is not as fast as it could be.** 67 MB in 25.75 ms is only 2.6 GB/s, far below this machine's memory bandwidth. The loop has a single running total, so each addition waits for the previous one: the bottleneck is the chain of additions, not memory. You saw the fix in chapter 1's `dot` (several running sums), and chapter 6 explains it.

## 6. The Rust behind it

### 6.1 The bugs that do not compile

These three programs are in the crate's documentation as `compile_fail` tests. `cargo test` checks that each one fails to compile, and with the exact error code shown, so these claims are verified, not just asserted.

**A view cannot outlive its tensor** (E0597):

```rust
let view;
{
    let t = Tensor::arange(&[2, 2]);
    view = t.view();
} // `t` is freed here, so `view` would point at freed memory
println!("{:?}", view.shape());
```

In C or C++ this compiles and reads freed memory: a use-after-free. It might print the right numbers (the memory has not been reused yet), or garbage, or crash, depending on what the allocator does next. In an inference server, the classic version is a request that keeps a pointer into a buffer after the buffer was recycled for another request.

**The data cannot change under a reader** (E0502):

```rust
let mut t = Tensor::arange(&[2, 2]);
let view = t.view();
t.data_mut()[0] = 42.0; // would change what `view` sees underneath it
println!("{}", view.get(&[0, 0]));
```

Rust's core rule: at any moment you can have **many readers or one writer**, never both. A view is a reader. This rule is what makes the next point possible.

**The buffer cannot be moved or freed while viewed** (E0505):

```rust
let t = Tensor::arange(&[2, 2]);
let view = t.view();
let buffer = t.into_vec(); // gives the buffer away while `view` uses it
println!("{} {}", view.get(&[0, 0]), buffer.len());
```

A C++ programmer knows this bug as **iterator invalidation**: keep a pointer into a `std::vector`, push onto the vector, the vector reallocates, and the pointer now dangles. Rust rejects the equivalent program.

### 6.2 Why this matters specifically for inference

An inference engine is full of long-lived shared buffers (weights, KV cache blocks, memory-mapped files) and short-lived pointers into them (views for each request, each layer, each attention head). The pressure to avoid copies is constant, because copies cost memory bandwidth, which is the resource we are shortest of. Zero-copy designs in C++ are fast and fragile: any code path that frees or reuses a buffer too early corrupts some other request's output, often without crashing. With lifetimes, the zero-copy design is checked by the compiler, and the check has no runtime cost.

### 6.3 Other things worth noticing

- **`Clone` on a view is cheap and on a tensor is expensive**, and the same `.clone()` spelling hides that difference. When you read `.clone()` in inference code, check the type.
- **Each view allocates two small `Vec<usize>`s.** Fine here. Production libraries avoid even that with small fixed-capacity arrays (like `[usize; 6]` plus a rank) or const-generic ranks (`TensorView<'a, const RANK: usize>`) that live entirely on the stack.
- **`Option` instead of hidden copies.** `reshape` and `as_slice` return `None` when the operation would need a copy, instead of copying silently. The caller decides, and the decision is visible.
- **`assert!` in constructors.** A shape that does not match the buffer is a bug in the caller. Panicking immediately, with the shapes in the message, is far easier to debug than a wrong answer three layers later.

## 7. Mistakes you will make

- **Transposing the wrong way.** A PyTorch linear weight is `[out, in]`. The math `y = W x` uses it as is; the math `y = x Wᵀ` (the way PyTorch writes it) transposes it. Mixing the two produces a wrong-but-plausible result for square matrices and a shape panic for non-square ones. Always test with non-square shapes.
- **Iterating a transposed view in a hot loop.** It works and gives the right answer, several times slower than necessary. Copy once to a contiguous layout, or better, pick a layout that needs no transpose.
- **Treating strides as bytes.** Our strides count elements. Some libraries (NumPy) count bytes. When you move between them, multiply or divide by the element size.
- **Forgetting the offset.** A view created by `slice` or `select` does not start at element 0 of the buffer. Code that uses `data[0..len]` instead of `data[offset..offset + len]` reads the wrong numbers, silently.
- **Broadcasting by accident.** A `[1, n]` and an `[n, 1]` tensor broadcast to `[n, n]`. If you meant element-wise, you have just done n times the work and gotten a matrix.

## 8. How the professionals do it

- **PyTorch** tensors are exactly this design: a storage (reference-counted buffer), plus sizes, strides and a storage offset. `t.stride()`, `t.is_contiguous()` and `t.contiguous()` are the same ideas. PyTorch allows negative strides only through special operations; NumPy allows them for reversed views.
- **`ndarray`** (Rust) provides `ArrayBase` with owned (`Array`) and borrowed (`ArrayView`, `ArrayViewMut`) variants, the same split as here, with lifetimes doing the same job.
- **`candle`** (Hugging Face's Rust ML framework) stores tensors as a reference-counted storage plus a `Layout` of shape, strides and start offset.
- **GPU kernels** receive raw pointers plus shape and stride arguments, and compute the same index formula in every thread. Kernel authors choose layouts so that neighbouring threads read neighbouring addresses ("coalescing", chapter 29), which is the GPU version of section 3.6.

## 9. Exercises

1. **`permute`.** Implement `permute(&self, order: &[usize]) -> TensorView<'a>`, which reorders all dimensions at once (`permute(&[1, 0])` is a transpose). Test that `permute(&[2, 0, 1])` on a `[2, 3, 4]` tensor gives shape `[4, 2, 3]` and the right elements.
2. **Split attention heads.** A tensor of shape `[seq, n_heads × head_dim]` holds, for each token, all heads side by side. Using only `reshape` and `transpose`, produce a view of shape `[n_heads, seq, head_dim]`. Is the result contiguous? Which element does `[h, s, d]` read?
3. **`unsqueeze` and `squeeze`.** Add a dimension of size 1 at a given position, and remove one. What stride should the new dimension get? Does it matter?
4. **Tile sizes.** Run `transpose_copy_tiled` for tile sizes 1, 2, 4, 8, 16, 32, 64, 128, 256 and 1024 on the 4096 × 4096 matrix. Plot or tabulate the times. Explain the shape of the curve at both ends.
5. **Broadcast add.** Write `fn add(a: &TensorView, b: &TensorView) -> Tensor` that broadcasts both inputs to a common shape and adds them. Test `[2, 3] + [3]` and `[2, 1] + [1, 3]`.
6. **The row-order sum.** Rewrite `sum_row_order` with eight running totals, like chapter 1's `dot`. How much faster is it at 4096 × 4096? Is it now limited by memory?

## 10. Check yourself

1. What are the strides of a contiguous `[4, 5, 6]` tensor?
2. What happens to shape, strides and offset when you (a) transpose, (b) slice, (c) broadcast?
3. Why can a transposed matrix not be reshaped without copying?
4. Why is summing a large matrix column by column slower than row by row, when it performs the same additions?
5. What is the lifetime `'a` in `TensorView<'a>` protecting against?
6. Why is it safe for `chunks_mut` to hand out several `&mut` slices of the same buffer at once?

## 11. Recap

- A tensor is a flat buffer plus shape, strides and offset. The address of an element is `offset + Σ index × stride`.
- Transpose, slice, select, broadcast (stride 0) and contiguous reshape only change metadata, and cost nanoseconds regardless of size.
- Non-contiguous views are free to create but can be slow to iterate, because memory is fetched in 64-byte lines. Copy once, deliberately, when a hot kernel needs contiguity, or choose layouts so it never does.
- Tiling turns a strided copy into a cache-friendly one: 237 ms down to 83 ms here.
- In Rust, `Tensor` owns and `TensorView<'a>` borrows. The borrow checker rejects dangling views, mutation under a reader, and freeing a buffer that is still viewed, at compile time and at no runtime cost.

## Answers

**Exercises**

1. Build new `shape` and `strides` vectors by indexing the old ones with `order`: `shape[i] = old_shape[order[i]]`, and the same for strides. Assert that `order` is a permutation (every dimension exactly once). Element `[a, b, c]` of the permuted view is element `[b, c, a]` of the original for `order = [2, 0, 1]`.
2. `reshape(&[seq, n_heads, head_dim])` (possible because the input is contiguous), then `transpose(0, 1)`. The result has shape `[n_heads, seq, head_dim]` and strides `[head_dim, n_heads × head_dim, 1]`. It is **not** contiguous: consecutive tokens of one head are `n_heads × head_dim` elements apart. Element `[h, s, d]` reads buffer position `s × n_heads × head_dim + h × head_dim + d`. Each head's row of `head_dim` numbers is still contiguous, which is what attention kernels need (chapter 12).
3. The stride of a size-1 dimension never gets multiplied by anything but 0, so any value works. PyTorch uses the stride the dimension would have in a contiguous layout. With the `is_contiguous` in this chapter (which skips size-1 dimensions), either choice keeps a contiguous tensor contiguous.
4. Measured on the reference machine (two runs): tile 1: 426/384 ms, 2: 213 ms, 4: 112-123 ms, 8: 82-93 ms, 16: 99-110 ms, 32: 74 ms, 64: 74-99 ms, 128: 85-87 ms, 256: 90 ms, 1024: 185-224 ms. With tiny tiles, the loop overhead of four nested loops dominates (tile 1 is slower than the plain naive copy). With huge tiles, a block no longer fits in cache and you are back to the naive access pattern. The best region, roughly 8 to 128, is flat and noisy: many sizes are "good enough", and 32 was the best here.
5. Compute the broadcast shape (right-aligned, each dimension the maximum of the two, after checking that each pair is equal or contains a 1), call `broadcast_to` on both views, then walk both with the odometer and push `x + y`. `[2, 1] + [1, 3]` gives `[[a0+b0, a0+b1, a0+b2], [a1+b0, a1+b1, a1+b2]]`.
6. Eight independent totals break the dependency chain. Measured on the reference machine over three runs: 7.0-8.7 ms against 26-31 ms for the single total, 3.6-4.0x faster, about 9.5 GB/s. Chapter 4 measures that one core of this machine reads main memory at roughly that speed, so the loop has gone from being limited by the addition chain to being limited by memory.

**Check yourself**

1. `[30, 6, 1]`.
2. (a) Two sizes and the matching two strides are swapped; the offset is unchanged. (b) One size shrinks and the offset moves forward by `start × stride`; strides are unchanged. (c) The shape grows; broadcast dimensions get stride 0; the offset is unchanged.
3. Reshape requires that the elements, in row-major order of the view, are evenly spaced in memory so that new strides can describe them. In a transposed matrix, consecutive elements of a view row are a whole original row apart, and wrapping to the next view row jumps back by almost the full buffer. No single set of strides describes that, so the data must be reordered.
4. Memory moves in 64-byte cache lines. Row order uses all 16 floats of each line; column order uses one per line and may have to fetch each line again later, so it moves up to 16 times more data from RAM.
5. Against the view being used after the buffer it points into has been freed, moved, reallocated or modified. The compiler proves the buffer outlives the view and stays unchanged while the view exists.
6. The chunks never overlap, so no two `&mut` references can reach the same element. The standard library encodes that guarantee in `chunks_mut`'s signature, and the borrow checker enforces that the parent buffer is not used while the chunks exist.

## Further reading

- The NumPy documentation on "The N-dimensional array" and on broadcasting rules. The stride model there is the same.
- Edward Z. Yang, "PyTorch internals" (blog post, 2019). How PyTorch's strided tensors and views work.
- Next: [Chapter 4: Memory is the bottleneck](../04-memory/README.md). We measure the cache sizes, bandwidths and latencies that made column order 6x slower, and turn them into a model that predicts the speed of any kernel.
