# Chapter 9: Weights on disk

> **In one sentence:** a model's weights arrive as a file, and a good loader validates that file as untrusted input, then maps it into memory so the weights can be used in place, without reading or copying hundreds of megabytes up front.

**Where this fits:** until now every weight was made up. From here on they come from files, starting with a small model in chapter 10 and SmolLM2's real weights in chapter 16. This chapter builds the loader both use.

**You need:** chapter 2 (`bf16`, byte order), chapter 3 (borrowed views and lifetimes) and chapter 4 (page faults).

**You will build:** a parser and a writer for the safetensors format, written from the specification with every field validated; zero-copy `&[Bf16]` and `&[f32]` views into a memory-mapped file; a random-corruption test that the parser never panics; and measurements of reading, mapping and converting SmolLM2-135M's 269 MB weight file.

---

## 1. The intuition

A model file is like a warehouse with an index card at the front door. The card lists every item (tensor) with its type, its size and the aisle and shelf where it sits. The rest of the building is shelves, packed end to end.

There are two ways to use the warehouse. You can hire a truck, empty every shelf into your own storage room, and then work from there (`read` the whole file into memory). Or you can read the index card, and walk to a shelf only when you actually need what is on it (memory-map the file). The second way lets you start working almost immediately, and if several people use the same warehouse, they all share it instead of each renting their own storage room.

The index card is the part you must check carefully. It might be wrong by accident, or written by someone who wants to send you to a shelf that does not exist, or to tell you a small box is ten terabytes. Before trusting any entry, check that it makes sense.

**Where the analogy breaks:** walking to a shelf is not free even with the map. The first time you touch each 4 KB page of a mapped file, the operating system has to find it (in its cache, or on disk) and map it in: a page fault. Mapping moves the cost of loading from "before you start" to "the first time each page is used". For a server that is started once and serves for weeks, that is usually the right trade.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Checkpoint** | A file (or set of files) holding a model's trained weights. |
| **safetensors** | A simple, safe weight format from Hugging Face: a JSON header plus raw bytes. |
| **Pickle** | Python's object serialization. PyTorch's `.bin`/`.pt` files use it; loading one can run arbitrary code. |
| **GGUF** | llama.cpp's single-file format: metadata, tokenizer, and (often quantized) tensors. |
| **Header** | The part of a file that describes the rest. |
| **Little-endian** | Byte order where the least significant byte comes first. safetensors, x86 and ARM all use it. |
| **mmap** | Memory-mapping: making a file's contents appear in the process's address space, loaded lazily. |
| **Page cache** | The OS's in-memory cache of file contents, shared by all processes. |
| **Zero-copy** | Using data where it already is, without copying it into a new buffer. |
| **Alignment** | Whether an address is a multiple of a given number of bytes. |
| **Sharded checkpoint** | A large model split over several files, plus an index saying which tensor is in which file. |

## 3. The concepts in depth

### 3.1 Where weight files come from

Training frameworks save weights in their own formats. The ones you will meet:

- **PyTorch `.bin` / `.pt`**: a zip archive of Python pickles. Pickle can describe "call this function with these arguments" as part of loading, so **opening an untrusted `.bin` file can execute arbitrary code** on your machine. It also needs Python (or a pickle reimplementation) to read. Many models on the Hugging Face Hub still ship this way; newer ones mostly do not.
- **safetensors**: designed to fix exactly that. A header in JSON, then raw tensor bytes. Nothing in the file is ever executed, the format is trivial to parse in any language, and the tensor bytes are laid out so they can be used straight from a memory map. This is what SmolLM2 ships, and what this chapter implements.
- **GGUF**: llama.cpp's format. One file holds the architecture metadata, the tokenizer, and the tensors, usually in quantized block formats (chapters 18-19), with tensor data aligned to 32 bytes by default.
- **ONNX**: a computation graph plus weights, in Protocol Buffers. Used by ONNX Runtime and many deployment tools.

Large models are **sharded**: `model-00001-of-00004.safetensors` and so on, with a `model.safetensors.index.json` mapping each tensor name to its file. Each shard is an ordinary safetensors file.

### 3.2 The safetensors format, byte by byte

Part 1 of the demo writes a two-tensor file and dumps it:

```text
== 1. a two-tensor file: 212 bytes in total
   first 8 bytes (header length, little-endian): [176, 0, 0, 0, 0, 0, 0, 0] = 176
   header: {"__metadata__":{"note":"chapter 9 demo"},"layer.bias":{"data_offsets":[24,28],"dtype":"BF16","shape":[2]},"layer.weight":{"data_offsets":[0,24],"dtype":"F32","shape":[2,3]}}
   data section: 28 bytes
```

The whole format:

1. **8 bytes**: the header length N, an unsigned 64-bit little-endian integer. Here 176.
2. **N bytes**: a JSON object, UTF-8. Each key is a tensor name; each value gives the `dtype`, the `shape`, and `data_offsets`, a `[begin, end)` byte range *relative to the start of the data section*. An optional `__metadata__` key holds string-to-string metadata. Writers pad the JSON with spaces to a multiple of 8 bytes.
3. **The data section**: the raw bytes of every tensor, back to back, little-endian, row-major (chapter 3).

That is all. The official specification adds one rule that matters for security: the tensors must cover the data section exactly, with no gaps and no overlaps.

### 3.3 Treat the file as untrusted input

A model file may come from the internet, from a user upload, or from a disk that was half-written when the machine crashed. A loader that trusts the header can be made to do bad things:

| Hostile header | What a trusting loader does | Our check |
|---|---|---|
| Header length 2⁴⁰ | Allocates a terabyte, or reads past the end | Limit of 100 MB; header must fit in the file |
| Offsets past the end of the file | Reads out of bounds (in C: memory corruption) | Every range must lie within the data section |
| Shape `[2³², 2³²]` with 4 bytes of data | `2³² × 2³² × 4` overflows to 0 and "matches" | `checked_mul`: overflow is an error |
| Two tensors sharing bytes | Writing through one changes the other; confusing | Sorted ranges must be contiguous, no overlap |
| Bytes not covered by any tensor | Hidden data travels with the model | Ranges must cover the whole data section |
| Invalid UTF-8 / JSON | Parser crash | Explicit errors, never a panic |

The last line is the one people forget. In a server that loads user-supplied models (fine-tuning services, model hubs, "bring your own LoRA" endpoints), a panic in the parser is a crash that anyone can trigger. The test `random_corruption_never_panics` mutates a valid file 5,000 times at random (flipped bytes, truncations) and checks that parsing always returns either `Ok` or `Err`. This is a small version of **fuzzing**; chapter 30 points at proper fuzzing tools.

One thing this parser does *not* catch: a JSON object with the same key twice. `serde_json` keeps the last one silently. Exercise 1 asks you to fix it.

### 3.4 Memory-mapping

`std::fs::read(path)` allocates a buffer the size of the file and copies the file into it. For 269 MB that means 269 MB of fresh memory (page faults, chapter 4), a full copy, and waiting for all of it before doing anything.

`mmap` instead asks the OS to map the file into the process's address space. The call returns immediately, having read nothing. When the program first touches a byte of the mapping, the CPU faults, the OS finds that page (in the page cache, or on disk), maps it, and the program continues. From then on the page is ordinary memory.

Consequences for inference:

- **Start-up is nearly instant.** Measured below: 0.3-0.5 ms to map the file and parse its header, against 160-175 ms to `read` it (warm cache).
- **Pages you never touch are never loaded.** Rarely used experts in a mixture-of-experts model, for example, cost nothing until used.
- **Memory is shared.** Several processes (or a restart of the same one) mapping the same file use the *same* physical pages from the page cache. Eight worker processes serving one model need one copy of the weights, not eight.
- **The OS can evict pages under memory pressure** and re-read them later from the file, because they are clean copies of disk contents. Weights copied into your own buffers cannot be evicted this way (only swapped, if swap exists).

The costs: first access to each page is slow (and very slow from a cold disk), access patterns become disk access patterns, and the file must not change while mapped.

### 3.5 Why `mmap` is `unsafe` in Rust

`memmap2::Mmap::map(&file)` is an `unsafe` function. A `&[u8]` in Rust promises the bytes will not change while you hold it. A memory map cannot keep that promise on its own: another process could write to the file (the bytes change under you) or truncate it (touching the missing pages raises `SIGBUS` and kills the process). Rust cannot prevent that, so the caller must accept it. Every inference engine that maps its weights makes the same assumption: model files are read-only while in use. Our `MappedFile::open` states it in its `SAFETY` comment.

### 3.6 Zero-copy views, alignment and byte order

Once the file is mapped, a `bf16` tensor's bytes are already in exactly the format our kernels want (chapter 6's `dot_bf16` takes `&[Bf16]`). Viewing `&[u8]` as `&[Bf16]` without copying needs three things:

1. **Every bit pattern must be a valid value.** True for `Bf16`, `f32`, and the integer types. Not true for `bool` or `char`, or for enums, which is why this is `unsafe` in general. The `Plain` marker trait records which types qualify.
2. **The address must be aligned.** `Bf16` needs 2-byte alignment and `f32` 4-byte. A misaligned `&[f32]` is undefined behaviour in Rust, even on CPUs that tolerate misaligned loads.
3. **The byte order must match.** The file is little-endian. On a little-endian CPU (x86, ARM as configured by every mainstream OS) the bytes can be used as they are. On a big-endian machine each value would need its bytes swapped, so zero-copy is impossible there, and `reinterpret` returns `None`.

What alignment does SmolLM2's file give us? Part 3 of the demo checks all 272 tensors:

```text
   tensors whose data starts 2-byte aligned: 272, 4-byte: 272, 64-byte: 0 (of 272)
```

The mapping starts at a page boundary, the data section starts at byte 30,536, and every tensor's size is a multiple of 64 bytes. So every tensor starts at 30,536 mod 64 = 8 bytes past a cache line boundary. Zero-copy `&[Bf16]` views work for all 272 tensors, but every 64-byte AVX-512 load from them will straddle two cache lines: chapter 6's alignment penalty. The alternative is to copy each tensor into a 64-byte-aligned buffer at load time: 157-177 ms for the whole model on this machine (exercise 3), and the weights are then private memory rather than shared page cache. Whether the aligned copy pays for itself during inference is a question for measurement, and chapter 16 measures it.

### 3.7 Lifetimes and the self-referential struct problem

`SafeTensors<'a>` borrows the bytes it parsed, and every `TensorView<'a>` it hands out borrows them too. The compiler therefore guarantees that the `MappedFile` outlives every view into it: you cannot unmap the file while a kernel still holds a slice of its weights. In C++ that would be a use-after-unmap, which crashes (at best) the next time the kernel reads a weight.

The catch appears when you write a `Model` struct. You would like it to *own* the mapped file and *also* hold `&[Bf16]` slices into it:

```rust
struct Model {
    file: MappedFile,
    q_proj: &'??? [Bf16],   // borrows from `file`, which is in the same struct
}
```

This is a **self-referential struct**, and safe Rust cannot express it: moving `Model` would move `file` (the struct, not the mapping), and the compiler has no way to know that the slices point into the mapping rather than into the struct itself. There are four standard ways out:

1. **Let the model borrow.** `struct Model<'a> { q_proj: &'a [Bf16], ... }`, with the `MappedFile` owned by the caller (for example `main`) and outliving the model. Simple and zero-copy; the lifetime parameter then appears on everything that holds a `Model`.
2. **Store offsets, resolve on use.** Keep the mapping in the struct and store each tensor as a byte range; a method returns the slice when asked. Zero-copy, no lifetime parameters, a little bookkeeping.
3. **Copy into owned buffers.** Each weight becomes a `Vec` or `AlignedVec` owned by the model. No lifetimes, and you control alignment; you pay load time and lose page-cache sharing.
4. **A self-referential crate** such as `self_cell` or `ouroboros`, which package option 1 behind a safe API with some `unsafe` inside.

Chapter 16 uses option 3 for its default path (after measuring the alignment effect) and discusses when option 1 or 2 is the better choice. The point to take now is that the choice between zero-copy and owned memory is a real design decision with numbers on both sides, and Rust makes you decide it explicitly.

## 4. The code

The parser, writer and memory map are in [`src/lib.rs`](src/lib.rs); the demo is [`src/main.rs`](src/main.rs).

### 4.1 Parsing the length prefix and the header

<!-- file: src/lib.rs -->
```rust
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let available = bytes.len() as u64;
        let Some(prefix) = bytes.first_chunk::<8>() else {
            return Err(Error::Truncated {
                needed: 8,
                available,
            });
        };
        let header_len = u64::from_le_bytes(*prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(Error::HeaderTooLarge(header_len));
        }
        let data_offset = 8 + header_len;
        if data_offset > available {
            return Err(Error::Truncated {
                needed: data_offset,
                available,
            });
        }
```

- `parse(bytes: &'a [u8]) -> Result<Self, Error>`: the returned `SafeTensors<'a>` carries the lifetime of the input. It cannot outlive the bytes.
- `first_chunk::<8>()` returns `Some(&[u8; 8])` if there are at least 8 bytes, `None` otherwise. The `let ... else` form handles the short-file case up front.
- `u64::from_le_bytes` states the byte order explicitly. The format is little-endian on every machine, so the code says so rather than relying on the CPU's native order.
- Size checks happen in `u64` before anything is converted to `usize` or used as an index. `8 + header_len` cannot overflow because `header_len` is at most 100 MB by then.

<!-- file: src/lib.rs -->
```rust
        let text = std::str::from_utf8(header)
            .map_err(|e| Error::InvalidHeader(format!("not UTF-8: {e}")))?;
        let json: Value = serde_json::from_str(text.trim_end_matches(' '))
            .map_err(|e| Error::InvalidHeader(format!("not JSON: {e}")))?;
        let Value::Object(entries) = json else {
            return Err(Error::InvalidHeader("not a JSON object".into()));
        };
```

JSON parsing is not the lesson here, so we use `serde_json` and parse into its generic `Value` type (the header's keys are tensor names, not known in advance). Every failure becomes a variant of our own `Error` enum. `?` returns early with that error; nothing in the parser can panic on bad input.

### 4.2 Validating one tensor

<!-- file: src/lib.rs -->
```rust
    // checked_mul: a hostile shape like [2^40, 2^40] must not overflow into a
    // small, plausible-looking size.
    let bytes = shape
        .iter()
        .try_fold(dtype.size(), |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad("shape is too large"))?;
    if end - start != bytes {
        return Err(bad(&format!(
            "shape {shape:?} of {dtype:?} needs {bytes} bytes, offsets give {}",
            end - start
        )));
    }
```

`try_fold` multiplies the dimensions together, starting from the element size, and stops at the first `None`. `checked_mul` returns `None` instead of wrapping around on overflow. In release builds, plain `*` on `usize` wraps silently, so `2³² × 2³² × 4` would become 0 and match an empty byte range. This is the bug class that has produced real vulnerabilities in model loaders: size arithmetic on untrusted numbers must always be checked.

### 4.3 The whole data section, exactly once

<!-- file: src/lib.rs -->
```rust
    let mut ranges: Vec<(usize, usize, &str)> = tensors
        .iter()
        .map(|(name, t)| (t.start, t.end, name.as_str()))
        .collect();
    ranges.sort_unstable();
    let mut expected = 0;
    for (start, end, name) in ranges {
        if start != expected {
            return Err(Error::InvalidLayout(format!(
                "tensor {name:?} starts at byte {start}, expected {expected} (gap or overlap)"
            )));
        }
        expected = end;
    }
    if expected != data_len {
        return Err(Error::InvalidLayout(format!(
            "tensors cover {expected} bytes but the data section has {data_len}"
        )));
    }
```

Sort the ranges by start. Then each must begin exactly where the previous one ended, the first at 0 and the last ending at the data length. One pass checks all three rules: no gaps, no overlaps, nothing out of bounds. After this, every later `&self.data[info.start..info.end]` is guaranteed in bounds.

### 4.4 Handing out views that outlive the parser

<!-- file: src/lib.rs -->
```rust
    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, Error> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_owned()))?;
        Ok(TensorView {
            name: name.to_owned(),
            dtype: info.dtype,
            shape: info.shape.clone(),
            data: &self.data[info.start..info.end],
        })
    }
```

Note the return type: `TensorView<'a>`, not `TensorView<'_>`. The view borrows from the *file bytes* (lifetime `'a`), not from the `SafeTensors` struct. So a model loader can parse the header, take views of every tensor, and drop the `SafeTensors` index; the views stay valid as long as the mapped file does. Getting this lifetime right is what makes the parser usable in a real loader.

### 4.5 Zero-copy reinterpretation

<!-- file: src/lib.rs -->
```rust
pub unsafe trait Plain: Copy {}
// SAFETY: every 32-bit pattern is a valid f32 (possibly NaN); no padding.
unsafe impl Plain for f32 {}
// SAFETY: `Bf16` is `repr(transparent)` over `u16`; every pattern is valid.
unsafe impl Plain for Bf16 {}
```

An `unsafe trait` is one whose *implementations* carry a promise the compiler cannot check. Implementing `Plain` for a type says "any bytes are a valid value of this type". Getting that wrong (implementing it for `bool`, say) would make `reinterpret` unsound, which is why implementing it requires `unsafe impl` and a `SAFETY` comment. The `bytemuck` crate's `Pod` trait is the production version of this idea.

<!-- file: src/lib.rs -->
```rust
pub fn reinterpret<T: Plain>(bytes: &[u8]) -> Option<&[T]> {
    if cfg!(target_endian = "big") || !bytes.len().is_multiple_of(size_of::<T>()) {
        return None;
    }
    // SAFETY: `T: Plain` means every bit pattern is a valid `T`. `align_to`
    // only puts correctly aligned, whole elements in the middle slice, and we
    // accept the result only if nothing was left over at either end.
    let (head, body, tail) = unsafe { bytes.align_to::<T>() };
    (head.is_empty() && tail.is_empty()).then_some(body)
}
```

`slice::align_to::<T>()` splits a byte slice into an unaligned prefix, a middle part of properly aligned `T`s, and a leftover suffix. If the prefix and suffix are both empty, the whole slice was already aligned and a whole number of `T`s, and the middle part *is* the tensor, with no copy. Otherwise we return `None` and the caller decides what to do (copy, or refuse). The function never silently produces a misaligned reference.

`cfg!(target_endian = "big")` is a compile-time constant: on little-endian machines the whole condition folds away.

### 4.6 The portable, copying path

<!-- file: src/lib.rs -->
```rust
            Dtype::BF16 => Ok(pairs().map(|u| Bf16::from_bits(u).to_f32()).collect()),
```

`to_f32_vec` reads each element with `u16::from_le_bytes` from the raw bytes: correct on any machine, at any alignment, for `F32`, `BF16` and `F16`. It is the fallback when zero-copy is not possible, and the tests check that both paths give identical values.

### 4.7 Writing files

<!-- file: src/lib.rs -->
```rust
    let mut json = Value::Object(header).to_string().into_bytes();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + json.len() + offset);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    for t in tensors {
        out.extend_from_slice(t.data);
    }
```

Padding the header to a multiple of 8 means the data section starts 8-byte aligned in the file, so a memory map gives 8-byte-aligned tensors (for tensors whose sizes are multiples of 8). The official writer does the same. Chapter 10 uses this writer to save a model it trains.

## 5. Run it

```bash
./tools/download_model.sh          # once, for parts 3 and 4
cargo test -p ch09-safetensors
cargo run --release -p ch09-safetensors
```

On the reference machine:

```text
== 2. rejecting broken files
   truncated file                   -> file truncated: need 8 bytes, have 3
   header size of 2^40              -> header of 1099511627776 bytes exceeds the limit
   shape does not match byte range  -> tensor "w": shape [3] of F32 needs 12 bytes, offsets give 4
   two tensors overlap              -> invalid data layout: tensor "b" starts at byte 2, expected 4 (gap or overlap)

== 3. .../models/smollm2-135m-instruct/model.safetensors
   272 tensors, 134515008 parameters (134.5 M), dtypes {"BF16": 272}
   header: 30528 bytes; data starts at byte 30536
   first tensors:
     model.embed_tokens.weight                        BF16 [49152, 576]
     model.layers.0.input_layernorm.weight            BF16 [576]
     model.layers.0.mlp.down_proj.weight              BF16 [576, 1536]
     model.layers.0.mlp.gate_proj.weight              BF16 [1536, 576]
     model.layers.0.mlp.up_proj.weight                BF16 [1536, 576]
     model.layers.0.post_attention_layernorm.weight   BF16 [576]
   tensors whose data starts 2-byte aligned: 272, 4-byte: 272, 64-byte: 0 (of 272)

== 4. loading the weights (file already in the OS page cache)
   std::fs::read of 269 MB:         161.35ms  (1.7 GB/s)
   mmap + parse header:               387.12µs
   first touch of every page:          13.68ms
   zero-copy &[Bf16] views:            51.60µs  (272 of 272 tensors)
   convert everything to Vec<f32>:       1.62s  (538 MB allocated)
```

What the numbers say:

- **SmolLM2-135M is 272 tensors, all `bf16`, 134.5 million parameters.** Names follow the Hugging Face convention (`model.layers.N.self_attn.q_proj.weight`), and chapter 16 maps each one onto our model. There is no separate output-layer weight: SmolLM2 reuses the embedding matrix ("tied embeddings").
- **Mapping and parsing takes under half a millisecond**; reading the file takes 160-175 ms even when it is already cached in memory, because `read` must allocate 269 MB and copy every byte.
- **Touching every page of the mapping takes 14 ms**: 65,700 page faults from the page cache, about 210 ns each.
- **Zero-copy views of all 272 tensors take 52 µs** in total: no data moves.
- **Converting everything to `f32` is the slowest option by far** and doubles the memory. Its time varied between 0.4 and 2.5 s across runs on this VM. Most of it is page faults on 538 MB of fresh memory, and in a virtual machine the *first* touch of guest memory is especially slow, because the hypervisor has to back each page on first use. That is also why the very first `fs::read` of the file in a fresh session took 1.12 s: fresh file cache, fresh memory.

## 6. The Rust behind it

**Lifetimes make zero-copy safe.** `SafeTensors<'a>` and `TensorView<'a>` cannot outlive the bytes they borrow. With a memory map, that means no kernel can ever read a weight after the file is unmapped. This is the same mechanism as chapter 3's views, now protecting against a much more painful bug.

**`Result` and a custom error enum for untrusted input.** Every way the file can be wrong is a variant of `Error`, with a `Display` implementation for readable messages and `std::error::Error` so it composes with `?` and `Box<dyn Error>`. Library code returns errors; only the demo's `main` decides to `expect`.

**Checked arithmetic on untrusted sizes.** `checked_mul`, `try_from`, and comparisons in `u64` before casting to `usize`. In release builds, integer overflow wraps silently, so unchecked arithmetic on sizes from a file is a security bug waiting to happen.

**`unsafe trait` and `unsafe impl`.** `Plain` shows the other side of `unsafe`: sometimes the promise is made by whoever *implements* a trait, not by whoever calls a function. `Send` and `Sync` (chapter 7) work the same way.

**`cfg!` versus `#[cfg]`.** `cfg!(target_endian = "big")` is an expression that is `true` or `false` at compile time; both branches must compile. `#[cfg(...)]` removes code entirely (chapter 6's per-architecture kernels). Use `cfg!` when both branches are valid code everywhere.

**Slice patterns.** `let [start, end] = offsets.as_slice() else { ... }` matches exactly two elements and rejects any other length, in one line.

## 7. Mistakes you will make

- **Loading `.bin` files from untrusted sources.** Use safetensors, or at least PyTorch's `weights_only=True` loading mode.
- **Reading offsets as relative to the file start.** They are relative to the start of the data section, 8 + N bytes in.
- **Using native byte order** (`from_ne_bytes`, or a raw pointer cast) instead of little-endian. It works on your machine and silently breaks on a big-endian one.
- **Assuming alignment.** A cast from `*const u8` to `*const f32` on an unaligned address is undefined behaviour even if the CPU tolerates it. Use `align_to` or copy.
- **Unchecked size arithmetic** on header values.
- **Holding the mapped file in a struct alongside views into it.** It will not compile; pick one of the four designs in section 3.7 instead of fighting the borrow checker.
- **Modifying a model file while a server has it mapped** (for example, copying a new version over the old path). The running server may crash with `SIGBUS` or read a mix of old and new weights. Write the new file under a new name and rename it into place, or restart.

## 8. How the professionals do it

- **Hugging Face's `safetensors` crate** is the reference Rust implementation: the same validation rules, zero-copy views, and `serialize_to_file`. Python's `safetensors` package is a wrapper around it.
- **`candle`** loads weights through `safetensors` and memory maps, with a `VarBuilder` that looks tensors up by name, exactly the pattern chapter 16 uses.
- **llama.cpp** memory-maps GGUF files by default (`--no-mmap` turns it off) and can `mlock` the mapping to prevent the OS from evicting weights under memory pressure.
- **vLLM and TensorRT-LLM** load safetensors shards in parallel, often straight into GPU memory, and for multi-GPU inference each GPU loads only its slice of each tensor (chapter 28).
- **Model hubs scan uploaded files** for pickle payloads and convert checkpoints to safetensors, because loading untrusted pickles has led to real compromises.

## 9. Exercises

1. **Duplicate keys.** Build a header with the same tensor name twice (write the JSON string by hand). What does `parse` do? Make it reject duplicates. Hint: parse the header twice, once as a `Value` and once counting keys with a streaming approach, or deserialize into a type that errors on duplicates.
2. **Sharded checkpoints.** Write `load_sharded(index_path)` that reads a `model.safetensors.index.json` (`{"weight_map": {"tensor.name": "model-00001-of-00002.safetensors", ...}}`), maps each shard once, and returns a lookup from tensor name to `TensorView`. What lifetime does the result have, and who owns the maps?
3. **Aligned copies.** Copy every SmolLM2 tensor into a chapter 6 `AlignedVec<Bf16>`. How long does it take, and how much memory does it use compared with the zero-copy views?
4. **Big-endian.** On a big-endian machine, what do `as_bf16` and `to_f32_vec` return for SmolLM2's file? Why is one of them `None` and the other still correct?
5. **GGUF.** Read the GGUF specification (in the ggml repository) and list three differences from safetensors that matter to an inference engine.
6. **Cold cache.** If you have root on a Linux machine, run `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` and rerun part 4. Which numbers change, and why?

## 10. Check yourself

1. What are the three parts of a safetensors file?
2. Why is loading a PyTorch `.bin` file from an untrusted source dangerous, and why is safetensors not?
3. Why must the shape-times-dtype-size computation use checked multiplication?
4. What does `mmap` do at the moment you call it, and when is the file actually read?
5. Why is `Mmap::map` an `unsafe` function?
6. Why can't a struct own a memory-mapped file and also hold `&[Bf16]` slices into it?
7. Every SmolLM2 tensor starts 8 bytes past a 64-byte boundary. Why does that matter for chapter 6's AVX-512 kernel?

## 11. Recap

- safetensors = 8-byte header length + JSON header + raw little-endian bytes. Simple, safe, and designed for memory mapping.
- Validate everything in the header: sizes against limits, offsets against the file, shapes with checked arithmetic, and full, non-overlapping coverage of the data section. Never panic on bad input.
- Memory-mapping makes start-up nearly free (0.4 ms against 160+ ms for `read` here), shares weights between processes, and moves page-fault costs to first use.
- Zero-copy views need a type where every bit pattern is valid, correct alignment, and little-endian byte order. `align_to` checks alignment without guessing.
- Lifetimes guarantee no view outlives the mapping. The flip side is the self-referential struct problem, which forces an explicit choice between borrowing, offsets, owned copies and helper crates.
- SmolLM2's tensors are 4-byte but not 64-byte aligned in the file; whether copying them into aligned buffers pays off is measured in chapter 16.

## Answers

**Exercises**

1. `serde_json` keeps the last occurrence and silently drops the first, so the file parses and one tensor's bytes become "unused". Our layout check then reports a gap, which catches this particular case by accident, but a crafted file could make the duplicate's range coincide with another tensor and pass. A robust fix: deserialize the header with a custom `Visitor` (or a `serde_json::Deserializer` with a map type that errors on duplicate insertion) so a repeated key is an explicit `InvalidHeader` error.
2. Parse the index JSON, collect the set of shard file names, map each one once into a `Vec<MappedFile>` (or a `HashMap<String, MappedFile>`) owned by the caller, then parse each map and build `HashMap<String, TensorView<'a>>` where `'a` is the lifetime of that collection of maps. The maps must outlive the lookup table, which is exactly the "let the model borrow" design of section 3.7. Alternatively, store `(shard index, TensorInfo)` pairs and resolve views on demand.
3. Measured on the reference machine: 934 ms on the first run (fresh VM memory), then 157-177 ms. It uses 269 MB of private memory on top of the mapped file (whose pages the OS can drop once nothing touches them), against essentially zero for the views.
4. `as_bf16` returns `Ok(None)` because `reinterpret` refuses to reinterpret little-endian bytes on a big-endian machine. `to_f32_vec` still returns correct values because it decodes each element explicitly with `u16::from_le_bytes`, which is correct on any machine.
5. Among others: GGUF stores the tokenizer and all architecture hyperparameters in typed metadata key-value pairs, so one file is enough to run the model; it supports quantized block types (Q4_0, Q8_0, K-quants...) whose element size is not a whole number of bytes; and it aligns every tensor's data to a configurable boundary (32 bytes by default), which helps SIMD kernels.
6. Measured on the reference machine (which allows dropping the cache): `fs::read` rose from about 165 ms to 399 ms (0.7 GB/s, the speed of the VM's disk), and the later steps did not change, because `fs::read` had just pulled the whole file into the cache. Measuring the memory map on its own after dropping the cache: map and parse took 2.5 ms (reading the first 30 KB from disk), and touching every page took 255 ms, against 7 ms warm. Mapping does not remove the disk reads; it moves them from start-up to the first time each page is used, which in a server means the first requests.

**Check yourself**

1. An 8-byte little-endian header length, a JSON header describing each tensor (dtype, shape, byte range), and the raw tensor bytes.
2. PyTorch `.bin` files are pickles, and unpickling can call arbitrary functions, so a malicious file can run code on your machine. safetensors contains only data and a JSON description; nothing in it is executed.
3. Because the dimensions come from an untrusted file. Unchecked multiplication can wrap around in release builds and produce a small size that passes validation for a tensor that is actually enormous.
4. It sets up the mapping and returns, reading nothing. Each page is read (from the page cache or the disk) the first time it is touched.
5. Because another process can modify or truncate the file while it is mapped, changing bytes behind a `&[u8]` that Rust assumes are immutable, or making accesses crash. The caller has to promise that does not happen.
6. The struct would contain references into its own field. Safe Rust cannot express a lifetime that refers to another field of the same struct, since moving the struct would appear to invalidate them.
7. A 64-byte AVX-512 load from an address that is not a multiple of 64 spans two cache lines and costs roughly twice as much, so an AVX-512 kernel reading these tensors in place pays that penalty on every load.

## Further reading

- The safetensors repository and its README (github.com/huggingface/safetensors), including the specification and the security audit.
- The GGUF specification in the ggml repository (`docs/gguf.md`).
- The `memmap2` crate documentation, and `man 2 mmap`.
- Next: [Chapter 10: A first model, end to end](../10-first-model/README.md). We train a small model, save it with this chapter's writer, load it back and serve it.
