# Chapter 16: Running a real model

> **In one sentence:** running a published model means reading three files exactly as their authors meant them (`config.json` for the shape, `model.safetensors` for the weights, `tokenizer.json` for the text encoding), plus a chat template that is written down nowhere but in a Jinja string, and proving the result right by comparing it, number by number and token by token, with the reference implementation.

**Where this fits:** chapters 11-15 built every piece with random weights and toy vocabularies. This chapter plugs in SmolLM2-135M-Instruct, a real chat model, and from here on every measurement in the course is on real weights. Chapter 17 profiles this model, chapters 18-19 quantize it, chapters 21-22 serve it.

**You need:** chapter 9 (safetensors, memory maps), chapter 11 (BPE), chapter 14 (the engine), chapter 15 (sampling). And the model: `./tools/download_model.sh`.

**You will build:** a `config.json` reader that refuses what it cannot run, a `bf16` weight matrix that can live in a memory-mapped file or in aligned memory, a checkpoint loader that checks every tensor, a `tokenizer.json` reader that matches Hugging Face token for token, the chat template, a command-line chat, and tests against fixtures generated with PyTorch.

---

## 1. The intuition

Think of receiving a piece of flat-pack furniture from another country. The parts are all in the box, but you also need the assembly instructions in a language you can read, the exact screw sizes, and the knowledge that "tighten" means a quarter turn and not as hard as you can. Get one of those wrong and you still end up with something that looks like a chair. It just wobbles.

Running someone else's model is the same. The weights are the parts. The config is the list of dimensions. The tokenizer is the part everyone underestimates: if your tokenizer splits text even slightly differently from the one used in training, the model receives input it has never seen, and it gets quietly worse. Nothing crashes. That is why this chapter's most important output is not the chat, it is the tests.

**Where the analogy breaks:** a wobbly chair is visible. A model with a slightly wrong tokenizer, a transposed weight or the wrong RoPE layout still writes fluent text. The only reliable check is a reference implementation run on the same inputs, compared numerically.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Checkpoint** | The files of a trained model: weights plus the configuration needed to use them. |
| **`config.json`** | Hugging Face's description of a model's architecture: sizes, counts, θ, ε, special token ids. |
| **`tokenizer.json`** | Hugging Face's tokenizer file: vocabulary, merges, pre-tokenizer, special tokens. |
| **Chat template** | The rule for turning a list of messages into one prompt string, stored as a Jinja template. |
| **Instruct / chat model** | A model fine-tuned to answer in turns, using the template's special tokens. |
| **Special (added) token** | A token like `<|im_end|>` matched literally in text, never split or merged. |
| **Byte-to-unicode table** | GPT-2's mapping that gives each byte a printable character so tokens can be stored as JSON strings. |
| **Golden fixtures** | Reference outputs, generated once with the trusted implementation and committed next to the tests. |
| **Zero-copy loading** | Using weights straight from a memory-mapped file instead of copying them. |
| **Resident memory (RSS)** | The physical memory a process currently occupies, split into anonymous (its own) and file-backed (mapped files). |

## 3. The concepts in depth

### 3.1 The three files, and what each can get wrong

| File | Tells you | Typical silent mistake |
|---|---|---|
| `config.json` | layers, sizes, heads, KV heads, θ, ε, tied embeddings, special ids | ignoring a field that changes the math (`rope_scaling`, biases, `head_dim`) |
| `model.safetensors` | every weight, with its type and shape | a tensor loaded into the wrong slot, or a shape that happens to match anyway |
| `tokenizer.json` | vocabulary, merges, pre-tokenization, special tokens | a pre-tokenizer detail that differs on 1% of inputs |

The loader's policy follows from that column: **read everything that changes the computation, and refuse anything you do not implement.** `parse_config` returns `Unsupported` for another architecture, another activation, RoPE scaling or bias terms. The weight loader checks every tensor's type and shape against the config, and afterwards checks that the file contains no tensor it did not use: an extra tensor means the checkpoint has a part this engine would silently ignore. An engine that runs a model slightly wrong is worse than one that refuses to run it.

### 3.2 Weights: memory-mapped or copied

SmolLM2's weights are 269 MB of `bf16`. Chapter 9 memory-mapped the file and found that every tensor starts 8 bytes past a 64-byte boundary. Chapter 6 found that the AVX-512 kernel ran at half speed on data misaligned like that. So there are two reasonable designs:

- **Mapped:** each `DenseBf16` keeps a shared handle to the mapped file and the tensor's offset. Nothing is copied; the operating system reads pages from disk the first time they are touched, and the pages belong to the page cache, shared with every other process that maps the same file.
- **Aligned:** copy each tensor into 64-byte-aligned memory, paying time and a private copy for aligned loads.

The chapter 6 result says aligned should be faster. The `bench` command measures both, plus a third option, every weight converted to `f32` (twice the bytes):

```text
SmolLM2-135M-Instruct, 4 threads
weights       load 1st prefill    prefill prefill tok/s decode tok/s      GB/s   RSS anon   RSS file
mapped         8ms       140ms    130.5ms           307         97.5      26.2      19 MB     260 MB
aligned      153ms       133ms    123.0ms           325         92.1      24.8     277 MB       4 MB
f32             1s       158ms    114.2ms           350         64.7      34.8     546 MB       4 MB
```

(Prefill is the 40-token chat prompt; decode is the median of 64 steps; the file was already in the page cache. Over four runs, decode measured 82-98 tokens/s mapped, 70-92 aligned and 53-65 in `f32`; prefill 290-410 tokens/s for all three.)

What the numbers say:

- **Alignment mattered far less than chapter 6 suggested.** Over seven paired runs, the aligned 135M model decoded no faster than the mapped one (slower in five, equal in one, faster in one; means 81 against 85 tokens/s). For the 360M model (section 9, exercise 1) the aligned copy was faster in four of five runs, by 12% on average. Nowhere near chapter 6's factor of 2: that penalty was measured on data in the cache, where the loads themselves are the bottleneck, while a decode step streams hundreds of megabytes from DRAM and mostly waits for memory. Prefill, which does work on cached data, showed no consistent difference either, probably because the `bf16` kernel converts every value before using it and so does more work per byte loaded than chapter 6's `f32` kernel. This is why you measure the real workload instead of carrying a micro-benchmark's conclusion over to it, and measure it more than once.
- **Mapping is free to load.** 8 ms, against 153 ms to copy. The mapped weights appear as "RSS file": memory the kernel can drop under pressure and read back from disk, and share between processes. Four server processes mapping the same file use one copy of the weights. The aligned copy is private, anonymous memory in every process.
- **`bf16` decodes 1.5 times faster than `f32`, not 2 times.** Halving the bytes should halve the time of a memory-bound step. But the `bf16` runs moved 19-26 GB/s against 28-35 GB/s for `f32`: the `bf16` kernel does not keep the memory system as busy. Per byte of weights it does twice the multiply-adds of the `f32` kernel, plus a widening of every value, so each core consumes bytes more slowly. Chapter 17 profiles the decode step in detail.
- **Prefill is compute-bound**, so `bf16` weights (which must be converted before every multiply) do not help it; `f32` was slightly faster in every run.
- **Converting to `f32` took 1-3 seconds.** This loader converts element by element into a temporary vector and then copies it into aligned memory, touching about 1 GB of fresh memory (exercise 4 removes the copy). It is a one-time cost and the least important number in the table.

The "RSS anon" of 19 MB for the mapped model is smaller than the scratch buffers it allocated (the logits buffer alone is 48 MB for 256 rows). Memory from `vec![0.0; n]` is requested zeroed from the operating system, which hands out pages only when they are first written, and this benchmark only ever writes the first row of logits.

### 3.3 The tokenizer, exactly

Chapter 11's BPE is the algorithm. `tokenizer.json` adds five details, and each one changes token ids on some input:

1. **Tokens are stored as characters, not bytes.** GPT-2's byte-to-unicode table gives each of the 256 bytes a printable character, so that tokens can be JSON strings: printable ASCII stands for itself, a space is `Ġ`, a newline `Ċ`. The loader inverts the table to get each token's bytes.
2. **Merges are listed in rank order** as `"Ġ t"` (older files) or `["Ġ", "t"]` (newer). Rank is the position in the list.
3. **Pre-tokenization uses GPT-2's pattern**, with contractions (`'s`, `'ll`...), and one alternative, `\s+(?!\S)`, that the Rust `regex` crate cannot express because it has no look-ahead. It gives the last space of a whitespace run to the following word: `"a   b"` becomes `"a"`, `"  "`, `" b"`. `split_words` reproduces it by hand (section 4.3).
4. **SmolLM2 splits digits first.** Its `Digits` pre-tokenizer makes every digit a piece of its own, so `2024` is always four tokens (a common choice, meant to help models with arithmetic).
5. **Special tokens** like `<|im_start|>` are found first and never pass through BPE.

And one surprise, found by a test: **SmolLM2's vocabulary has no token for 21 of the 256 bytes** (six control characters, and bytes like 0xC0, 0xC1 and 0xF5-0xFF that never occur in valid UTF-8). Byte-level BPE is supposed to encode anything; this vocabulary cannot. Hugging Face's tokenizer, with no "unknown" token to fall back on, silently drops those bytes: `"a\x04b"` encodes to the tokens for `a` and `b`. Matching the reference means dropping them too, and the fixtures include a control character to prove we do.

How do you know you got all of it right? `tools/make_fixtures.py` runs Hugging Face's own tokenizer on 25 awkward strings (runs of whitespace, `\r\n`, digits, contractions, accents, Japanese, Cyrillic, Arabic, emoji, code, special tokens, a control byte) and on 42 KB of a novel (10,740 tokens), and stores the ids. The test compares every id. It passes; it did not pass the first time.

### 3.4 The chat template

An instruct model was fine-tuned on conversations in one exact format, and it answers well only in that format. SmolLM2's is ChatML:

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

The prompt ends with the start of the assistant's turn, so the most likely continuation is the answer, and the model ends it with `<|im_end|>` (id 2, the config's `eos_token_id`), which is where generation stops.

The format is defined by a Jinja template stored as a string in `tokenizer_config.json`. It adds a default system message when the conversation does not start with one. `chat_prompt` reimplements it in 20 lines; the test checks that its output tokenizes to exactly the ids Hugging Face's template produces. Every model family has its own template (Llama 3 uses `<|start_header_id|>`, Mistral `[INST]`), and getting it wrong is one of the most common reasons a correctly loaded model "is dumb".

### 3.5 Proving it right

Section 3.1's worry, a model that runs fluently but wrong, is answered by comparing against PyTorch on the same inputs:

| Test | Compares | Result |
|---|---|---|
| `tokenizer_matches_hugging_face_token_for_token` | 26 texts, including 10,740 tokens of a novel | identical ids; decoding gives the text back |
| `the_chat_prompt_tokenizes_like_the_reference` | our template + tokenizer vs Hugging Face's | identical ids |
| `logits_match_pytorch` | all 49,152 logits at the last prompt position | largest difference 6.5e-5 (logits up to 31.3) |
| `greedy_generation_matches_pytorch` | greedy answers to two questions, 8 and 48 tokens | identical tokens |
| `mapped_aligned_and_f32_weights_agree` | the three weight placements | mapped and aligned bit-identical; `f32` within 1e-3 |
| `mapped_weights_are_where_the_file_puts_them` | the address of every matrix | mapped: 8 bytes past a cache line; aligned: 0 |

A difference of 6.5e-5 is the size of rounding differences between two correct `f32` implementations that add in different orders. Real bugs are not subtle in this comparison, even when they are subtle in the output. Loading the model with the other RoPE layout (chapter 12's interleaved pairs instead of halves) makes the largest logit difference 14.7, and the model still answers in fluent English: asked why the sky is blue, it says "The sky is a person who loves to help you." Only the numbers give it away. Chapter 30 uses the fixture's per-layer hidden states to find *which* layer a bug is in.

The reference runs in `float32`. Our engine also computes in `f32` with `bf16` weights converted exactly, so both run the same arithmetic with the same weights, and the comparison can be tight. Comparing against a `bf16` PyTorch run would need a much looser tolerance, and would hide small bugs.

### 3.6 Talking to it

```text
$ cargo run --release -p ch16-real-model -- ask "Explain in two sentences why the sky is blue."
[loaded mapped weights in 8ms]
The sky appears blue because the Earth's atmosphere scatters sunlight in all directions, including blue light, which is scattered more than other colors by large molecules like water droplets in clouds. This scattering effect is known as Rayleigh scattering. As a result, blue light is scattered in all directions, while other colors are scattered in only one direction, resulting in the blue color.
[40 prompt tokens, 74 new; first token after 186ms, then 78.4 tokens/s; finish: StopToken]
```

It answers in two sentences... and then a third, with some physics that is wrong (clouds do not make the sky blue; Rayleigh scattering is by molecules much smaller than the wavelength). That is the model, not the engine: the first 48 tokens of this answer are one of the test fixtures, identical to PyTorch's. A 135-million-parameter model is small. It is fluent, fast and often wrong.

With sampling, the same question gives a different answer per seed:

```text
$ cargo run --release -p ch16-real-model -- ask --temperature 0.7 --top-p 0.9 --seed 4 "Write a haiku about the sea."
The salty waves crash against the shore,
A reminder of the endless sea,
A reflection of the vast, unforgiving blue,
A beauty that never fails.
[38 prompt tokens, 35 new; first token after 151ms, then 70.1 tokens/s; finish: StopToken]
```

(Not a haiku either.) The `chat` command keeps the conversation and re-encodes all of it every turn:

```text
> My name is Ada.
I'm Ada, a kind and gentle AI assistant who has been listening to your concerns and offering support. [...]
[35 prompt tokens, 46 new; first token after 163ms, then 79.4 tokens/s; finish: StopToken]
> What is my name?
My name is Ada, and I'm a kind and gentle AI assistant [...]
[97 prompt tokens, 50 new; first token after 361ms, then 73.3 tokens/s; finish: StopToken]
```

The model takes the user's name as its own (PyTorch gives the same first answer, word for word). Note the second turn's prompt: 97 tokens, because the whole conversation is processed again, and the time to first token grows with it. Chapter 24 keeps the cache from one turn to the next instead.

The tokens/s printed by `ask` (70-83 in repeated runs) is lower than `bench`'s median because it is the mean over all steps, including the slow ones this shared VM produces.

## 4. The code

[`src/config.rs`](src/config.rs) reads the config, [`src/weights.rs`](src/weights.rs) the weights, [`src/tokenizer.rs`](src/tokenizer.rs) the tokenizer, [`src/chat.rs`](src/chat.rs) holds the template, [`src/main.rs`](src/main.rs) the command line, [`tests/reference.rs`](tests/reference.rs) the comparisons, and [`fixtures/`](fixtures/) the reference data (regenerated by [`tools/make_fixtures.py`](../../tools/make_fixtures.py), see [`tools/README.md`](../../tools/README.md)).

### 4.1 Refusing what we cannot run

<!-- file: src/config.rs -->
```rust
    let unsupported = |what: String| Err(Error::Unsupported(what));
    if !hf.architectures.iter().any(|a| a == "LlamaForCausalLM") {
        return unsupported(format!("architectures {:?}", hf.architectures));
    }
    if hf.hidden_act != "silu" {
        return unsupported(format!("activation {:?}", hf.hidden_act));
    }
    if hf.rope_scaling.as_ref().is_some_and(|v| !v.is_null()) {
        return unsupported("rope_scaling (e.g. Llama 3.1's long-context RoPE)".into());
    }
    if hf.attention_bias || hf.mlp_bias {
        return unsupported("bias terms in attention or MLP".into());
    }
```

`HfConfig` is a `serde` struct with only the fields the engine reads; `serde` ignores the rest. Optional fields have the defaults Hugging Face uses (`num_key_value_heads` defaults to one per query head, `head_dim` to `hidden_size / num_attention_heads`), and `eos_token_id`, which is a number in some configs and a list in others, is an `#[serde(untagged)]` enum that accepts both.

### 4.2 A `bf16` matrix in two places

<!-- file: src/weights.rs -->
```rust
    pub fn values(&self) -> &[Bf16] {
        match &self.data {
            Bf16Data::Mapped { file, start } => {
                let bytes = &file.bytes()[*start..*start + self.rows * self.cols * 2];
                reinterpret(bytes).expect("alignment was checked when loading")
            }
            Bf16Data::Aligned(values) => values,
        }
    }
```

A mapped matrix holds an `Arc<MappedFile>` and a byte offset, not a slice. A slice would borrow from the file, and a struct that owns the file *and* borrows from it is self-referential, which Rust does not allow (section 6). Holding a shared handle plus an offset avoids the problem: each matrix keeps the mapping alive, and the slice is rebuilt on each use, which costs an addition and an alignment check per matrix multiplication.

Everything else is chapter 14's `Matrix` trait with chapter 6's `dot_bf16` as the kernel:

<!-- file: src/weights.rs -->
```rust
        let bytes = self.bytes(name, &[rows, cols])?;
        let values: &[Bf16] = reinterpret(bytes)
            .ok_or_else(|| Error::Tensor(format!("{name} is not 2-byte aligned in the file")))?;
        let data = match placement {
            Placement::Mapped => Bf16Data::Mapped {
                file: Arc::clone(self.file),
                // Offset of the tensor inside the file.
                start: bytes.as_ptr() as usize - self.file.bytes().as_ptr() as usize,
            },
            Placement::Aligned => Bf16Data::Aligned(AlignedVec::from_slice(values)),
        };
```

`self.bytes` has already checked the dtype and shape. `reinterpret` (chapter 9) checks that the bytes can be viewed as `Bf16`, which needs 2-byte alignment; the file's layout guarantees it, and the check turns a violation into an error instead of undefined behaviour.

### 4.3 The look-ahead the regex crate lacks

<!-- file: src/tokenizer.rs -->
```rust
    fn split_words<'t>(&self, text: &'t str, words: &mut Vec<&'t str>) {
        let mut pos = 0;
        while pos < text.len() {
            let m = self
                .pattern
                .find_at(text, pos)
                .expect("every character matches some alternative");
            let mut end = m.end();
            let run = &text[pos..end];
            if end < text.len()
                && run.chars().all(char::is_whitespace)
                && let Some((last, _)) = run.char_indices().last()
                && last > 0
            {
                end = pos + last;
            }
            words.push(&text[pos..end]);
            pos = end;
        }
    }
```

The pattern's alternatives cover every character (letters, numbers, whitespace, everything else), so each search matches at `pos`. When the match is a whitespace run that is followed by more text and is longer than one character, the loop gives its last character back, which is exactly what `\s+(?!\S)` followed by `\s+` does in the original. The `regex` crate leaves out look-around deliberately: without it, every search runs in time linear in the input, a guarantee a server that tokenizes untrusted text should want.

The words are `&str` slices of the input, so splitting allocates nothing but the vector of slices. The lifetime `'t` in the signature says so: the words borrow from `text`, not from the tokenizer.

### 4.4 Loading, checking every tensor

<!-- file: src/weights.rs -->
```rust
    // A tensor nobody asked for means the checkpoint has parts this engine
    // does not know about (biases, extra norms...): refuse rather than
    // silently compute something else.
    if let Some(extra) = ck.tensors.names().find(|n| !ck.used.contains(*n)) {
        return Err(Error::Unsupported(format!("unexpected tensor {extra}")));
    }
```

`load` is generic over the matrix type and takes a closure that builds one matrix from a tensor name and shape; `load_bf16` and `load_f32` differ only in that closure. Every tensor it reads is recorded in `used`, and a leftover tensor is an error. For SmolLM2, the 272 tensors of the file are exactly the 272 the loader asks for.

### 4.5 A command line that stops when you stop reading

The `turn` function in `main.rs` streams tokens through chapter 11's `StreamDecoder` (so a multi-byte character split across tokens is never printed half) and chapter 15's `generate`. Its callback returns `ControlFlow::Break(())` when writing to standard output fails, which is what happens when the reader goes away (`... | head -1`). Chapter 21 uses the same mechanism to stop generating for a client that disconnected.

`ask`, `chat` and `bench` must work with `Model<DenseBf16>` and `Model<DenseF32>`, two different types chosen at run time. `with_model` loads the right one and calls a generic method on a small trait, `WithModel`, compiled once per weight type. A closure cannot do this, because a closure cannot be generic.

## 5. Run it

```bash
./tools/download_model.sh                    # once, about 270 MB
cargo test -p ch16-real-model                # the comparisons with PyTorch
cargo run --release -p ch16-real-model -- ask "What is the capital of France?"
cargo run --release -p ch16-real-model -- chat
cargo run --release -p ch16-real-model -- bench
```

Without the model, the tests that need it print a skip message and pass; the config and chat-template unit tests still run. `bench` also measures the tokenizer:

```text
tokenizer: loaded in 61ms; 42509 bytes -> 10740 tokens in 7.6ms (5.6 MB/s, 3.96 bytes per token)
```

5.6 MB/s is about 1.4 million tokens per second, far faster than the model can use them: a 4,000-token prompt tokenizes in about 3 ms and takes more than 10 seconds to prefill here. Tokenization is not a bottleneck for a single sequence, and a server can run it on another thread.

## 6. The Rust behind it

**Self-referential structs, and the `Arc` + offset way out.** A struct holding a `MappedFile` and a `&[Bf16]` into it would borrow from itself. Rust forbids it: moving the struct would move the file handle, and the compiler cannot prove the slice stays valid. The usual solutions are to keep the owner outside (the model borrows from a file that lives longer, `Model<'a>`, which infects every type that holds a model with a lifetime), to leak the owner (`Box::leak` gives a `&'static`, fine for a process that loads one model forever), or, as here, to share ownership with `Arc` and store offsets instead of references. Crates like `ouroboros` and `self_cell` package the unsafe version.

**Higher-ranked closures.** `load(dir, Checkpoint::f32)` does not compile, with the error "implementation of `FnMut` is not general enough". `load` asks for a function that accepts `&mut Checkpoint<'a>` for *every* lifetime `'a` (a higher-ranked bound, `for<'a> FnMut(&mut Checkpoint<'a>, ...)`), and the path `Checkpoint::f32` names the method of one particular `Checkpoint<'x>`. A closure, `|ck, name, rows, cols| ck.f32(name, rows, cols)`, is inferred as generic over the lifetime and works. Clippy suggests turning the closure back into the path; the code keeps the closure with an `#[expect]` explaining why.

**`serde` for configs.** `#[derive(Deserialize)]` on a struct with only the needed fields, `Option<T>` for optional ones, `#[serde(default)]` or `#[serde(default = "function")]` for defaults, `#[serde(untagged)]` for "a number or a list". The generated code validates types as it parses, so a string where a number belongs is an error with a line and column.

**An error enum with `From`.** `Error` lists what can go wrong while loading; `impl From<ch09_safetensors::Error> for Error` lets `?` convert chapter 9's errors automatically; `source()` keeps the underlying cause for anyone printing an error chain. The command line boxes everything as `Box<dyn std::error::Error>`: a library should say precisely what went wrong, an application mostly needs to report it.

**Traits where closures cannot go.** `WithModel::run<W: Matrix>` is a generic method on a trait. Closures cannot have type parameters, so "call this code with whichever model type was loaded" needs a trait (or a macro).

## 7. Mistakes you will make

- **Trusting that fluent output means correct output.** It does not. Compare with the reference on the same token ids.
- **Comparing against a `bf16` or `float16` reference** with a tight tolerance (it fails) or a loose one (it hides bugs). Run the reference in `float32`.
- **Approximating the tokenizer**: chapter 11's simplified pre-tokenizer, no digit splitting, or `\s+` without the look-ahead. The first 100 test strings may pass.
- **Forgetting the chat template**, or the default system message, or the final `<|im_start|>assistant\n`. The model then continues the user's message instead of answering it.
- **Printing the stop token**, or stopping on the wrong one (`<|endoftext|>`, id 0, instead of `<|im_end|>`, id 2). Take the stop tokens from the config.
- **Ignoring config fields** such as `rope_scaling`, `head_dim` or biases because the model "is basically Llama".
- **Transposing weights.** PyTorch's `nn.Linear` stores `[out_features, in_features]`, which is exactly our row-major `rows × cols` with one row per output. If you find yourself transposing, check twice.

## 8. How the professionals do it

- **Hugging Face `transformers`** reads the same three files; its `AutoModelForCausalLM` dispatches on `config.json`'s `model_type` to a per-family implementation. Rust ports like **candle** do the same with a `Config` struct per family and memory-mapped safetensors.
- **llama.cpp** converts checkpoints once into its own GGUF format, which puts the config, the tokenizer (vocabulary, merges, pre-tokenizer type, chat template) and the weights in a single file, aligned for direct memory mapping. Its conversion script has to recognise each model's pre-tokenizer, which it does by hashing the tokenizer's output on a fixed test string.
- **vLLM and SGLang** load safetensors directly (sharded across files for large models) and use Hugging Face's `tokenizers` library, plus the `chat_template` from `tokenizer_config.json` rendered by a Jinja engine. Rust servers use the `minijinja` crate for the same purpose.
- **Loading time matters at scale.** Serving systems memory-map weights, keep them in the page cache between restarts, stream them straight to GPU memory, or use formats designed for fast parallel loading; a server that takes minutes to start cannot scale up quickly when traffic grows.
- **Golden tests against the reference** are how every serious engine validates a new model family: logits within tolerance at several positions, greedy generations identical for some tokens, and per-layer hidden states when something is off.

## 9. Exercises

1. **SmolLM2-360M.** Download it (`./tools/download_model.sh 360M`) and run `ask --model 360m` and `bench 360m`. Nothing in the engine changes. Before running `bench`, predict its decode tokens/s from the file size and the bandwidth the 135M model reached.
2. **A different pre-tokenizer.** Replace `split_words` with chapter 11's simplified `pre_tokenize` and run the tokenizer test. Which cases fail first?
3. **Longest-first.** In `Tokenizer::encode`, special tokens are matched earliest-first and, at the same position, longest-first. Construct a vocabulary where longest-first matters.
4. **A faster `f32` load.** Make `load_f32` convert straight into the aligned buffer, without the temporary vector. Measure the load time.
5. **A second conversation turn without re-processing.** In `chat`, the second turn re-processes the whole conversation. What would you need to keep, and what must you be careful about, to process only the new message? (Chapter 24 does it.)
6. **Break it on purpose, twice.** First swap `w_gate` and `w_up` in the loader; then, instead, set `rope_layout` to `RopeLayout::Interleaved` in `parse_config`. Each time, run `logits_match_pytorch` and ask the model why the sky is blue. What is the largest logit difference, and does the model still produce English?

## 10. Check yourself

1. What do the three files each contain, and which one is most often subtly wrong?
2. Why does the loader refuse a checkpoint with an unexpected tensor?
3. Why did aligning the weights not speed up decoding, when chapter 6 measured a 2x penalty for misalignment?
4. What does "RSS file" mean for the mapped model, and why does it matter for a server running several processes?
5. What are GPT-2's byte-to-unicode table and the `\s+(?!\S)` alternative for?
6. What does SmolLM2's tokenizer do with a byte its vocabulary has no token for?
7. Why is the reference run in `float32`?

## 11. Recap

- A model is three files plus a chat template. Read every field that changes the computation, refuse everything you do not implement, and check every tensor.
- Memory-mapped `bf16` weights load in 8 ms and are shared through the page cache. They decoded as fast as an aligned copy for the 135M model and 12% slower for the 360M one: far from chapter 6's factor of 2, because the bottleneck is DRAM.
- `bf16` weights decode 1.5 times faster than `f32` (82-98 against 53-65 tokens/s), not the 2x that halving the bytes suggests: the `bf16` kernel does more work per byte.
- The tokenizer must match the reference exactly: byte-to-character table, merge ranks, GPT-2's pattern with its look-ahead, digit splitting, special tokens, and even the bytes it silently drops.
- Our engine matches PyTorch: identical tokens, logits within 6.5e-5, identical greedy generations. The model itself is small and often wrong; the engine is not.

## Answers

**Exercises**

1. The 360M model has 361.8 million parameters, 724 MB in `bf16`: 2.7 times the 135M model's bytes. At the 19-26 GB/s the `bf16` kernel reached, a decode step should take 28-38 ms: 26-36 tokens per second. Measured on the reference machine, in five runs: 27.8-30.2 tokens/s mapped and 30.6-36.6 aligned (the only case where alignment helped consistently, section 3.2), and 22.6-23.2 tokens/s with `f32` weights. Its answer to the sky question is much better (Rayleigh scattering by nitrogen and oxygen molecules, shorter wavelengths scattered more), though not flawless: more parameters, better answers, slower tokens.
2. The cases with runs of whitespace fail (`"a   b"`, `"trailing spaces   "`, `"tabs\tand\nnewlines\n\n\nx"`), then the digits (chapter 11's pre-tokenizer keeps `12345` together) and the contractions (`"don't"`). The novel fails at its first contraction or double space.
3. Take two special tokens, `<|im|>` and `<|im|>x`, and the text `<|im|>x`. Both match at position 0, so "earliest" does not decide. Longest-first encodes the text as the single token `<|im|>x`; without the rule, the result would depend on the order in which the tokens are listed, and could be `<|im|>` followed by `x`.
4. Give `DenseF32` a constructor that takes an `AlignedVec<f32>`, and build that vector with `AlignedVec::from_fn`, converting each element from the tensor's bytes. Measured on the reference machine (four runs each, after a first run of about 2 s for both versions): 464-486 ms against 567-896 ms for the original. The copy was not the whole cost: converting 134 million values and touching 538 MB of new memory remain.
5. Keep the KV cache and the token ids of the conversation so far. The new prompt must extend the old one exactly: the previous turn's answer must be in the cache as the tokens the model generated (and the template's `<|im_end|>\n` after it, which the model produced only partly: it generated `<|im_end|>` but the cache does not contain it, since the stop token is never fed back). Then only the new tokens need a forward pass. The subtle part is that re-tokenizing the text of the answer does not always give the tokens that were generated, so compare token ids, not text.
6. Swapping gate and up computes `silu(up) ⊙ gate` instead of `silu(gate) ⊙ up`: largest logit difference 25.7, and the output is not language at all (an endless run of `<filename>` tokens and fragments of unrelated words). The wrong RoPE layout gives a difference of 14.7 and fluent, wrong English: "The capital of France" (and nothing more) for the capital question, "The sky is a person who loves to help you." for the sky. The second bug is the dangerous kind. Only a numerical comparison catches it reliably.

**Check yourself**

1. `config.json`: the architecture's numbers and special token ids. `model.safetensors`: the weights with types and shapes. `tokenizer.json`: vocabulary, merges, pre-tokenizer and special tokens. The tokenizer is most often subtly wrong, because small differences affect only some inputs and nothing crashes.
2. Because the checkpoint then contains a part of the model this engine does not know how to use (a bias, an extra norm), and running without it computes a different function that still produces plausible text.
3. Chapter 6's penalty was measured with data in the cache, where load instructions are the bottleneck. Decode streams the weights from DRAM, and the wait for memory hides most of the cost of loads that cross cache lines (none measurable for the 135M model, about 12% for the 360M one).
4. Resident memory backed by the mapped file: pages in the operating system's page cache. They are shared by every process that maps the file and can be dropped and re-read under memory pressure. Several server processes then share one copy of the weights instead of each holding a private one.
5. The table gives each byte a printable character, so tokens (which are byte strings) can be stored as JSON strings. `\s+(?!\S)` makes a whitespace run leave its last character to the following word, so `" b"` stays one token.
6. It drops it silently, because it has no unknown token to use instead. Our tokenizer does the same, to match.
7. Our engine computes in `f32` with the `bf16` weights converted exactly, so a `float32` reference runs the same arithmetic on the same numbers, and any difference beyond rounding is a bug. A `bf16` reference would differ by rounding everywhere and need a loose tolerance.

## Further reading

- The SmolLM2 model card and paper (Allal et al., "SmolLM2: When Smol Goes Big", 2025).
- Hugging Face `tokenizers` documentation: pre-tokenizers, BPE model, added tokens.
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019, and its `encoder.py`: the byte-to-unicode table and the pre-tokenization pattern.
- Hugging Face `transformers`, "Chat templates" documentation.
- The safetensors format specification, and llama.cpp's GGUF specification for a single-file alternative.
- Next: [Chapter 17: Measuring and profiling](../17-profiling/README.md). Find out where the time goes, and what can still be sped up.
