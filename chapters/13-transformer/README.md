# Chapter 13: The transformer

> **In one sentence:** a Llama-style language model is an embedding lookup, then the same layer (normalize, attend, add back; normalize, gated MLP, add back) repeated N times, then a final normalization and a projection to one score per vocabulary token, and every quantity an inference engineer cares about (parameters, FLOPs, memory, time) can be computed from a handful of numbers in its config.

**Where this fits:** chapters 5-12 built every part separately. This chapter assembles them into a complete model with exactly SmolLM2's architecture, as a slow, obviously correct **reference implementation**. Chapter 14 builds the fast engine and tests it against this one; chapter 16 loads SmolLM2's real weights into both.

**You need:** chapters 5-8 (linear layers, operators), 11 (tokens) and 12 (attention).

**You will build:** a model configuration with SmolLM2-135M's shape, parameter and FLOP counting, random weights, a full forward pass that returns logits for every position, and greedy generation that recomputes everything at every step.

---

## 1. The intuition

Think of a document passing through an office of 30 identical desks, one after another. At each desk a clerk does two things:

1. **Consults the file** (attention): looks back over all the earlier documents in the stack and pulls in whatever information is relevant to this one.
2. **Thinks about it alone** (the MLP): processes the document on its own, using everything the clerk has learned.

After each step, the clerk does not replace the document: they **add their notes to it** (the residual connection). The document that leaves desk 30 carries the original content plus 60 layers of annotations. At the end, a final clerk reads it and writes down, for every word in the dictionary, how likely it is to come next.

**Where the analogy breaks:** the clerks do not "understand" anything in the human sense, and their notes are not words: each document is a vector of 576 numbers, and each clerk's knowledge is millions of fixed weights. And unlike an office, the model processes every position of the text at once during prefill, with the causal mask ensuring each position only consults earlier ones.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Decoder-only transformer** | A stack of causal self-attention + MLP layers, predicting the next token. GPT, Llama, SmolLM2. |
| **Layer / block** | One repeated unit: attention block + MLP block, each with a norm and a residual. |
| **Residual stream** | The hidden-state vector each layer reads from and adds to. |
| **Pre-norm** | Normalizing the input of each block (not its output), as Llama does. |
| **MLP / feed-forward network** | The per-token part of a layer: gate and up projections, SwiGLU, down projection. |
| **LM head** | The final projection from hidden size to vocabulary size, producing logits. |
| **Tied embeddings** | Using the embedding matrix as the LM head, saving vocab × hidden parameters. |
| **Config** | The numbers that define a model's shape: sizes, counts, θ, ε. |
| **Reference implementation** | A simple, trusted implementation used to check fast ones. |
| **Greedy decoding** | Always choosing the highest-scoring next token (chapter 15 covers alternatives). |

## 3. The concepts in depth

### 3.1 The whole model on one page

```text
tokens ──► embedding lookup ──► x (n × 576)
                                  │
            ┌─────────────────────┴───────────────────── repeat 30 times ──────────┐
            │  a = RMSNorm(x)                                                      │
            │  q, k, v = a·W_qᵀ, a·W_kᵀ, a·W_vᵀ        576 → 576, 192, 192         │
            │  q, k = RoPE(q), RoPE(k)                  position-dependent rotation│
            │  o = causal GQA attention(q, k, v)        9 query heads, 3 KV heads  │
            │  x = x + o·W_oᵀ                           576 → 576, residual add     │
            │  a = RMSNorm(x)                                                      │
            │  x = x + (silu(a·W_gateᵀ) ⊙ a·W_upᵀ)·W_downᵀ   576 → 1536 → 576      │
            └─────────────────────┬────────────────────────────────────────────────┘
                                  │
                   RMSNorm ──► x·W_embedᵀ ──► logits (n × 49,152)
```

Row `t` of the logits scores every possible token for position `t + 1`. For generation only the last row matters.

The design choices that define "Llama-style", all shared by SmolLM2: pre-norm with RMSNorm (chapter 8), rotary position embeddings in half-split layout (chapter 12), grouped-query attention (chapter 12), a SwiGLU MLP with three matrices (chapter 8), no biases anywhere, and tied embeddings.

### 3.2 Counting parameters from the config

With hidden size `h`, head dimension `d`, `H` query heads, `G` KV heads, MLP size `m`, `L` layers and vocabulary `V`:

```text
attention per layer:  h·Hd  +  2·h·Gd  +  Hd·h          (W_q, W_k, W_v, W_o)
MLP per layer:        3·h·m                              (W_gate, W_up, W_down)
norms per layer:      2·h
embedding:            V·h          (+ V·h again if the LM head is not tied)
final norm:           h
```

For SmolLM2 (h = 576, d = 64, H = 9, G = 3, m = 1,536, L = 30, V = 49,152), part 1 of the demo prints:

```text
== 1. SmolLM2-135M: 134515008 parameters
   embedding (also the output layer):    28311552   21.0%
   attention, 30 layers x    884736:    26542080   19.7%
   MLP,       30 layers x   2654208:    79626240   59.2%
   norms:                                   35136    0.0%
```

134,515,008 is exactly the number chapter 9 counted in `model.safetensors`, and the test `smollm2_parameter_count_matches_the_checkpoint` checks it. When your parameter formula matches the checkpoint to the last digit, you know you have the architecture right: the right number of layers, heads, MLP size and tying.

Things this breakdown tells an inference engineer:

- **The MLP is the biggest part** (59%). Most weight bytes read per token are MLP weights, so that is where quantization (chapters 18-19) saves the most.
- **The embedding is 21%**, and because it doubles as the LM head, it is read in full for every generated token: a 49,152 × 576 matrix-vector product, the largest single one in the model.
- **Norms are negligible** in size, which is why they are usually kept in high precision even when everything else is quantized.

### 3.3 Counting FLOPs

Every weight in a matmul is used in one multiply-add (2 FLOPs) per token. The embedding lookup does no arithmetic, but the LM head does. Attention adds 4·d FLOPs per head per earlier token (chapter 12). So:

```text
FLOPs per token ≈ 2 × (matmul weights) + 4 · L · H · d · context
```

```text
   FLOPs for one token with     0 tokens of context: 0.269 GFLOP
   FLOPs for one token with  1000 tokens of context: 0.338 GFLOP
   FLOPs for one token with  8000 tokens of context: 0.822 GFLOP
```

The rule of thumb **"2 FLOPs per parameter per token"** holds at short contexts. At long contexts attention grows until it dominates: at 8,000 tokens of context it is two thirds of SmolLM2's per-token work. For large models the matmul term is so large that the crossover happens much later, but for small models with long contexts, attention is the main cost.

### 3.4 The residual stream

Each block computes something and *adds* it to `x` instead of replacing `x`. Two consequences:

- The input embedding reaches the last layer directly, plus corrections. A layer that has nothing to add can output zeros and change nothing. This is what makes deep stacks trainable, and why the final RMSNorm is needed: after 60 additions the vector's size has drifted, and the LM head expects a normalized input.
- In code, `x` is the one buffer that persists across the layer loop; everything else (`q`, `k`, `v`, MLP activations) is scratch space that is overwritten in every layer. Chapter 14 allocates exactly one set of scratch buffers for the whole model.

### 3.5 Why a reference implementation

The fast engine of chapter 14 will have a KV cache, parallel attention, reused buffers, and later quantized weights and batching. Each is a place for bugs that produce *plausible* output: a model that talks, just slightly worse. The defence is a second implementation that is too simple to be wrong in those ways, and a test that the two agree on the same weights.

This forward pass is that implementation. It has one path through the code, allocates fresh buffers for every call, recomputes everything from scratch, and has no state between calls. It is slow on purpose.

### 3.6 Generation without a cache, and the waste it reveals

Greedy generation: run the model on the sequence, take the last row of logits, append its argmax, repeat. Part 2 of the demo does this with random weights of SmolLM2's shape (random weights make the output text meaningless, but the computation and its cost are identical to the real model):

```text
== 2. greedy generation with random SmolLM2-shaped weights, no cache (4 threads)
   step  1: ran the model on 32 tokens in  168.6ms (1.0x the first step)
   step  5: ran the model on 36 tokens in  213.6ms (1.3x the first step)
   step  9: ran the model on 40 tokens in  187.4ms (1.1x the first step)
   step 13: ran the model on 44 tokens in  307.3ms (1.8x the first step)
   ...
   step 29: ran the model on 60 tokens in  268.7ms (1.6x the first step)
   generated 32 tokens; the model processed 1520 token positions to do it
```

Two things to see (the individual step times are noisy on this shared VM; the trend and the count are not):

- **Each step costs more than the last**, because it reprocesses the whole, growing sequence.
- **To generate 32 tokens from a 32-token prompt, the model processed 1,520 token positions.** With the KV cache of chapter 14 it processes 63: the 32 prompt tokens once, then one position for each generated token except the last (whose successor is never needed). Generating `g` tokens after a prompt of `p` costs O(g·(p + g)) positions without a cache and O(p + g) with one. For a 1,000-token prompt and a 500-token answer, that is 625,000 positions against 1,500.

There is a second waste: this forward pass computes logits for *every* position (a 60 × 49,152 output at the last step), and generation uses only the last row. Chapter 14 computes the LM head for the last position only.

## 4. The code

All of it is in [`src/lib.rs`](src/lib.rs); the demo is [`src/main.rs`](src/main.rs).

### 4.1 The config

<!-- file: src/lib.rs -->
```rust
    pub fn smollm2_135m() -> Self {
        Self {
            vocab_size: 49_152,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 30,
            num_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            rope_theta: 100_000.0,
            rope_layout: RopeLayout::HalfSplit,
            rms_norm_eps: 1e-5,
            max_positions: 8192,
            tie_embeddings: true,
        }
    }
```

Every value comes from SmolLM2's `config.json` (printed in full in chapter 16), except `head_dim`, which Llama configs leave implicit as `hidden_size / num_attention_heads` = 576 / 9 = 64, and `rope_layout`, which comes from `"rope_interleaved": false`. Chapter 16 parses the JSON file instead of hard-coding it.

<!-- file: src/lib.rs -->
```rust
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.hidden_size;
        let head = if self.tie_embeddings { 0 } else { embed };
        embed + self.num_layers * self.layer_params() + self.hidden_size + head
    }
```

Section 3.2's formula. The `+ self.hidden_size` is the final norm's weight.

### 4.2 Weights and the tied head

<!-- file: src/lib.rs -->
```rust
    pub fn lm_head(&self) -> &[f32] {
        self.lm_head.as_deref().unwrap_or(&self.embed)
    }
```

`lm_head: Option<Vec<f32>>` is `None` for tied models. `as_deref()` turns `Option<Vec<f32>>` into `Option<&[f32]>`, and `unwrap_or(&self.embed)` substitutes the embedding when there is no separate head. One line expresses "use the LM head, or the embedding if tied", and callers never need to know which.

In `Weights::random`, `(!config.tie_embeddings).then(|| matrix(...))` builds the head only when needed: `bool::then` returns `Some(closure())` if true and `None` otherwise, without running the closure when false.

### 4.3 The forward pass

<!-- file: src/lib.rs -->
```rust
        // 1. Embedding lookup: one row per token.
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| {
                self.embed[t as usize * h..(t as usize + 1) * h]
                    .iter()
                    .copied()
            })
            .collect();
```

The embedding lookup copies each token's row into the residual stream `x`. (Chapter 8's `embedding` returns a borrowed row; here we need an owned, mutable copy because the layers add to it.)

<!-- file: src/lib.rs -->
```rust
        for layer in &self.layers {
            // 2. Attention block.
            for (xi, ni) in x.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
                rms_norm(xi, &layer.attn_norm, c.rms_norm_eps, ni);
            }
            matmul_nt_pool(pool, &normed, &layer.wq, &mut q, n, h, q_dim);
            matmul_nt_pool(pool, &normed, &layer.wk, &mut k, n, h, kv_dim);
            matmul_nt_pool(pool, &normed, &layer.wv, &mut v, n, h, kv_dim);
            for (pos, (qt, kt)) in q
                .chunks_exact_mut(q_dim)
                .zip(k.chunks_exact_mut(kv_dim))
                .enumerate()
            {
                rope.apply_heads(qt, pos);
                rope.apply_heads(kt, pos);
            }
            attention(&q, &k, &v, &mut attn, c.heads(), 0, true, &mut scores);
            matmul_nt_pool(pool, &attn, &layer.wo, &mut proj, n, q_dim, h);
            add_inplace(&mut x, &proj);
```

Section 3.1's diagram, line for line:

- RMSNorm each token's row of `x` into `normed`. `x` itself is kept for the residual add.
- Three projections, each chapter 7's parallel NT matmul with `n` input rows.
- RoPE rotates each position's query and key heads; the position of row `pos` is `pos` because this pass always starts at position 0.
- Chapter 12's attention with `q_start = 0` and the causal mask.
- The output projection, then `x += proj`.

The MLP block has the same shape: norm, gate and up projections, SwiGLU, down projection, residual add. After the loop, the final norm and one big matmul against the LM head give the logits.

This is the whole model: about 60 lines. Everything a production engine adds (caching, batching, quantization, fused kernels) makes it faster without changing what it computes.

### 4.4 Generation

<!-- file: src/lib.rs -->
```rust
    let mut tokens = prompt.to_vec();
    for _ in 0..new_tokens {
        let start = std::time::Instant::now();
        let logits = weights.forward(pool, &tokens);
        let last = &logits[(tokens.len() - 1) * vocab..];
        tokens.push(argmax(last));
        on_step(tokens.len() - 1, start.elapsed());
    }
```

The simplest possible generation loop: whole-sequence forward, last row, argmax, append. `on_step` is a callback (a closure the caller passes in) that lets the demo print timings without the library knowing anything about printing.

## 5. Run it

```bash
cargo test -p ch13-transformer
cargo run --release -p ch13-transformer
```

The demo's output is shown in sections 3.2, 3.3 and 3.6. Building 134.5 million random weights takes about 3.6 s on the reference machine (a slow random number generator, called 12 times per weight); the real weights in chapter 16 load in a fraction of that.

## 6. The Rust behind it

**`Option<Vec<T>>` plus `as_deref`** expresses optional owned data with borrowing access, without duplicating the tied matrix or using `Rc`.

**`bool::then`** turns a condition into an `Option`, running the closure only when needed.

**Closures as callbacks.** `generate_without_cache(..., mut on_step: impl FnMut(usize, Duration))` accepts any closure that may mutate its captured state (the demo's closure updates a running total). `impl FnMut` means the callback is monomorphized: no dynamic dispatch, and the compiler can inline it.

**Mutable closures that capture mutably.** In `Weights::random`, `let mut matrix = |rows, cols| ... rng.normal() ...` captures `rng` by mutable reference, so it must itself be declared `mut` to be called. The borrow checker ensures nothing else uses `rng` while `matrix` exists.

**Configs as plain data.** `Config` derives `Clone`, `Debug` and `PartialEq`, so it can be copied into the weights, printed in error messages and compared in tests. Keeping the architecture description separate from the weights is what lets chapter 16 validate a checkpoint against a config before loading a single tensor.

## 7. Mistakes you will make

- **Normalizing `x` in place** instead of into a separate buffer. The residual then adds the normalized vector, not the original, and the model degrades without crashing.
- **Forgetting the final norm** before the LM head, or using a layer's norm weights for it.
- **Mixing up `W_gate` and `W_up`.** SwiGLU is `silu(gate) ⊙ up`; swapping them gives `silu(up) ⊙ gate`, a different function. The names in the checkpoint decide which is which.
- **Assuming an untied LM head** and looking for a tensor that does not exist (or, for untied models, using the embedding).
- **Computing `head_dim` wrongly** for models where it is not `hidden_size / num_heads`. Some architectures set it explicitly; read the config.
- **Trusting a fast implementation that "produces text".** Compare against a reference, with tolerances, on the same weights.

## 8. How the professionals do it

- **Hugging Face transformers' `modeling_llama.py`** is the de facto reference implementation for Llama-family models, and the one chapter 16 compares against. Engines like vLLM, llama.cpp and TensorRT-LLM validate their implementations against it.
- **Model families differ in small ways** that matter: Qwen2 adds biases to the Q/K/V projections, Gemma scales embeddings by √h and adds 1 to its RMSNorm weights, Phi-3 fuses Q/K/V into one matrix, Mistral uses sliding-window attention. Engines keep a separate "model definition" per family, sharing the kernels.
- **Parameter and FLOP formulas** like section 3.2's are the basis of capacity planning (chapter 30) and of published comparisons such as "Chinchilla-optimal" training budgets.

## 9. Exercises

1. **A different model.** Write `Config` values for a model with h = 2,048, 16 query heads, 8 KV heads, d = 128, m = 8,192, 16 layers, V = 128,256 and untied embeddings. How many parameters? What fraction is the embedding plus LM head?
2. **Where attention wins.** For SmolLM2, at what context length do attention FLOPs equal matmul FLOPs per token?
3. **Last row only.** Change `forward` (or add `forward_last`) so the LM head is applied only to the last position. How much faster is step 29 of the demo?
4. **Break the residual.** Normalize `x` in place instead of into `normed`, and run the `appending_tokens_never_changes_earlier_logits` test. Does it still pass? Why is that test not enough to catch this bug?
5. **Count positions.** Write the formula for the number of token positions processed when generating `g` tokens after a `p`-token prompt, with and without a cache. Check it against the demo's 1,520.

## 10. Check yourself

1. Name the operations in one Llama layer, in order.
2. Why is SmolLM2's embedding matrix read in full for every generated token?
3. Which part of SmolLM2 has the most parameters?
4. What does "2 FLOPs per parameter per token" leave out, and when does it matter?
5. What is a reference implementation for, and why is it deliberately slow?
6. Why does generating without a cache waste so much work?

## 11. Recap

- A Llama-style model: embed; L × (RMSNorm → Q/K/V → RoPE → GQA causal attention → O → residual; RMSNorm → SwiGLU MLP → residual); final RMSNorm; LM head.
- Parameters and FLOPs follow from the config. SmolLM2-135M: 134,515,008 parameters (checked against the checkpoint), 59% MLP, 21% embedding, 0.27 GFLOP per token at short context.
- Attention's share of FLOPs grows with context; at 8,000 tokens it is two thirds of SmolLM2's work.
- The residual stream is the only state that persists across layers; everything else is scratch.
- A simple reference implementation is the baseline every optimization is tested against.
- Without a cache, generation reprocesses the whole sequence at every step: 1,520 positions for 32 tokens in the demo, against 63 with a cache.

## Answers

**Exercises**

1. Attention per layer: 2048·2048 + 2·2048·1024 + 2048·2048 = 12,582,912. MLP: 3·2048·8192 = 50,331,648. Norms: 4,096. Per layer: 62,918,656; 16 layers: 1,006,698,496. Embedding and LM head: 2 × 128,256 × 2,048 = 525,336,576. Final norm: 2,048. Total: 1,532,037,120 (about 1.5 B). Embedding plus head: 34%. Small models with large vocabularies spend a big share of their parameters on the vocabulary.
2. Matmul FLOPs per token are 2 × (30 × (3,540,096 − 1,152) + 28,311,552) ≈ 0.269 GFLOP. Attention adds 4 × 30 × 9 × 64 × context = 69,120 × context. They are equal at about 269e6 / 69,120 ≈ 3,890 tokens of context.
3. Apply the final norm and LM head only to the last row: `matmul_nt_pool(pool, &normed[(n - 1) * h..], self.lm_head(), &mut logits, 1, h, vocab)`. The saving is the LM head for n − 1 positions: at 60 positions, 59 × 49,152 × 576 × 2 ≈ 3.3 GFLOP of the step's roughly 16 GFLOP, so the FLOP count predicts about 20% off. Measure it: on this noisy VM you may need several runs to see it clearly.
4. It still passes: normalizing in place changes what every position computes, but it changes it the same way whether or not later tokens exist, so causality is preserved. The test checks one property (causality), not correctness. Only a comparison against known-correct outputs (a reference with real weights, chapter 16) catches this bug. Property tests and reference tests catch different bugs, and you need both.
5. Without a cache, step `i` (i = 1..g) processes p + i − 1 positions, so the total is g·p + g(g − 1)/2. For p = 32, g = 32: 1,024 + 496 = 1,520, matching the demo. With a cache: p + g − 1 = 63 (the prompt once, then one position per new token, except that the last generated token never needs to be processed).

**Check yourself**

1. RMSNorm, Q/K/V projections, RoPE on Q and K, causal grouped-query attention, output projection, residual add; RMSNorm, gate and up projections, SwiGLU, down projection, residual add.
2. Because the embeddings are tied: the same matrix is the LM head, and the LM head multiplies the final hidden state by all 49,152 rows to score every token.
3. The MLPs: 59% of the parameters.
4. It counts only the matmul weights. It leaves out attention's scores and weighted sums, which grow with context length; for long contexts, especially in small models, they can dominate.
5. It is a simple, trusted implementation used to check that faster, more complex implementations compute the same thing. Simplicity (no caching, no reuse, one code path) is what makes it trustworthy; speed would add the very complexity it exists to check.
6. Each step recomputes keys, values and outputs for every earlier position, although, because of the causal mask, none of them changed since the previous step.

## Further reading

- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023: the architecture choices (pre-norm RMSNorm, SwiGLU, RoPE).
- Hugging Face `transformers`, `src/transformers/models/llama/modeling_llama.py`: the reference implementation.
- Kaplan et al., "Scaling Laws for Neural Language Models", 2020, appendix on parameter and FLOP counting.
- Next: [Chapter 14: The KV cache](../14-kv-cache/README.md). Stop recomputing the past.
