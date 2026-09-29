# Chapter 11: Tokenization

> **In one sentence:** a language model reads and writes integers, not text, and the tokenizer that converts between them decides how many steps every request costs, which languages are cheap or expensive, and whether streamed output ever shows a broken character.

**Where this fits:** this is the first chapter about language models. Before a transformer (chapters 12-14) can do anything, text must become a sequence of token IDs; after it has predicted a token, that ID must become text again, usually streamed to a user one token at a time. Chapter 16 loads SmolLM2's real tokenizer, which is the same algorithm as this chapter with a larger vocabulary and a few format details.

**You need:** chapters 1-3. Knowing that UTF-8 encodes characters as 1-4 bytes helps; the lesson explains the rest.

**You will build:** byte-level BPE from scratch (training, encoding with merges in rank order, special tokens, decoding), an encoding cache, and a streaming decoder that never splits a UTF-8 character. The demo trains tokenizers on this course's own lessons.

---

## 1. The intuition

Imagine telegrams where you pay per word-block. A clever telegraph company gives every common word ("the", "and", "cache") its own block, and spells out rare words letter by letter. Messages full of common words are cheap; messages in a language or jargon the company did not plan for are spelled out and expensive.

A tokenizer is that company's codebook. Each **token** is a block: sometimes a whole word (" cache"), sometimes a fragment ("ization"), sometimes a single byte. The codebook is built once, from example text (the **training corpus**): the more often a fragment appears there, the more likely it gets its own token.

A language model is billed, in time and memory, per token. Every token of the prompt is processed in prefill; every generated token is a full forward pass; every token of context occupies KV cache memory (chapter 14). So the codebook directly sets the cost of every request.

**Where the analogy breaks:** telegraph blocks are whole words. Tokens are arbitrary byte sequences, and a token can end in the middle of a character. A single "é" or "👋" can be split across two or four tokens, and anyone streaming tokens to a screen has to reassemble the bytes before showing anything. That is section 3.6.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Token** | One entry of the vocabulary; the unit a model reads and writes. |
| **Token ID** | The integer that names a token. |
| **Vocabulary** | The fixed list of all tokens. SmolLM2 has 49,152. |
| **BPE** | Byte Pair Encoding: build tokens by repeatedly merging the most frequent adjacent pair. |
| **Merge** | A learned rule "tokens a and b next to each other become token c". |
| **Rank** | A merge's position in the learned order. Lower rank = learned earlier = applied first. |
| **Byte-level** | The base alphabet is the 256 byte values, so any input can be encoded; nothing is "unknown". |
| **Pre-tokenization** | Splitting text into chunks (roughly words) that merges may not cross. |
| **Special token** | A reserved token like `<\|endoftext\|>` or `<\|im_start\|>`, matched literally, never merged. |
| **Chat template** | The special-token format that marks roles and turns in a conversation. |
| **Detokenization** | Turning token IDs back into text. |
| **UTF-8** | The encoding of Unicode text as 1-4 bytes per character. |

## 3. The concepts in depth

### 3.1 Why not characters, or words?

- **One token per character** makes vocabularies tiny, but sequences long: every character is a forward pass to generate, and attention cost grows with the square of the length (chapter 12). And the set of Unicode characters is enormous and open-ended.
- **One token per word** makes sequences short, but vocabularies huge, and any word not in the vocabulary ("unknown") cannot be represented at all.
- **Subwords** are the compromise every modern model uses: common words are single tokens, rare words are spelled from pieces, and with a byte-level base, *any* byte sequence can be encoded.

### 3.2 Training BPE

Start with 256 tokens, one per byte value. Then repeat:

1. Count every pair of adjacent tokens in the corpus.
2. Take the most frequent pair, say (`t`, `h`), and create a new token `th`.
3. Replace every occurrence of that pair in the corpus with the new token.

Each round adds one token. Stop at the desired vocabulary size. The list of merges, in the order learned, *is* the tokenizer.

Trained on this course's lessons, the first merges are `"  "` (two spaces: the corpus is Markdown with indented code), `" t"`, `" a"`, four spaces, `"he"`, `"in"`, `"er"`, `"re"`, `" s"` and `" the"`. The 2,048th merge is `" fresh"`. The algorithm has no idea what a word is; frequent fragments simply rise to the top.

Two details matter for real tokenizers:

- **Pre-tokenization.** Before counting pairs, the text is split into chunks (roughly: words with their leading space, runs of digits, runs of punctuation, runs of whitespace), and merges never cross chunk boundaries. Otherwise the tokenizer would learn tokens like "e c" from "the cat", wasting vocabulary on accidents of word order.
- **Determinism.** Ties between equally frequent pairs must be broken by a fixed rule, or two training runs produce different tokenizers.

### 3.3 Encoding: apply merges in the order they were learned

To encode new text: pre-tokenize, turn each chunk into bytes, then repeatedly find the adjacent pair whose merge has the **lowest rank** (was learned earliest) and merge it, until no adjacent pair has a merge. This reproduces what training would have done to that chunk.

Why lowest rank first, and not "longest match" or "left to right"? Because later merges were learned on text where the earlier merges had already been applied. A tokenizer that applies merges in a different order produces different token IDs from the ones the model was trained on, and the model then sees text it never saw in training. The output looks almost right and is subtly worse. This is why chapter 16 checks our tokenizer against the reference implementation token by token.

### 3.4 Special tokens and chat templates

Some tokens are not text at all: `<|endoftext|>` marks the end of a document; SmolLM2's `<|im_start|>` and `<|im_end|>` mark the start and end of a chat turn. They are matched literally in the input before pre-tokenization and never merged with neighbouring text. A chat request becomes a single string in the model's **chat template**:

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

The model then generates the assistant's reply and ends it with `<|im_end|>`, which is how the server knows to stop. Getting the template wrong (a missing newline, a different system prompt, the wrong role name) is one of the most common reasons a model "works but gives worse answers" in a new serving stack.

A security note: if user text is allowed to contain the literal string `<|im_start|>` and it is encoded as the special token, users can forge turns (for example, write fake "system" instructions). Production tokenizers can be told to encode special-token text found in user input as ordinary text.

### 3.5 What tokens cost

Part 1 of the demo trains tokenizers of increasing size on 356 KB of this course's text:

```text
   vocab | training time | tokens | bytes per token
     256 |        7.82ms | 356458 |   1.00
     512 |      171.96ms | 186455 |   1.91
    1024 |      554.60ms | 141885 |   2.51
    2048 |         1.05s | 115635 |   3.08
```

With 2,048 tokens, English text averages 3.08 bytes per token. Real tokenizers with 32k-200k tokens reach about 4 bytes per token on English. Every doubling of the vocabulary gives smaller and smaller gains, and every token in the vocabulary costs a row of the embedding matrix and a row of the output layer. SmolLM2's 49,152 × 576 embedding matrix is 28.3 million parameters, 21% of the whole model, and the output layer (which shares those weights) is the largest single matrix-vector product of every decode step.

Part 3 shows how differently the same tokenizer treats different text:

```text
   15 tokens for 52 bytes: The| K|V| cache| st|ores| k|ey|s| and| values| for| every| token|.
   28 tokens for 57 bytes: let| sum|:| f|32| =| a|.|iter|().|zip|(|b|).|map|(||(|x|,| y|)||| x| *| y|).|sum|();
   27 tokens for 27 bytes: �|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�
```

English prose: 3.5 bytes per token. Rust code: 2 bytes per token. Chinese (the phrase 推理引擎的内存带宽, "memory bandwidth of the inference engine"): one token per *byte*, three per character, because the corpus contained almost no Chinese and no Chinese merges were learned. Each `�` is a single byte that is not a whole character on its own.

For inference this has direct consequences:

- **Cost and latency depend on language.** The same request in a poorly covered language needs several times the tokens: more prefill time, more decode steps, more KV cache, and in paid APIs, a bigger bill. Tokenizers for multilingual models are trained on multilingual corpora for exactly this reason.
- **Token counts, not character counts, set limits.** Context windows, `max_tokens`, rate limits and prices are all in tokens.
- **Numbers are tokenized unpredictably** ("3.14159" became `3|.|14|15|9`), which is one reason small models are bad at arithmetic. SmolLM2's tokenizer splits every digit into its own token for consistency.

### 3.6 Streaming detokenization and UTF-8

A server streams generated text to the user token by token. The naive approach, `decode(&[token])` for each token and send the result, breaks as soon as a character spans two tokens. Part 5 of the demo shows it happening in our tokenizer:

```text
   token  195 bytes [c3] -> emits ""
   token  169 bytes [a9] -> emits "é"
   ...
   token  240 bytes [f0] -> emits ""
   token  159 bytes [9f] -> emits ""
   token  145 bytes [91] -> emits ""
   token  139 bytes [8b] -> emits "👋"
```

"é" is the two bytes `c3 a9`; "👋" is the four bytes `f0 9f 91 8b`. Decoding `[c3]` alone is not valid UTF-8. A naive server would send the replacement character `�`, then another `�` for `a9`, and the user sees garbage where "é" should be. The `StreamDecoder` keeps incomplete bytes back until the character is complete.

It has to tell two situations apart:

- The pending bytes are the *start* of a valid character that has not finished arriving (`c3` alone). Wait for more.
- The pending bytes are *invalid* and can never become valid (`ff`, or `c3` followed by a byte that is not a continuation byte). Replace them with `�` and continue, or the stream stalls forever.

Rust's `std::str::from_utf8` error reports exactly this: `valid_up_to()` says how many bytes are good, and `error_len()` returns `None` for "incomplete at the end" and `Some(n)` for "n invalid bytes".

The same problem, in a different form, appears in **stop strings** ("stop generating when the output contains `\n\n`"): the stop string can span two tokens, so the server must look at the decoded text, not at single tokens (chapter 15).

### 3.7 Tokenizer speed

Tokenization runs on every request, before the model sees anything. Part 4 measures encoding the whole 356 KB corpus:

```text
   no cache:              27.83ms  (12.8 MB/s)
   cache, first pass:     10.99ms  (32.4 MB/s, 6452 distinct chunks)
   cache, second pass:     5.35ms  (66.6 MB/s)
```

Most chunks in real text are repeated words, so caching each chunk's encoding more than doubles throughput even on the first pass. At 12.8 MB/s, a 100 KB prompt takes 8 ms to tokenize: small next to prefill for a large model, but not zero, and it scales with prompt length. Servers run tokenization on request-handling threads, never on the engine thread that runs the model (chapter 21), so a long prompt being tokenized does not delay other users' tokens.

## 4. The code

All of it is in [`src/lib.rs`](src/lib.rs); the demo is [`src/main.rs`](src/main.rs).

### 4.1 The tokenizer's state

<!-- file: src/lib.rs -->
```rust
pub struct Bpe {
    /// `merges[r] = (a, b)`: the r-th learned merge glues token `a` and
    /// token `b` into token `256 + r`.
    merges: Vec<(Token, Token)>,
    /// Pair → rank (position in `merges`). Lower rank = learned earlier =
    /// applied first.
    ranks: HashMap<(Token, Token), u32>,
    /// The bytes each token stands for.
    vocab: Vec<Vec<u8>>,
    /// Special tokens like `<|endoftext|>`: matched literally, never merged.
    specials: Vec<(String, Token)>,
}
```

Token IDs 0-255 are the bytes; 256 onwards are merges, in learning order, so token `256 + r` is merge `r`. The `ranks` map answers "is there a merge for this pair, and how early was it learned?" in constant time, which the encoder asks constantly. `vocab` stores each token's bytes for decoding.

### 4.2 Merging in place

<!-- file: src/lib.rs -->
```rust
fn merge_pair(ids: &mut Vec<Token>, a: Token, b: Token, new: Token) {
    let mut out = 0;
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
            ids[out] = new;
            i += 2;
        } else {
            ids[out] = ids[i];
            i += 1;
        }
        out += 1;
    }
    ids.truncate(out);
}
```

A read index `i` and a write index `out` walk the same vector. Merged pairs make the write index fall behind the read index, so no element is ever overwritten before it has been read, and no second vector is allocated. `truncate` drops the leftover tail. This two-index compaction pattern shows up wherever you filter or merge a buffer in place.

### 4.3 Training

<!-- file: src/lib.rs -->
```rust
        while tok.vocab.len() < vocab_size {
            let mut counts: HashMap<(Token, Token), u64> = HashMap::new();
            for (ids, n) in &words {
                for pair in ids.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += n;
                }
            }
            // Most frequent pair; ties broken by the smallest pair.
            let Some((&(a, b), _)) = counts
                .iter()
                .max_by(|x, y| x.1.cmp(y.1).then_with(|| y.0.cmp(x.0)))
            else {
                break; // nothing left to merge
            };
            let new = tok.vocab.len() as Token;
            for (ids, _) in &mut words {
                merge_pair(ids, a, b, new);
            }
            tok.add_merge(a, b);
        }
```

- `words` holds each *distinct* chunk once, with its count. The corpus has 6,452 distinct chunks, so each round works on thousands of short sequences instead of 356,000 bytes.
- `ids.windows(2)` yields every adjacent pair as a 2-element slice.
- `counts.entry(...).or_default()` inserts 0 for a new pair and returns a mutable reference either way: one hash lookup per pair.
- `max_by` with `then_with` compares by count, then (reversed) by the pair itself, so among equally frequent pairs the smallest wins. `HashMap` iteration order is random in Rust (it is seeded per process to resist hash-flooding attacks), so without an explicit tie-break, two runs could learn different merges.

This simple version recounts all pairs every round, which is why 2,048 tokens take a second here. Production trainers update pair counts incrementally after each merge, and train vocabularies of 100k+ tokens on gigabytes of text.

### 4.4 Encoding a chunk

<!-- file: src/lib.rs -->
```rust
    pub fn encode_chunk(&self, chunk: &str, out: &mut Vec<Token>) {
        let mut ids: Vec<Token> = chunk.bytes().map(Token::from).collect();
        while ids.len() >= 2 {
            let best = ids
                .windows(2)
                .filter_map(|p| self.ranks.get(&(p[0], p[1])).map(|&r| (r, p[0], p[1])))
                .min();
            let Some((rank, a, b)) = best else { break };
            merge_pair(&mut ids, a, b, 256 + rank);
        }
        out.extend_from_slice(&ids);
    }
```

Section 3.3 in code: look up every adjacent pair's rank, take the minimum (tuples compare element by element, so `min` picks the lowest rank), merge that pair everywhere in the chunk, repeat. `filter_map` skips pairs with no merge. The result is appended to a caller-provided output vector, so encoding a whole text builds one vector instead of one per chunk.

This is O(n²) in the chunk length. Chunks are words, typically under 20 bytes, so it does not matter, and production implementations use a priority queue over a linked list of tokens for long chunks.

### 4.5 Special tokens first

<!-- file: src/lib.rs -->
```rust
            let next_special = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|pos| (pos, s.len(), *id)))
                .min_by_key(|&(pos, len, _)| (pos, std::cmp::Reverse(len)));
```

Find the earliest special token in the remaining text (ties at the same position go to the longest, via `Reverse`). Everything before it is ordinary text, encoded normally; the special token itself becomes its single ID; then continue after it. This is simple and O(specials × text); real tokenizers build a single automaton (Aho-Corasick) that finds all special tokens in one pass.

### 4.6 The streaming decoder

<!-- file: src/lib.rs -->
```rust
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // The prefix up to `valid` is complete, valid UTF-8.
                    out.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("checked by valid_up_to"),
                    );
                    match e.error_len() {
                        // The remaining bytes are the *start* of a character
                        // that has not finished arriving: keep them.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // The bytes are invalid and can never become valid:
                        // replace them and keep going.
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }
```

- Append the new token's bytes to whatever was left over.
- If everything is valid UTF-8, emit it all.
- Otherwise emit the valid prefix, then look at `error_len()`: `None` means the rest is an unfinished character, so keep it and wait; `Some(n)` means `n` bytes are garbage, so emit `�`, drop them, and loop to check what follows.
- `drain(..valid)` removes the emitted bytes from the front of `pending`.

The `expect` is not a gamble: `valid_up_to()` is documented to return a length whose prefix is valid UTF-8, so that `from_utf8` cannot fail. The message states why.

`finish` handles the end of the stream: a character that never completed is replaced with `�` rather than silently dropped.

## 5. Run it

```bash
cargo test -p ch11-tokenization
cargo run --release -p ch11-tokenization
```

The full output on the reference machine is shown piece by piece in section 3; the remaining part:

```text
== 2. the first and last merges learned (vocab 2048)
   first: "  " " t" " a" "    " "he" "in" "er" "re" " s" " the"
   last:  "rr" "yst" "−" " apart" "release" " square" " cre" " cle" " fre" " fresh"
```

Note that `"−"` (a Unicode minus sign, 3 bytes) earned a token: it appears often enough in these lessons' math. A tokenizer is a portrait of its training corpus.

## 6. The Rust behind it

**`&str` is guaranteed UTF-8; `&[u8]` is not.** Every `&str` in Rust is valid UTF-8, which is why `chunk.bytes()` can be encoded freely and `text[start..end]` panics if the indices are not on character boundaries. Tokens are raw bytes (`Vec<u8>`), because a token may be part of a character. The code converts between the two only at the edges, with explicit handling of invalid sequences.

**`char_indices` and `len_utf8`.** Pre-tokenization walks characters with their byte offsets, so the chunks it returns are `&str` slices of the input: no copies. `c.len_utf8()` gives a character's byte length for computing the end of a slice.

**`std::str::from_utf8`'s error API** distinguishes incomplete input from invalid input (`error_len()`), which is exactly what a streaming decoder needs. Many languages' standard libraries do not expose this, and their streaming decoders end up reimplementing UTF-8 parsing.

**`HashMap` is randomly seeded.** Iteration order differs between runs, so any algorithm that picks "the first maximum" from a `HashMap` must break ties explicitly to be deterministic. `BTreeMap` iterates in key order if you need that instead.

**`String::from_utf8_lossy` returns a `Cow<str>`**: borrowed when the input is already valid (no copy), owned when replacements were needed. `.into_owned()` turns it into a `String` either way.

**Let chains** (`if c == ' ' && let Some(..) = chars.peek() && kind(next) != Kind::Space`) keep the "single space before a word" rule in one readable condition.

## 7. Mistakes you will make

- **Using a different tokenizer from the one the model was trained with**, or the right vocabulary with merges applied in the wrong order. The model still produces text, just worse. Always check token IDs against the reference implementation.
- **Getting the chat template slightly wrong.** A missing `\n` after the role name changes every token that follows.
- **Streaming raw per-token decodes.** Users see `�` in every accented word and emoji.
- **Counting characters instead of tokens** for limits and billing.
- **Letting user text become special tokens.** Encode user input so literal `<|im_start|>` strings stay ordinary text.
- **Tokenizing on the engine thread.** A 1 MB prompt then blocks every other request's generation while it is encoded.

## 8. How the professionals do it

- **Hugging Face `tokenizers`** is written in Rust (with Python bindings) and is what most open models ship with, as a `tokenizer.json` file containing the vocabulary, merges, pre-tokenizer regex and special tokens. Chapter 16 reads that file directly.
- **OpenAI's `tiktoken`** is a byte-level BPE with a Rust core, known for encoding speed.
- **SentencePiece** (used by Llama 1/2, Gemma, T5) implements BPE and the Unigram language-model algorithm on Unicode characters with byte fallback, and treats spaces as a visible character (`▁`).
- **Servers** such as TGI and vLLM run tokenization in separate threads or processes, and implement incremental detokenization that also handles tokenizers whose tokens change the spacing of their neighbours.

## 9. Exercises

1. **GPT-2's whitespace rule.** In GPT-2, a run of spaces before a word gives its *last* space to the word (`"a   b"` → `"a"`, `"  "`, `" b"`). Change `pre_tokenize` to do that, and check that `pre_tokenize(s).concat() == s` still holds.
2. **Another language.** Add Chinese (or another language you know) to the training corpus. At what share of the corpus does it start getting its own merges? What happens to English compression?
3. **A faster encoder.** Replace the O(n²) loop in `encode_chunk` with a doubly linked list of tokens plus a priority queue of candidate pairs keyed by rank. Test it against the current encoder on random text.
4. **Stop strings.** Write `fn find_stop(decoded_so_far: &str, stops: &[&str]) -> Option<usize>` for use with `StreamDecoder`, and explain why checking each token's text on its own is not enough.
5. **The embedding share.** For a model with vocabulary V and hidden size d, the embedding matrix has V·d parameters. Compute its share of SmolLM2-135M (V = 49,152, d = 576, 134.5M parameters) and of a hypothetical 8B model with V = 128,256 and d = 4,096.
6. **`error_len`.** What would go wrong if `StreamDecoder::push` treated every UTF-8 error as "incomplete, wait for more bytes"?

## 10. Check yourself

1. Why do modern tokenizers use bytes, rather than characters, as their base alphabet?
2. Why must merges be applied in rank order when encoding?
3. What problem does pre-tokenization solve?
4. Why is a request in a poorly covered language more expensive to serve?
5. Why can't a streaming server just decode each token separately?
6. Why does BPE training need an explicit tie-breaking rule in Rust?

## 11. Recap

- Models read and write token IDs. Byte-level BPE builds a vocabulary by merging frequent adjacent pairs, starting from the 256 bytes, so every input is encodable.
- Encoding applies merges in the order they were learned. A different order gives different IDs and a subtly worse model.
- Special tokens and chat templates frame conversations; they must match the model's training exactly.
- Token counts set the cost of everything: prefill, decode steps, KV memory, price. They depend heavily on language and content (3.5 bytes/token for English prose here, 1 for Chinese).
- Streaming output must reassemble UTF-8 characters split across tokens; `from_utf8`'s `valid_up_to` and `error_len` make that straightforward.
- Tokenization is a per-request CPU cost; cache chunk encodings and keep it off the engine thread.

## Answers

**Exercises**

1. When a whitespace run is followed by a non-space character and has more than one space, end the whitespace chunk one character early and let the last space start the next chunk (which then takes the "single leading space" path). The concatenation property holds because chunks are still consecutive slices; only the boundaries move.
2. Measured on the reference machine with a 2,048-token vocabulary: adding one short Chinese paragraph (about 600 bytes, 0.2% of the corpus) produced no Chinese merges at all, and the test phrase stayed at 27 tokens for 27 bytes. Repeating that paragraph until Chinese was 5.5% of the corpus brought the phrase down to 5 tokens (optimistic, since its words appear in the training paragraph), while English compression dropped from 3.08 to 2.90 bytes per token, because merges spent on Chinese are no longer available for English. Vocabulary is a shared, fixed budget.
3. Keep tokens in a linked list (`Vec` of nodes with `prev`/`next` indices), and a binary heap of `(rank, position)` for every adjacent pair that has a merge. Pop the lowest rank, skip it if either node has since been merged away, merge, and push the new pairs formed with the neighbours. Each merge is O(log n), so a chunk of length n costs O(n log n). Test by encoding thousands of random strings with both encoders and comparing outputs.
4. Keep the decoded text so far (or at least its last `max_stop_len − 1` characters plus the new piece) and search for each stop string in that window. A stop string such as `"\n\n"` can be split as `"\n"` at the end of one token and `"\n"` at the start of the next, so no single token's text contains it. Also note that text that *might* be the start of a stop string should be held back from the client until it is resolved, or the client will see part of the stop string.
5. SmolLM2: 49,152 × 576 = 28.3M, which is 21% of 134.5M. For the 8B example: 128,256 × 4,096 = 525M, about 6.6% of 8B (13% if the output layer is a separate, untied matrix of the same size). Large vocabularies weigh heavily on small models.
6. An invalid byte (for example `0xFF`, which can never appear in UTF-8) would be held forever, waiting for bytes that can never make it valid, and every later token would pile up behind it: the stream would go silent for the rest of the response.

**Check yourself**

1. Bytes give a small, fixed base alphabet (256) that can represent any input, including every Unicode character, emoji and binary garbage, so there is never an "unknown" token.
2. Because each merge was learned on text that had already been transformed by all earlier merges. Applying them in another order produces token sequences the model never saw during training.
3. It stops merges from crossing word and category boundaries, so the vocabulary is not wasted on fragments like "e c" that exist only because two words happened to be adjacent.
4. Its text is split into more tokens, so it needs more prefill computation, more decode steps for the same output length, more KV cache memory, and costs more in any per-token pricing.
5. A token can end in the middle of a multi-byte UTF-8 character. Decoding it alone produces invalid UTF-8 (shown as `�`); the bytes must be held until the character completes.
6. Rust's `HashMap` iterates in a random order that changes between runs. Without an explicit tie-break, equally frequent pairs could be chosen differently each time, giving different tokenizers from the same corpus.

## Further reading

- Sennrich, Haddow and Birch, "Neural Machine Translation of Rare Words with Subword Units", 2016: BPE for NLP.
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019: byte-level BPE and its pre-tokenization regex.
- Andrej Karpathy's `minbpe` repository and lecture "Let's build the GPT Tokenizer": a clear walkthrough of the same algorithm.
- Next: [Chapter 12: Attention](../12-attention/README.md). Token IDs become vectors, and vectors look at each other.
