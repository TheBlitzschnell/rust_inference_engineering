# Chapter 27: Structured output

> **In one sentence:** a program that consumes a model's answer needs it in an exact format, which a model asked nicely produces only most of the time; constrained decoding guarantees it by running a parser alongside generation and, at every step, forbidding every token that would make the text unparseable, so that the model chooses freely among the continuations that stay valid.

**Where this fits:** chapters 21 to 26 made generation fast. This chapter changes what is generated: not faster or slower, but always in a shape another program can read. It is the mechanism behind "JSON mode", "structured outputs" and tool calling in every inference API.

**You need:** chapter 11 (tokens are byte strings; a token can split a character), chapter 15 (sampling from logits, why `-inf` removes a token), chapter 16 (SmolLM2's tokenizer and chat template).

**You will build:** a byte-level JSON matcher that follows a schema (fixed keys in order, strings with a maximum length, integers, numbers, booleans, enums, arrays, or any JSON), and that can always say how to finish what it has read; token masks computed by testing every token, and by walking a trie of the vocabulary; and constrained generation, which closes the JSON itself if the token budget runs out. Then measurements with SmolLM2-135M and -360M: extracting fields from ten short texts with and without constraints, what a mask costs, and a budget too small for the answer.

---

## 1. The intuition

A form with boxes, instead of a blank page. Ask someone to write down a person's name, age and city, and you get "Maria, 34, from Madrid", or "Name: Maria Lopez (age 34), Madrid, Spain", or a paragraph. Hand them a form with three labelled boxes, and the age box that only takes digits, and every answer can be read by a machine. The person still decides what goes in each box. The form only decides where boxes are and what may go in them.

**Where the analogy breaks:** the model writes in tokens, not characters. One token can close a string and add a comma at once (`",` is one SmolLM2 token; so are `":` and `{"`), so "what may go in the box" has to be decided token by token, for whole multi-character pieces, against a parser that thinks in characters.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Structured output** | Model output that must follow a format exactly: JSON, a JSON schema, a regular expression, a grammar. |
| **Constrained decoding** | Restricting, at every step, which tokens may be sampled, so that the output follows the format. |
| **Token mask** | For one step, the set of tokens that may come next. The others get a logit of minus infinity. |
| **JSON Schema** | A JSON description of what a JSON document may contain: types, required properties, enums, lengths. |
| **Pushdown automaton** | A state machine with a stack: enough to follow nested structure (objects in arrays in objects), which a plain state machine cannot. |
| **Trie** | A tree of strings in which strings with a common beginning share a path. |
| **Completion** | Here: the shortest text that turns a valid prefix into a complete valid document. |
| **Jump-forward decoding** | Appending text the format forces (a fixed key, a closing bracket) without asking the model for each token. |

## 3. The concepts in depth

### 3.1 Asking is not enough

The usual first attempt is to ask: "Reply with only a JSON object with the keys name, age and city." Part 1 does that with SmolLM2, for ten short texts such as *"Kenji Watanabe, 52, teaches history in Osaka."*, decoding greedily:

```text
== 1. extract {name, age, city} as JSON from ten short texts (greedy)
                                        valid JSON:          fields  tokens   per decoded token:
                                        whole  contains  conforming right   each      total    masking
   SmolLM2-135M on "Maria Lopez is a 34-year-old nurse who lives in Madrid."
      unconstrained:          "JSON object:\n\n```json\n{\n  \"name\": \"Maria Lopez\",\n  \"age\": 34,\n  \"city\": \"Madrid\"\n}\n```\n\nThis JSON object contains the following keys:\n\n- `name`: Maria Lopez is a 34-year-old nurse.\n- `age`: 34 years old.\n- `city`: Madrid."
      constrained:            "{\"name\": \"Maria Lopez\", \"age\": 34, \"city\": \"Madrid\"}"
      constrained, not asked: "{\"name\": \"Maria Lopez\", \"age\": 34, \"city\": \"Madrid\"}"
   SmolLM2-135M, free                        0       10        10     30/30    78.5      11.2 ms     0.00 ms
   SmolLM2-135M, constrained                10       10        10     30/30    26.8      12.0 ms     0.12 ms
   SmolLM2-135M, constrained, not asked     10       10        10     18/30    63.6      11.1 ms     0.08 ms
      e.g. not asked, on "Kenji Watanabe, 52, teaches history in Osaka.":
      "\t\t\t\t\t\t\t\t\t\t\t\t{\n\t\t\t\t\t\t\t\t\t\t\t\"name\"\n\t\t\t\t\t\t\t\t\t\t\t: \t\t\"Kenji Watanabe\"\n\t\t\t\t\t\t\t\t\t\t\t, \t\t\"age\"\n\t\t\t\t\t\t\t\t: \t\t-52\n\t\t\t\t\t\t,\"city\":\"\"}"
   SmolLM2-135M, not asked, compact         10       10        10     30/30    22.4      11.7 ms     0.10 ms
   SmolLM2-360M, free                       10       10        10     30/30    29.4      30.8 ms     0.00 ms
   SmolLM2-360M, constrained                10       10        10     30/30    29.4      31.4 ms     0.11 ms
   SmolLM2-360M, constrained, not asked     10       10        10     30/30    30.7      30.7 ms     0.11 ms
   SmolLM2-360M, not asked, compact         10       10        10     28/30    22.6      30.1 ms     0.10 ms
```

The columns: whether the whole answer parses as JSON; whether it contains a JSON object somewhere (for example inside a Markdown code block, which a lenient client could cut out); whether that object has exactly the three keys with the right types; how many of the 30 fields (3 per text) are right; tokens per answer; and time per decoded token (after the prompt), with the part of it spent computing masks. This section is about the `free` rows; the others come up in sections 3.2 and 3.5.

SmolLM2-135M knows the answers (once the JSON is cut out of its reply, every field is right) but never replies with JSON alone: it introduces it ("JSON object:"), wraps it in a Markdown code block, and explains it afterwards, 78 tokens per answer. Not one of its ten replies parses. SmolLM2-360M follows the instruction on all ten. A client that wants to use the 135M must search the reply for something that looks like JSON, and hope.

Larger models follow format instructions much better, and API providers fine-tune for it, but "much better" is not "always". A program that calls a model thousands of times an hour sees every way of almost following an instruction. The fix is to make invalid output impossible rather than unlikely.

### 3.2 Constrained decoding

At each step the model produces logits for every token (chapter 15). Constrained decoding inserts one operation between the logits and the sampler: set the logit of every token that would break the format to minus infinity. Its probability becomes exactly zero, and the sampler (greedy, temperature, top-p, anything) chooses among the rest with their relative probabilities unchanged. The model is never asked to "try again": it simply cannot produce an invalid token, so the answer is valid by construction.

Three things are needed:

1. **A matcher** that reads text and can say, after any prefix, whether that prefix can still become a valid document (section 3.3).
2. **A token mask** for each step: which tokens' bytes the matcher accepts next (section 3.4).
3. **A way to end.** The end-of-turn token is allowed only when the matcher says the document is complete; and if the token budget runs out first, the generator appends the shortest text that completes the document (section 3.6).

In part 1's `constrained` rows, every answer parses, has exactly the three fields, and all 30 values are right, for both models. The 135M's answers also became three times shorter (27 tokens instead of 78): the constraint ends the answer at the closing brace, so the explanation is never generated. The cost of masking is 0.1 ms per token, against 11 to 12 ms for the 135M's decode step and 30 ms for the 360M's (section 3.4 explains how it got that low).

### 3.3 A matcher for JSON

JSON nests: an object can contain an array of objects. A finite state machine cannot count unclosed brackets to any depth; a **pushdown automaton**, a state machine with a stack, can. The matcher's stack has one frame per value being read:

```text
input so far:   {"name": "Maria", "tags": ["a", "b
stack:          Object { property 2 of 2, reading its value }
                Array  { after a comma }
                String { 1 byte so far }
```

Each frame is a small state machine: an object frame knows which property comes next and whether a key, a colon, a value, a comma or the closing brace is expected; a string frame tracks escapes (`\n`, `é`) and its length; a number frame follows JSON's number syntax (no leading zeros, a digit after the point, an optional exponent). A byte is fed to the top frame, which accepts it, rejects it, finishes (popping itself), or starts a child (pushing one). A number has no closing character: it ends at the first byte that cannot continue it, and that byte is handed to the frame below.

A **schema** turns the general JSON matcher into a specific one. `Schema::Object(vec![("name", String), ("age", Integer), ("city", String)])` makes the object frame accept exactly those keys in that order, and start a value frame of the right type after each colon. Schemas here are a small subset of JSON Schema: fixed properties in a fixed order (all required), strings with a maximum length, integers, numbers, booleans, null, enums of strings, arrays, and "any JSON". Production engines support most of JSON Schema, and full context-free grammars (section 8).

Three limits protect the generation rather than the format: at most 12 whitespace bytes in a row (configurable, down to none), at most 15 digits in a number's integer or fraction part, and at most 2 in its exponent (so that every number fits an `f64`; `1e999` is valid JSON that most parsers reject). Beyond those limits the text is still valid JSON; but a model that can emit whitespace or digits forever sometimes does, and a constraint that permits it lets a small model fill its budget with spaces (section 3.5 shows one doing it).

The tests compare the matcher with `serde_json` (a production JSON parser) on 200,000 random short strings of JSON characters: they agree on every one (apart from exponents of 3 digits or more, which the matcher refuses on purpose).

### 3.4 Token masks

For each step, the mask is the set of tokens all of whose bytes the matcher accepts, in order, from its current state. The straightforward way tests every token: copy the matcher, feed the token's bytes, keep the token if none is rejected. For SmolLM2's 49,152 tokens, that is about 300,000 bytes fed per step.

Tokens share beginnings: `"`, `",`, `"name`, `"age`, `":` all start with a quote; ` the`, ` then`, ` there` share ` th`. A **trie** of the vocabulary puts each shared beginning on one path. Walking it, the matcher is copied and fed once per edge; when an edge's byte is rejected, the whole subtree below it (every token starting that way) is skipped at once. Part 2 measures both in four states of the person schema, with a third method, explained after the table:

```text
== 2. one token mask over 49135 tokens (trie of 100298 nodes), single thread
   after                                  allowed   each token     trie walk     with shortcut
   ""                                          84      1.56 ms       0.01 ms          0.01 ms
   "{\"name\": \""                          48768      5.03 ms       7.43 ms          0.07 ms
   "{\"name\": \"Maria Lopez\", \"age\": "      90      1.97 ms       0.01 ms          0.01 ms
   "{\"name\": \"Maria Lopez\", \"age\": 34"      93      2.17 ms       0.02 ms          0.01 ms
```

Testing each token costs 1.5 to 5 ms per step, a large share of the 135M's 11 ms decode step. The trie walk is almost free wherever few tokens are possible (84 to 93 allowed tokens; the walk tries about 500 matcher states and skips everything else), but *inside a string* it is slower than testing each token: nearly every token (48,768 of 49,135) is allowed there, so the walk visits nearly all 100,298 nodes and copies the matcher at each.

The fix is to notice that inside a string, the answer for most tokens does not depend on the matcher at all. A token that is plain text (whole UTF-8 characters, no quote, no backslash, no control character) is allowed exactly when it fits in the string's remaining length. That property is computed once, when the trie is built; at run time only the other tokens (those containing a quote, a backslash or part of a character, about 3,000) go through the matcher. Inside a string, the mask drops from 7 ms to 0.07 ms. This is the idea production engines generalize: split the vocabulary, for each state of the grammar, into tokens whose fate is decided by the state alone (precomputed) and tokens that must be checked at run time (section 8).

The matcher is `Copy`: a fixed array of frames that refer to the schema rather than own parts of it. Copying it for each edge costs a memory copy of a few hundred bytes, no allocation, which is what makes thousands of copies per step affordable.

### 3.5 What constraints do to the answers

A constraint changes the distribution the model samples from, not the model. When the model's preferred continuation is invalid (a code fence, a sentence), the probability moves to the valid tokens, and the model continues from whatever it was forced into. Usually that is what we want: the model "knew" the answer and only the format was wrong. Sometimes the forced token takes the model somewhere it would not have gone, and the values it writes inside a valid structure are worse.

Part 1's third row shows it. The prompt no longer mentions JSON ("Who is the person in this text, how old are they, and where do they live?"), and the constraint alone imposes the format. SmolLM2-360M copes: 30 of 30 fields right. SmolLM2-135M gets 18: the example shows a model that was about to write a sentence and finds only `{`, whitespace, or tabs allowed. It spends its tokens on runs of tabs (each run within the 12-character limit, but there are many), writes `-52` for an age of 52, and runs out of budget before the city, which the matcher's completion fills with an empty string.

The fourth row allows no whitespace at all outside strings (`Matcher::with_max_whitespace(&schema, 0)`): compact JSON. The 135M's answers become correct again, 30 of 30, in 22 tokens. The 360M loses two fields with the same change. A constraint is part of the model's input in effect, and whether a tighter one helps depends on the model.

Two practices reduce the risk. Tell the model the format in the prompt as well, so that its own preferences and the constraint agree (the prompts here do). And design schemas whose fields follow the order a person would write them in: an answer forced to state a conclusion before its reasoning reasons worse.

### 3.6 Ending, and running out of budget

The end-of-turn token is allowed only when the matcher's document is complete; until then it is masked like any other invalid token, and the model cannot stop early. The opposite problem is a budget too small for the answer: `max_tokens` runs out in the middle of a string, and the output is cut. A server that promises valid JSON must still deliver it.

The matcher knows the shortest way to finish: for each frame on its stack, from the top down, the fewest bytes that close it validly (a closing quote for a string, `0` for a number that needs a digit, the remaining required properties with their shortest values for an object, a closing bracket for an array). Part 3 gives the extraction request a budget of 8 tokens:

```text
== 3. the same request with a budget of 8 tokens
   generated "{\"name\": \"Maria Lopez\", \"", completed with "age\":0,\"city\":\"\"}"
```

The result is valid and follows the schema, though the values the model never got to write are placeholders. OpenAI's API, for comparison, returns `finish_reason: "length"` and incomplete JSON in that case; completing it is a design choice, worth making visible to the client (here `Output::completed` counts the bytes the matcher added).

## 4. The code

The matcher is in [`src/json.rs`](src/json.rs), the masks in [`src/mask.rs`](src/mask.rs), generation in [`src/constrained.rs`](src/constrained.rs), the demo in [`src/main.rs`](src/main.rs).

### 4.1 Feeding a byte

<!-- file: src/json.rs -->
```rust
            let top = self.stack[self.depth - 1].expect("frames below depth exist");
            let step = self.step(top, b);
            // ...
            match step {
                Step::Stay(f) | Step::Replace(f) => self.stack[self.depth - 1] = Some(f),
                Step::Push(f, child) => {
                    if self.depth == MAX_DEPTH {
                        return false;
                    }
                    self.stack[self.depth - 1] = Some(f);
                    self.stack[self.depth] = Some(child);
                    self.depth += 1;
                }
                Step::Pop => self.pop(),
                Step::PopAndRetry => {
                    self.pop();
                    continue;
                }
```

The top frame decides what happens to the byte; the matcher only applies the decision to the stack. `PopAndRetry` is the number's case: the byte that ends a number (a comma, a bracket, a space) is not part of it, so the number frame is removed and the same byte goes to the frame below, which expects exactly such a byte. Keys of fixed objects are matched by a separate small state (`key`), so that the object frame's stack slot is not needed for them.

### 4.2 Strings are UTF-8

<!-- file: src/json.rs -->
```rust
        Expect::Char => match b {
            b'"' => Step::Pop,
            b'\\' => next(Expect::Escape, 2),
            0x20..=0x7f => next(Expect::Char, 1),
            0xc2..=0xdf => utf8(1, 0x80, 0xbf),
            0xe0 => utf8(2, 0xa0, 0xbf),
            0xed => utf8(2, 0x80, 0x9f),
            0xe1..=0xef => utf8(2, 0x80, 0xbf),
            0xf0 => utf8(3, 0x90, 0xbf),
            0xf1..=0xf3 => utf8(3, 0x80, 0xbf),
            0xf4 => utf8(3, 0x80, 0x8f),
```

A byte-level tokenizer has tokens that are parts of characters (chapter 11), so the matcher must check UTF-8 itself, or a token ending in the middle of a character followed by a closing quote would produce invalid text. The first byte of a character says how many continuation bytes follow, and the second byte's allowed range excludes the encodings UTF-8 forbids (overlong forms after `0xe0` and `0xf0`, UTF-16 surrogates after `0xed`, code points beyond U+10FFFF after `0xf4`). The second argument of `next` is the room the whole character or escape needs, so that a string never starts something it cannot finish within its maximum length; the completion can then always close it.

### 4.3 Walking the trie

<!-- file: src/mask.rs -->
```rust
    fn walk_from(&self, node: usize, matcher: &Matcher<'_>, mask: &mut [bool], tried: &mut usize) {
        for &(b, child) in &self.children[node] {
            let mut m = *matcher;
            *tried += 1;
            if m.push(b) {
                for &t in &self.tokens[child as usize] {
                    mask[t as usize] = true;
                }
                self.walk_from(child as usize, &m, mask, tried);
            }
        }
    }
```

Depth-first: a copy of the matcher for each edge, the edge's byte pushed into it; if accepted, every token ending at that node is allowed and the walk continues below with the advanced copy; if rejected, the whole subtree is skipped. `*matcher` copies because `Matcher` is `Copy` (section 6). Inside a string, `allowed` takes the shortcut instead:

<!-- file: src/mask.rs -->
```rust
        let Some(room) = matcher.string_room() else {
            return self.walk(matcher, mask);
        };
        for &(t, len) in &self.plain {
            if len <= room {
                mask[t as usize] = true;
            }
        }
```

### 4.4 One constrained step

<!-- file: src/constrained.rs -->
```rust
        mask.fill(false);
        vocab.trie.allowed(&matcher, &mut mask);
        mask[vocab.eos as usize] = matcher.is_complete();
        for (l, &ok) in logits.iter_mut().zip(&mask) {
            if !ok {
                *l = f32::NEG_INFINITY;
            }
        }
        out.mask_time += start.elapsed();
        let token = sampler.sample(&logits);
```

The sampler is chapter 15's, unchanged: it never learns that some tokens were removed. Special tokens never enter the trie (they stand for markup, not their text), and the end-of-turn token is allowed only once the value is complete. After sampling, the token's bytes are pushed into the matcher, which must accept them (an assertion checks it); when the budget runs out, `completion()` closes what is open.

## 5. Run it

```bash
cargo test -p ch27-structured-output
cargo run --release -p ch27-structured-output                # all three parts, about 1 minute
cargo run --release -p ch27-structured-output -- mask        # or: extract, budget
```

The tests check that ten valid JSON documents (including escapes, UTF-8, a root number, pretty-printing) are accepted byte by byte and end complete; that seventeen invalid ones are rejected; that the matcher agrees with `serde_json` on 200,000 random strings over JSON's characters; that a schema fixes keys, their order, types, enum values and lengths; that for every prefix of four documents, the prefix plus its completion is valid and follows the schema; that strings are valid UTF-8 (stray continuation bytes, a quote in the middle of a character, overlong encodings and surrogates are rejected, as `serde_json` rejects them); that a string never starts an escape it has no room to finish; that the three ways of computing a mask (each token, the trie walk, the trie with the string shortcut) agree in every state reached while reading two documents, over 3,000 random tokens of up to 5 bytes that include parts of UTF-8 characters; that a tiny model with random weights, sampling at temperature 1, always produces JSON that follows a six-field schema (object, integer, boolean, enum, array, string, any); and that budgets of 0 to 12 tokens are completed into valid JSON.

## 6. The Rust behind it

**A `Copy` automaton.** `Matcher` is a fixed array of `Option<Frame>` plus a few counters, and every `Frame` holds references into the schema (`&'s Schema`, `&'s [(String, Schema)]`) rather than owned data. So `#[derive(Clone, Copy)]` applies, and trying a byte on a copy (`let mut m = *matcher;`) is a memory copy of under a kilobyte, with no allocation. A `Vec<Frame>` stack would be more flexible and would allocate on every copy, which the mask does thousands of times per step.

**Lifetimes tie the matcher to its schema.** `Matcher<'s>` cannot outlive the `Schema` it reads, and the compiler checks it: a schema built in a function and dropped while a matcher still points into it does not compile.

**`u64` as a set.** An enum's options still possible are a bit mask (`alive`); intersecting with the options that match the next byte is a loop over at most 64 bits, and the empty set is `0`. It limits enums to 64 options, which `Schema::check` asserts.

**`let ... else` for the common path.** `let Some(room) = matcher.string_room() else { return self.walk(matcher, mask); };` handles the exception (not in a string) first and leaves the main path unindented.

**`Option::is_none_or`** reads as the rule it checks: "no object found, or an object with a wrong field".

**Byte strings for bytes.** The UTF-8 tests use `b"\"\xc3\xa9\""` literals: a Rust `&str` cannot hold invalid UTF-8, which is exactly what those tests need to feed the matcher.

## 7. Mistakes you will make

- **Masking characters instead of tokens.** A token is checked against all of its bytes: `",` closes a string *and* adds a comma; accepting it because `"` is valid lets the comma through unchecked.
- **Forgetting the end-of-turn token.** If it is never allowed, the model cannot stop even when the document is complete; if it is always allowed, it stops in the middle.
- **Treating special tokens as text.** `<|im_end|>` is 10 bytes of valid string content to a byte-level matcher. Special tokens must be kept out of the mask.
- **Unbounded whitespace and digits.** Valid, and an easy way for a small model to waste its budget.
- **A budget that cuts the JSON.** Complete it, or report clearly that the output is incomplete.
- **Schema fields in an order that forces conclusions before reasoning.**
- **Checking the output afterwards and retrying.** It works, at the cost of a whole generation per failure, and can fail every time.
- **Recomputing the mask from scratch for large grammars.** For big schemas and vocabularies, the mask is the bottleneck; precompute what does not depend on the state (section 8).

## 8. How the professionals do it

- **Outlines** (Willard and Louf, 2023) compiles regular expressions and JSON schemas into finite state machines and precomputes, for every state, the tokens allowed, so that a step's mask is a lookup. It is used inside vLLM, TGI and others.
- **XGrammar** (Dong et al., 2024) handles full context-free grammars with a pushdown automaton, and splits tokens into *context-independent* ones (decided by the grammar position alone, precomputed into bitmasks) and *context-dependent* ones checked at run time, as section 3.4's shortcut does for strings. vLLM and SGLang adopted it as their default structured-output backend in 2025; it computes the mask on the CPU while the GPU runs the forward pass, so that the mask is ready when the logits are.
- **llguidance** (Microsoft, written in Rust) computes masks for context-free grammars with a lexer and an Earley parser, in tens of microseconds per token for large vocabularies; vLLM offers it as its `guidance` backend, and llama.cpp can be built with it.
- **llama.cpp** has GBNF grammars and converts JSON schemas into them (`--json-schema`).
- **SGLang** introduced *jump-forward* decoding: when the grammar forces a string (a key, a closing bracket), it is appended without sampling, and processed in one forward pass.
- **APIs**: OpenAI's "Structured Outputs" (`response_format` with `json_schema` and `strict: true`) and tool calling with strict schemas, Anthropic's tool use, Google's `response_schema`: all constrain decoding on the server. OpenAI documents that a response cut by `max_tokens` can still be incomplete JSON.

## 9. Exercises

1. **Jump-forward.** After `{` the person schema forces `"name": "` (up to whitespace). Find the text the matcher forces at each step (the bytes for which exactly one is accepted), encode it with the tokenizer and feed it to the model in one forward pass, as a prompt chunk, instead of sampling it token by token. Measure the tokens saved on part 1, and check whether the answers change. Why might they?
2. **Cache the masks.** In a free string with plenty of room, the mask is the same at every step. Cache masks by a description of the matcher's top frame, and measure part 1's masking time again.
3. **Optional properties.** Extend `Schema::Object` with properties that may be left out. What changes in the object frame, and in the shortest completion?
4. **Regular expressions.** Write a matcher for a regular expression such as `[0-9]{3}-[0-9]{4}` (a state machine without a stack is enough) and use it with the same masks and generator.
5. **Tool calls.** A tool call is an object `{"name": <one of the tools>, "arguments": <that tool's schema>}`. The arguments' schema depends on the name. How would you express that dependency in the matcher?

## 10. Check yourself

1. Why does setting a logit to minus infinity remove a token without changing the relative probabilities of the others?
2. Why can a finite state machine not check JSON, and what does the stack add?
3. How does a number frame know that the number has ended?
4. Why must a token be checked against all of its bytes?
5. What does the trie save, compared with testing each token?
6. When may the end-of-turn token be sampled?
7. What happens if the budget runs out in the middle of a string, here and in an API that does not complete the JSON?
8. How can a constraint make the values in a valid answer worse?

## 11. Recap

- Asked for JSON only, SmolLM2-135M never replied with JSON alone (0 of 10 parsed); SmolLM2-360M did every time. Asking is not a guarantee.
- Constrained decoding sets the logits of tokens that would break the format to minus infinity before sampling; the output is valid by construction (10 of 10 for both models, all fields right), and shorter when the model would have added prose.
- A pushdown automaton (a stack of frames) follows JSON's nesting; a schema fixes keys and types; UTF-8, escapes, number syntax and limits on whitespace and digits are checked byte by byte. It agreed with `serde_json` on 200,000 random strings.
- A token is allowed if the matcher accepts all of its bytes. Testing each token cost 1.5-5 ms per step; a trie made most states nearly free but was slower inside strings, where a precomputed shortcut (plain tokens are allowed if they fit) brought the mask to 0.07 ms. Masking cost 0.1 ms per token in generation.
- Constraints change what the model writes: not asked for JSON, the 135M forced into it got 18 of 30 fields right, padding with tabs; forbidding whitespace brought it back to 30, and cost the 360M two fields.
- The end-of-turn token is allowed only when the value is complete; when the budget runs out, the matcher's shortest completion closes the value.

## Answers

**Exercises**

1. Find forced bytes by trying all 256 bytes on a copy of the matcher; while exactly one is accepted, append it and continue. With whitespace allowed, very little is forced (whitespace is an alternative almost everywhere); with compact JSON, `"name":"` after `{`, `","age":` after the name's closing quote, and so on are forced. Feeding the forced text's tokens in one pass saves one decode step per forced token after the first. The answers can change, because the tokenizer's encoding of the forced text (`","` as one token, say) may differ from the tokens the model would have chosen to write it, and the model then sees a token sequence it rarely saw in training; production engines handle this carefully ("token healing").
2. A key such as "inside a free string, with at least the longest token's length of room" describes a state whose mask never changes; cache it once. Part 1's masking time, already 0.1 ms per token, drops further inside strings, and stays the same elsewhere (where the walk is already cheap). For grammars where many states share masks, caching by state is the main technique (Outlines precomputes all of them).
3. Each property gets a flag `required`. After a value, the object frame may accept a comma followed by the key of any later property whose earlier optional ones are skipped, or the closing brace if every remaining property is optional; with fixed order, the set of possible next keys is the next required property and the optional ones before it, so a key frame must match one of several names (like an enum). The shortest completion skips optional properties.
4. A regular expression compiles to a finite state machine (no stack): the matcher's state is one integer, `push` is a table lookup, `completion` is a shortest path to an accepting state. The mask and the generator are unchanged, since they only use `push`, `is_complete` and `completion`.
5. The arguments' schema is chosen by the name: an object frame whose second property's schema depends on the value of the first. One way is to read the name as an enum and, when it completes, push a value frame with the schema of the tool whose name matched (the enum frame knows which option completed). Grammar engines express it as a choice between one rule per tool: `{"name": "search", "arguments": <search args>} | {"name": "weather", ...}`.

**Check yourself**

1. The sampler turns logits into probabilities with a softmax: `exp(-inf) = 0`, so the token's probability is exactly zero, and the others are divided by the same (smaller) sum, which keeps their ratios.
2. A finite state machine has a fixed number of states, so it cannot remember how many brackets are open when the nesting is arbitrarily deep. The stack remembers, for every open value, what kind it is and where in it the input is.
3. It does not, until the next byte: a byte that cannot continue the number (a comma, a bracket, whitespace) ends it if the number is complete at that point; the number frame is removed and the byte is given to the frame below.
4. Because a token can contain several characters of different roles (`",` closes a string and adds a comma); accepting a token after checking only its first byte could let through text the format forbids.
5. Tokens that start with the same bytes share a path, so the matcher processes each shared prefix once instead of once per token; and when a prefix is rejected, every token below it is ruled out without being looked at.
6. Only when the matcher says the value is complete; before that, its logit is set to minus infinity like any other token that cannot come next.
7. Here, the matcher appends the shortest text that closes everything open (a quote, missing properties with minimal values, brackets) and reports how many bytes it added; the result is valid but its last values are placeholders. An API that does not complete returns the cut text, which does not parse, with `finish_reason: "length"`.
8. The constraint removes the model's preferred continuation and spreads its probability over the allowed ones; if none of them is what the model "meant", it continues from a token it did not choose, and later choices are made in a context it would not have produced (as in part 1: tabs, a negative age, an empty city).

## Further reading

- Willard and Louf, "Efficient Guided Generation for Large Language Models", 2023 (Outlines: regular expressions and grammars compiled to token masks).
- Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models", 2024.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", 2024 (compressed finite state machines and jump-forward decoding).
- Tam et al., "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models", 2024.
- The JSON specification, RFC 8259, and the JSON Schema specification (json-schema.org).
- Next: [Chapter 28: More than one device](../28-parallelism/README.md). Splitting a model across devices when one is not enough.
