# Chapter 22: An HTTP server

> **In one sentence:** an inference server is a thin, careful layer in front of the engine: it speaks the API every client already knows (OpenAI's chat completions), streams tokens as server-sent events, refuses work it cannot take instead of letting a queue grow without end, cancels what nobody is waiting for any more, and shuts down without cutting off the requests in progress.

**Where this fits:** chapter 21 put the model on its own thread behind channels. This chapter puts HTTP in front of those channels, so that any program on the network can use the model. Chapter 23 then makes the engine serve many of these requests at once.

**You need:** chapter 21 (the engine thread and its events), chapter 15 (sampling parameters, stop strings), chapter 11 (decoding a token stream without splitting UTF-8 characters). Some familiarity with HTTP and JSON; the chapter explains the parts that matter.

**You will build:** an [axum](https://github.com/tokio-rs/axum) server with `POST /v1/chat/completions` (whole answers and streamed ones), `GET /v1/models` and `GET /health`; request validation with OpenAI-style errors; admission control with `429 Too Many Requests`; cancellation when a client disconnects; graceful shutdown on Ctrl-C or `SIGTERM`; a minimal HTTP client to see the bytes on the wire; and tests that run all of it over real sockets.

---

## 1. The intuition

Chapter 21's kitchen now gets a front desk. The host at the desk greets guests in a language they all speak, writes their orders on tickets, and brings each course to the table as it comes out of the kitchen.

When every table is taken and a few people are already waiting, the host says "we're full, try again in a minute" to the next guest. That sounds unfriendly, but the alternative is worse: a crowd at the door that waits an hour, most of whom give up and leave, while the kitchen still cooks for the ones who left. When a guest walks out, the host tells the kitchen to stop their order. At closing time, the host locks the front door but lets the guests already inside finish.

**Where the analogy breaks:** a host can guess how long the wait will be by looking around. A server cannot see how long each request will take (it depends on how many tokens the model will generate), so the limit here is a count of requests, which is only a rough proxy for the work behind them.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **OpenAI-compatible** | Accepting the same JSON requests and returning the same responses as OpenAI's API, so that its SDKs and tools work unchanged. vLLM, TGI, llama.cpp, Ollama all offer this. |
| **SSE (server-sent events)** | A streaming format over one HTTP response: `data: ...` lines, each event ended by a blank line. Content type `text/event-stream`. |
| **Chunked transfer encoding** | HTTP/1.1's way of sending a body whose length is not known in advance: each piece is preceded by its size. |
| **Admission control** | Deciding at the door whether to accept a request, based on current load. |
| **429 Too Many Requests** | The HTTP status for "not now"; a `Retry-After` header says how many seconds to wait. |
| **Graceful shutdown** | Stop accepting new connections, finish the requests in progress, then exit. |
| **`SIGTERM`** | The signal process managers (`docker stop`, Kubernetes, systemd) send to ask a program to shut down, before forcing it with `SIGKILL`. |
| **Health check** | An endpoint that load balancers and orchestrators call to see whether the server is alive and ready. |

## 3. The concepts in depth

### 3.1 Why OpenAI's API

The API itself is ordinary: a JSON list of messages in, a message out. What makes it the right choice is that thousands of programs already speak it: the official SDKs in every language, command-line tools, editors, agent frameworks. Nearly all of them let you change the base URL. A server that accepts the same JSON is usable by all of them on day one.

This server implements the part that matters for text chat:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in French."}],"max_tokens":16,"temperature":0}'
```

```json
{"id":"chatcmpl-1","object":"chat.completion","created":1790759352,"model":"SmolLM2-135M-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Bonjour, merci pour notre aide. Je suis de nombre"},"finish_reason":"length"}],"usage":{"prompt_tokens":35,"completion_tokens":16,"total_tokens":51}}
```

(SmolLM2-135M's French is about as good as its size suggests.) `finish_reason` is `"length"` because the answer hit `max_tokens`; it would be `"stop"` at the end-of-turn token or a stop string. `usage` counts tokens, which is what API providers bill.

Accepted: `messages`, `max_tokens` (or its newer name `max_completion_tokens`), `temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `n` (only 1), `stream`, `stream_options.include_usage`. Unknown fields are ignored, as other compatible servers do, because clients send fields a small server has no use for.

### 3.2 The life of a request

```text
parse JSON ─► validate ─► encode prompt ─► admit ─► submit to engine ─► relay events ─► respond
   400          400           400           429           503             (stream or whole)
```

Each step can refuse, and each refusal has its own status code. The order matters: everything that can be checked cheaply (malformed JSON, unknown roles, a prompt that does not fit the context) is checked *before* the request takes a place, so a bad request never occupies capacity. Errors use OpenAI's shape, so client libraries can show them:

```text
HTTP/1.1 400 Bad Request
content-type: application/json

{"error":{"message":"the prompt is 31 tokens and `max_tokens` is 5000; together they exceed this model's context of 2048","type":"invalid_request_error","code":"context_length_exceeded"}}
```

`context_length_exceeded` is the code OpenAI uses; clients check for it to shorten a conversation and retry.

### 3.3 Streaming with server-sent events

With `"stream": true`, the response is `text/event-stream`, and each event is one JSON "chunk":

```text
data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1790759353,"model":"SmolLM2-135M-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1790759353,"model":"SmolLM2-135M-Instruct","choices":[{"index":0,"delta":{"content":"Bon"},"finish_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1790759353,"model":"SmolLM2-135M-Instruct","choices":[{"index":0,"delta":{"content":"jour"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1790759353,"model":"SmolLM2-135M-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}

data: [DONE]
```

The sequence is fixed by convention: a first chunk with the role, one chunk per piece of text (`delta` is what to append), a chunk with the finish reason, optionally one with `usage`, and the literal `[DONE]`. Clients rely on every part of it; leaving out `[DONE]` makes some of them wait forever.

Part 2 of the demo timestamps each event as the client receives it (SmolLM2-135M, 4 threads, the reference machine of chapter 17; the JSON is shortened here with `...`):

```text
== 2. the same with "stream": true (server-sent events as they arrive)
   status 200 after 1.1 ms, content-type text/event-stream
      1.2 ms  data: {"id":"chatcmpl-2", ... "delta":{"role":"assistant","content":""},"finish_reason":null}]}
    148.6 ms  data: {"id":"chatcmpl-2", ... "delta":{"content":"The"},"finish_reason":null}]}
    160.6 ms  data: {"id":"chatcmpl-2", ... "delta":{"content":" primary"},"finish_reason":null}]}
    172.6 ms  data: {"id":"chatcmpl-2", ... "delta":{"content":" colors"},"finish_reason":null}]}
   ...
    464.3 ms  data: {"id":"chatcmpl-2", ... "delta":{"content":","},"finish_reason":null}]}
    464.4 ms  data: {"id":"chatcmpl-2", ... "delta":{},"finish_reason":"length"}]}
    464.4 ms  data: [DONE]
   27 events; first text after 148.6 ms
```

The status line and headers arrive after 1.1 ms, before the model has done anything; the first text after the prefill (149 ms); then one event every 12 ms, the decode step time. On the wire, the response uses chunked transfer encoding (it has no `Content-Length`), and each event travels as one chunk; [`src/client.rs`](src/client.rs) decodes both layers in about 60 lines.

If nothing is sent for a while (a request waiting in the queue), proxies and load balancers may close the connection as idle. The server sends an SSE comment line (`:` followed by nothing) every 15 seconds of silence to keep it open.

### 3.4 Text is produced here, not in the engine

The engine produces token ids. Turning them into text happens in the server, for two reasons from earlier chapters:

- A token can end in the middle of a UTF-8 character (chapter 11). `StreamDecoder` holds those bytes until the character is complete, so no event ever contains half a character.
- A stop string can span tokens (chapter 15). `StopMatcher` holds back text that might be the start of a stop string until the next token shows whether it is. When a stop string completes, the server closes the request's channel, which stops the engine at its next token, and ends the answer with `"stop"`.

Keeping this in the server keeps the engine a pure token machine, and it runs on the async runtime's threads, which have spare time, instead of the engine thread, which has none.

### 3.5 Admission control

The engine serves requests one at a time (until chapter 23), so every accepted request adds its whole duration to the wait of the requests behind it. The server admits at most `max_in_flight` requests (running or queued) and answers the rest with `429 Too Many Requests` and `Retry-After: 1`. Part 4 sends ten requests at once with a limit of four:

```text
== 4. ten clients at once, streaming 32 tokens each, 4 allowed in flight
   client  status  response after  first text after  done after
        0     429          5.6 ms                 -      5.6 ms
        1     429          2.2 ms                 -      2.2 ms
        2     200          1.6 ms          149.6 ms    571.0 ms
        3     200          1.9 ms         1776.5 ms   2205.7 ms
        4     200          1.3 ms          687.7 ms   1107.8 ms
        5     200          1.9 ms         1225.7 ms   1675.6 ms
        6     429          1.9 ms                 -      1.9 ms
        7     429          1.8 ms                 -      1.8 ms
        8     429          1.9 ms                 -      1.9 ms
        9     429          1.9 ms                 -      1.9 ms
```

Which four got in depends on which connections reached the server first. The four admitted requests run one after another, about 0.5 s each, so their first text arrives at 0.15, 0.69, 1.23 and 1.78 s. The six refused learn it within 6 ms and can retry, go to another server, or tell their user. Without the limit, the tenth would have waited about 4.6 s for its first token; with a hundred clients, fifty seconds, far past the point where clients time out and retry, adding the same work again. A queue without a limit does not serve more requests; it turns overload into timeouts for everyone.

How to choose the limit: `max_in_flight × time per request` is the longest wait an admitted request can have. With 0.5 s per request and a target of at most 2 s before the first token, 4 is about right, which is why the demo uses it. Chapter 25 revisits this with batching, where time per request depends on how many run together.

### 3.6 What HTTP costs

Part 3 asks for `/health` 200 times, each on a new TCP connection:

```text
== 3. GET /health, 200 times in a row (new connection each time)
   median 0.2 ms, 99th percentile 1.2 ms
```

Against 150 ms for the first token and 12 ms per token after it, the HTTP layer is negligible. That is typical: in an inference server the model is so expensive that the server around it rarely needs to be clever about speed. It needs to be correct about load, cancellation and failure.

### 3.7 Clients that leave

A client that disconnects should stop costing anything. Here that happens by ownership, with no code written for it:

- For a streamed answer, the response body owns the SSE stream, which owns the request's `TextStream`, which owns the engine's event receiver and the admission permit. When the connection closes, hyper (the HTTP library under axum) drops the body; the receiver is dropped, so the engine stops at its next token (chapter 21), and the permit is returned, so the place is free.
- For a whole answer, the handler is still running when the client leaves. hyper notices the closed connection and drops the handler's future, with the same effect. A test measures it: the place is free within a few milliseconds.

### 3.8 Graceful shutdown

Deployments restart servers all the time: new versions, moved machines, scaling down. The process manager sends `SIGTERM`, waits a grace period (30 s by default in Kubernetes), then kills the process. A server that exits at once on `SIGTERM` cuts off every answer in progress.

`axum::serve(...).with_graceful_shutdown(signal)` stops accepting connections when `signal` completes and waits for the open ones to finish. Part 5 starts a streamed request, then shuts down:

```text
== 5. shutdown while a request is streaming
   the request kept streaming: 50 more events, the last "[DONE]", 766.8 ms after shutdown began
   server stopped 778.3 ms after shutdown began
   new connections refused: true
```

The limit of this approach is the grace period: a request longer than it is still cut off. Keep `max_tokens` bounded (here by the context length), or add a deadline after which the server stops waiting (exercise 3).

## 4. The code

The JSON types are in [`src/api.rs`](src/api.rs), the server in [`src/lib.rs`](src/lib.rs), the client in [`src/client.rs`](src/client.rs), the demo in [`src/main.rs`](src/main.rs), the tests in [`tests/api.rs`](tests/api.rs).

### 4.1 The request type

<!-- file: src/api.rs -->
```rust
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatRequest {
    /// Accepted and echoed back; this server has one model.
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// Newer clients send `max_completion_tokens`, older ones `max_tokens`.
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    // ...
    #[serde(default)]
    pub stop: Option<Stop>,
```

The Rust struct is the wire format. `#[serde(default)]` makes a field optional in the JSON; `alias` accepts a second name; `Stop` is an `#[serde(untagged)]` enum, because OpenAI allows `"stop": "\n"` as well as `"stop": ["\n", "User:"]`, and `untagged` tries each variant's shape in turn. A missing `messages` or a `temperature` of `"hot"` is rejected by `serde` with a message naming the field, which the server passes on:

```text
{"error":{"message":"Failed to deserialize the JSON body into the target type: temperature: invalid type: string \"hot\", expected f32 at line 1 column 64","type":"invalid_request_error"}}
```

### 4.2 The handler

<!-- file: src/lib.rs -->
```rust
async fn chat(
    State(state): State<AppState>,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    // Malformed JSON, or JSON that does not fit `ChatRequest`: axum's
    // message says which field is wrong.
    let Json(req) = body.map_err(|e| ApiError::invalid(e.body_text()))?;
    let shared = &state.0;
    let prepared = prepare(shared, req)?;
    // Admission control: a bounded number of requests in flight. Queueing
    // without a limit would only turn overload into timeouts for everyone.
    let permit = Arc::clone(&shared.admission)
        .try_acquire_owned()
        .map_err(|_| ApiError::busy())?;
    let events = shared
        .engine
        .submit(prepared.request)
        .map_err(|_| ApiError::unavailable("the engine has stopped"))?;
```

axum calls handlers with arguments it extracts from the request: `State` is the shared state given to the router, `Json<ChatRequest>` parses the body. Taking `Result<Json<...>, JsonRejection>` instead of `Json<...>` lets the handler turn a parse failure into an OpenAI-style error rather than axum's plain-text default. Every step returns `Result<_, ApiError>`, and `ApiError` implements `IntoResponse`, so `?` turns any failure into the right status and JSON body.

Admission is a `tokio::sync::Semaphore` with `max_in_flight` permits. `try_acquire_owned` takes one if available and never waits: waiting is what admission control is meant to avoid. The permit is an RAII guard: the place is returned when it is dropped.

### 4.3 Who holds the permit

The permit goes into the `TextStream`, not a local variable of the handler:

<!-- file: src/lib.rs -->
```rust
pub struct TextStream {
    events: UnboundedReceiver<Event>,
    state: AppState,
    decoder: StreamDecoder,
    stops: StopMatcher,
    tokens: usize,
    prompt_tokens: usize,
    queue: VecDeque<Piece>,
    finished: bool,
    _permit: OwnedSemaphorePermit,
}
```

This matters. For a streamed answer, the handler returns as soon as the response *starts*; the stream keeps running afterwards, owned by hyper. A permit held by the handler would be released after 1 ms, and the limit would count nothing. Held by the stream, it is released exactly when the request ends, however it ends: completed, failed, or dropped because the client left. The field name starts with `_` because nothing reads it; it exists only to be dropped at the right time.

### 4.4 From events to text

<!-- file: src/lib.rs -->
```rust
            match self.events.recv().await {
                Some(Event::Token(t)) => {
                    self.tokens += 1;
                    let text = self.decoder.push(self.state.0.tokenizer.token_bytes(t));
                    if self.push_text(&text) {
                        // A stop string: end now, and close the channel so
                        // the engine stops at its next token.
                        self.events.close();
                        self.end("stop");
                    }
                }
```

`next` returns one `Piece` at a time: `Text`, then finally `End` (with the finish reason and token counts) or `Failed`. One token can produce zero pieces (half a character, or text held back by the stop matcher) or two (text, then `End`), so pieces go through a small queue. `close()` on the receiver makes the engine's next send fail, the same signal as a client leaving.

### 4.5 A stream of events

<!-- file: src/lib.rs -->
```rust
    let rest = stream::unfold((text, meta), move |(mut text, meta)| async move {
        let events = match text.next().await? {
            Piece::Text(s) => vec![json(&meta.chunk(
                Delta {
                    content: Some(s),
                    ..Delta::default()
                },
                None,
            ))],
```

axum's `Sse` takes any `Stream` of events. `stream::unfold` builds one from a state and an async function that returns the next item and the new state, or `None` to end. Here the state is the `TextStream` and the response metadata; each step turns one piece into its events (`End` becomes up to three: the finish chunk, the usage chunk, `[DONE]`), and `flat_map` flattens those lists into single events. The first event (the role) is prepended with `stream::once(...).chain(rest)`.

### 4.6 Shutting down

<!-- file: src/lib.rs -->
```rust
pub async fn serve(
    listener: TcpListener,
    state: AppState,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}
```

`main` passes `shutdown_signal()`, which completes on Ctrl-C or `SIGTERM`; the tests and the demo pass a `oneshot` channel instead, so they can trigger it themselves. When `serve` returns, the router and its state are dropped, the last `EngineHandle` with them, and the engine thread finishes (chapter 21).

## 5. Run it

```bash
cargo test -p ch22-http-server                            # 8 tests over real sockets, about a second
cargo run --release -p ch22-http-server -- demo           # parts 1-5 above, about 10 seconds
cargo run --release -p ch22-http-server -- serve --port 8080 --max-in-flight 8
```

With the server running, from another terminal:

```bash
curl -s localhost:8080/health
curl -sN localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Tell me a joke."}],"stream":true}'
```

(`-N` turns off curl's output buffering, so the events appear as they arrive.) Any OpenAI client works too, pointed at `http://localhost:8080/v1` with any API key.

The tests start the real server on a free port with a tiny random model and a byte-level tokenizer built in the test, and check: a whole answer equals direct generation; a streamed answer has the same text and the event sequence above; a stop string cuts the answer; bad requests get 400s; the capacity limit returns 429 and frees the place when a client leaves (streamed or not); graceful shutdown lets a running stream finish and then refuses connections.

## 6. The Rust behind it

**Extractors are types.** An axum handler's arguments say what it needs from the request: `State<AppState>`, `Json<ChatRequest>`, headers, the path. axum implements a trait (`FromRequest`) for each of them and calls the handler only if all of them succeed, or passes the failure if the argument is a `Result`.

**Errors as values that become responses.** `ApiError` implements `IntoResponse`. A handler returning `Result<Response, ApiError>` can use `?` at every step, and each error carries its own status code. No exceptions, no global error handler: the type says what can fail.

**RAII for capacity.** `OwnedSemaphorePermit` returns its place to the semaphore in `Drop`. Moving it into the object whose lifetime equals the request's makes the count correct on every path, including ones nobody thought of, such as hyper dropping a response body halfway.

**`impl Stream` in return position.** `stream_response` returns `Sse<impl Stream<Item = ...>>`. The concrete type (an `unfold` over an async closure, flattened, chained after a `once`) has no name that could be written down; `impl Trait` lets the function return it anyway, with no boxing.

**`Send + 'static` futures.** `tokio::spawn` and axum both require futures that can move between threads and borrow nothing temporary. That is why the state is an `Arc` cloned into each handler, and why `TextStream` owns an `AppState` instead of borrowing the tokenizer.

## 7. Mistakes you will make

- **Holding the permit in the handler.** For streamed responses the handler returns immediately; the limit would count nothing (section 4.3).
- **An unbounded queue.** Everything works in testing, then at the first real traffic spike every request times out.
- **Blocking in a handler.** Chapter 21's lesson again: tokenizing a prompt is fine (microseconds), running the model is not.
- **Forgetting `[DONE]`,** or sending the role chunk late: OpenAI clients hang or show nothing.
- **Sending half a character.** A client that decodes each event as UTF-8 fails on a split multi-byte character.
- **Returning 500 for the client's mistakes.** A too-long prompt is a 400 with `context_length_exceeded`; a 500 makes clients retry the same bad request.
- **Logging prompts and answers by default.** They are user data; log sizes and timings instead.

## 8. How the professionals do it

- **vLLM** serves the OpenAI API from Python (FastAPI) in front of its engine, supports far more of the API (tools, logprobs, `n > 1`, the completions and embeddings endpoints), and queues rather than refuses by default, relying on its high batched throughput.
- **Text Generation Inference** (Hugging Face) is written in Rust with axum, like this chapter, and refuses requests beyond `--max-concurrent-requests` with 429 ("Model is overloaded"). It also validates prompt lengths in the router before a request reaches a model shard.
- **llama.cpp's server** implements the OpenAI endpoints in C++, with a fixed number of slots (`--parallel`) and a queue.
- In production, a **gateway** usually sits in front: API keys, per-user rate limits (OpenAI's `x-ratelimit-*` headers), routing between servers and models, retries. The inference server itself stays simple.

## 9. Exercises

1. Add API keys: read `Authorization: Bearer <key>`, compare with a configured list, and answer `401` with an OpenAI-style error otherwise. Where in the pipeline of section 3.2 does the check go, and why there?
2. Add `/v1/completions` (a raw `prompt` string instead of `messages`, no chat template), sharing everything after `prepare`.
3. Make shutdown bounded: after the signal, wait at most 10 s for requests in progress, then exit anyway. (`tokio::select!` between the `serve` future and a timer that starts when the signal arrives.)
4. Add a `/metrics` endpoint with request counts by status, the number in flight, and a histogram of time to first token. What would you alert on?

## 10. Check yourself

1. Why is a 429 better for clients than waiting in an unbounded queue?
2. Where is the admission permit stored, and what goes wrong if it is stored anywhere shorter-lived?
3. What does an SSE response look like on the wire, and why does it use chunked transfer encoding?
4. How does the engine learn that a client disconnected, for a streamed request and for a whole one?
5. Why are stop strings handled in the server and not in the engine?
6. What happens to a request in progress when the server receives `SIGTERM`?

## 11. Recap

- Speak OpenAI's chat completions API, and every client that exists already works.
- A request passes parse → validate → admit → submit → relay; everything cheap is checked before a request takes a place.
- Streaming: headers after 1 ms, first text after the prefill (149 ms), then one event per decode step, ending with `[DONE]`.
- Admission control with a semaphore: ten clients against four places, six refused within 6 ms instead of waiting up to 4.6 s. The permit lives exactly as long as the request.
- HTTP costs 0.2 ms per request; the model costs hundreds.
- Cancellation and capacity are both handled by ownership and `Drop`.
- Graceful shutdown finishes the requests in progress (767 ms in the demo) and then refuses connections.

## Answers

**Exercises**

1. After parsing the headers and before reading or validating the body: an unauthenticated client should learn nothing about the server's limits or the body's validity, and should not cost the work of parsing a large body. In axum, a middleware layer (or an extractor that fails) does this for every route at once.
2. The body is `{"prompt": "...", ...}` and the response has `choices[].text` instead of `message`, and `object: "text_completion"`. `prepare` differs only in how the prompt is built (`tokenizer.encode(prompt)` without `chat_prompt`); `TextStream`, admission and streaming are shared.
3. Split the shutdown in two: `serve(...).with_graceful_shutdown(signal)` stays as it is, and `main` runs `tokio::select!` on the serve future and on `async { signal_received.await; sleep(10 s).await }`. When the timer wins, returning from `main` ends the process; the engine thread is stopped with it.
4. Counters by status code (a rising share of 429s says the server needs more capacity), requests in flight (always at the limit: overloaded), TTFT and time per token as histograms (their 99th percentiles are the latencies users feel). Alert on the error rate and on TTFT's 99th percentile against the service's target. Chapter 30 builds this.

**Check yourself**

1. A refused client learns it in milliseconds and can retry later, go elsewhere, or tell its user. A client in a long queue waits, often times out anyway, and then retries, so the server does the work of both attempts.
2. In the `TextStream`, which lives exactly as long as the request (inside the SSE stream for streamed answers). Stored in a local of the handler, it would be released when the handler returns, which for a streamed answer is when the response starts, so the limit would not count running requests.
3. `data: <json>` lines, each event ended by a blank line, with content type `text/event-stream`. The server does not know the total length in advance, so it cannot send `Content-Length`; chunked encoding sends each event as a piece with its own size.
4. Streamed: hyper drops the response body when the connection closes, dropping the `TextStream` and its receiver; the engine's next send fails. Whole: hyper drops the handler's future when it sees the connection closed, with the same result. Either way, the engine stops within one decode step.
5. Stop strings are text, and turning tokens into text (including holding back partial characters and partial stop strings) needs the tokenizer and some buffering. Doing it in the server keeps the engine thread, the one scarce resource, doing only model work.
6. The server stops accepting new connections and waits for open ones to finish; the streamed answer runs to `[DONE]`. Then `serve` returns, the state and the last engine handle are dropped, and the engine thread finishes. If the process manager's grace period ends first, the process is killed.

## Further reading

- OpenAI API reference: "Chat Completions" and "Streaming", and the error codes page.
- The WHATWG HTML standard, section "Server-sent events" (the `text/event-stream` format).
- The axum documentation: extractors, `IntoResponse`, `Sse`, `serve::WithGracefulShutdown`.
- Marc Brooker, "Timeouts, retries, and backoff with jitter" (AWS Builders' Library), on why unbounded queues and retries make overload worse.
- Next: [Chapter 23: Continuous batching](../23-continuous-batching/README.md). Many requests in one forward pass.
