# Chapter 21: The engine thread

> **In one sentence:** the model, its thread pool, its KV cache and its scratch buffers belong to one engine thread, and everything else talks to it through channels: requests go in, each request gets its own stream of tokens back, and dropping that stream cancels the request.

**Where this fits:** until now one program asked one question and waited. A server has many clients arriving at any time, and most of its code is async (chapter 22's HTTP server). This chapter builds the boundary between the two worlds, measures what one-request-at-a-time costs the clients who wait, and sets up chapter 23 (batching) to fix that.

**You need:** chapter 7 (the spinning pool), chapter 14 (the engine: `Model`, `KvCache`, `Scratch`), chapter 15 (`generate`, stop tokens, `ControlFlow`), chapter 20 (the fastest model so far). Basic familiarity with `async`/`.await` helps; the chapter explains what it uses.

**You will build:** an engine thread with a job queue, per-request token streams usable from both synchronous and async code, cancellation by dropping the receiver, shutdown by dropping the handles, and measurements of streaming, queueing, cancellation, idle cost and what happens when the model runs on the async runtime.

---

## 1. The intuition

A restaurant kitchen with one chef. Waiters take orders at the tables and pin tickets on a rail; the chef takes the next ticket, cooks, and sends each course out as soon as it is ready instead of the whole meal at the end. If a table leaves, the chef finds out when the next course has nowhere to go, and moves on to the next ticket.

The chef never walks into the dining room, and the waiters never cook. Waiters are cheap and many (async tasks); the chef is one, expensive, and owns the kitchen (the model and its memory). Nobody else touches the stove, so nobody needs to take turns with it.

**Where the analogy breaks:** a real chef cooks several orders at once. This engine cooks one at a time, and part 2 measures what that costs the people waiting. Chapter 23 teaches the chef to batch.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Engine thread** | The one thread that owns the model and runs every forward pass. |
| **Channel** | A queue between threads: senders push messages, a receiver takes them in order. `mpsc`: multiple producers, single consumer. |
| **Streaming** | Sending each token to the client as soon as it is produced. |
| **TTFT** | Time to first token: from submitting a request to receiving its first token. What a user perceives as "responsiveness". |
| **Queueing delay** | Time a request waits before the engine starts it. |
| **Head-of-line blocking** | A request waiting only because another one is ahead of it. |
| **Async runtime** | The scheduler (here tokio) that runs many async tasks on a few threads, switching between them at each `.await`. |
| **Blocking the runtime** | Running long computation inside an async task, so that the thread cannot switch to other tasks until it finishes. |

## 3. The concepts in depth

### 3.1 Why one thread owns the model

The model's forward pass needs `&mut SpinPool`, `&mut KvCache` and `&mut Scratch`. If several request handlers shared them, they would need a `Mutex`, and every handler would wait on the lock while another request ran: the same one-at-a-time behaviour, plus lock contention, plus handlers stuck in a blocking wait.

Putting all of it on one thread makes ownership simple: the engine thread owns everything mutable, handlers own nothing but channels. There is no lock anywhere in the hot path, and the one place requests meet (the job queue) is a channel built for exactly that.

It is also how the professional engines are organized (section 8): vLLM, SGLang, TGI and llama.cpp's server all keep the model in one loop and feed it through queues.

### 3.2 Two kinds of channels

Requests go in over `std::sync::mpsc`. The engine thread is ordinary synchronous code; `queue.recv()` puts it to sleep until a job arrives, using no CPU, and returns an error once every sender is gone, which the engine treats as the signal to shut down.

Events come back over `tokio::sync::mpsc::unbounded_channel`, one channel per request. Its sending side is a plain function that never blocks, which is what the engine needs; its receiving side can be awaited in async code (`recv().await`) or read from a normal thread (`blocking_recv()`). One type serves the HTTP handlers of chapter 22 and the tests.

Why unbounded? A bounded channel would make the engine wait when a client reads slowly, and one slow client would stall everyone. An unbounded one never makes the engine wait, and its size is limited anyway: at most `max_tokens` events per request, a few kilobytes. The place to limit load is at admission (how many requests are accepted), which chapter 22 handles.

### 3.3 Streaming and time to first token

Part 1 sends one request and prints tokens as they arrive (SmolLM2-135M, `bf16`, chapter 20's attention, 4 threads on the reference machine of chapter 17):

```text
== 1. one request, streamed
   The sky is blue because of a process called Rayleigh scattering. This is the scattering of light by tiny particles in the atmosphere, specifically nitrogen and oxygen molecules. When sunlight enters the Earth's atmosphere, it encounters these particles.

The light is initially scattered in all directions, but the scattered light is not completely absorbed
   36 prompt tokens, 64 generated; first token after 89 ms, then 81.2 tokens/s (Length)
```

The first token needs the whole prompt processed (prefill); every later one needs one decode step. Without streaming, the user would see nothing for 0.9 s; with it, text starts after 89 ms and then appears faster than anyone reads.

### 3.4 What waiting costs

Part 2 submits four requests at the same moment:

```text
== 2. four clients submit at the same moment, 32 tokens each
   client     queued  first token      total  tokens
        0     0.2 ms       107 ms     199 ms       7
        1     199 ms       292 ms     727 ms      32
        2     727 ms       875 ms    1251 ms      32
        3    1251 ms      1330 ms    1696 ms      32
```

(Client 0's answer ended after 7 tokens at the end-of-turn token; the others used all 32.) Client 3 waits 1.25 s before its request even starts, and sees its first token 12 times later than client 0. With 20 clients, the last would wait many seconds. That is head-of-line blocking, and it is why engines batch: a decode step for 4 requests reads the weights once, like a step for 1 (chapter 5), so in principle the four could be served in not much more than the time of one. Chapter 23 builds that and measures how close it gets.

### 3.5 Cancellation

Clients leave: a browser tab closes, an HTTP client times out, a user presses stop. An engine that keeps generating for a client that is gone wastes compute that queued clients are waiting for. Part 3 measures it:

```text
== 3. client A asks for 200 tokens, client B is queued behind it
   A reads all 200 tokens:   B waits  2692 ms to start
   A leaves after 10 tokens: B waits   226 ms to start, 11 ms after A left
```

Cancellation needs no extra message. The client drops its receiver; the engine's next `send` fails; the callback returns `ControlFlow::Break`, `generate` stops, and the engine takes the next job. The delay is at most one decode step (11 ms here). In Rust, dropping a value is a reliable signal: it happens on every path, including a panic or an early `return` in the client's code.

One gap remains: the engine only sends after a token is produced, so a client that leaves during a long prefill is noticed only when prefill ends. Exercise 3 closes it.

### 3.6 What waiting for work costs

Chapter 7's pool spins while it waits, because waking a sleeping thread takes tens of microseconds and a decode step has many short parallel sections. Between requests, though, there may be nothing to do for seconds or hours. Part 4:

```text
== 4. CPU time used during 1 s with no requests
   the engine, waiting for requests : 0.00 s of CPU per second
   plus an idle SpinPool (4 threads): 2.95 s of CPU per second
   starting and stopping a 4-thread pool: 220 µs
```

An idle 4-thread pool burns three cores (its three workers; the fourth thread is the caller, asleep here). On a shared machine, that slows everything else down. In a first version of this demo, the engine kept its pool for its whole life, and part 5 below, which runs a second model on the same machine, took 9 s instead of about 0.5 s: two pools of spinning threads fighting over four cores.

The fix is in the engine loop: create the pool when a job arrives, keep it while jobs are queued, drop it when the queue is empty. Creating it costs about 0.2 ms, against a request that takes at least tens of milliseconds.

### 3.7 Never run the model on the async runtime

An async runtime runs many tasks on a few threads. A task gives the thread back only at an `.await`. A forward pass has no `.await` in it: while it runs, nothing else scheduled on that thread runs. No timers, no other clients, no accepting connections.

Part 5 runs a task that sleeps 10 ms in a loop and records the longest gap between wake-ups, on a single-threaded runtime:

```text
== 5. a 10 ms timer on the async runtime while 32 tokens are generated
   on the engine thread:  generation  566 ms, longest gap between ticks   13 ms
   inside an async task:  generation  543 ms, longest gap between ticks  548 ms
```

With the engine on its own thread, the runtime only waits for events and the timer stays on time. With the model running inside an async task, the runtime is frozen for the whole generation. A multi-threaded runtime hides the problem longer, until as many requests run as it has threads.

## 4. The code

Everything is in [`src/lib.rs`](src/lib.rs); the demo is [`src/main.rs`](src/main.rs).

### 4.1 Submitting

<!-- file: src/lib.rs -->
```rust
    pub fn submit(
        &self,
        request: Request,
    ) -> Result<async_mpsc::UnboundedReceiver<Event>, EngineGone> {
        let (events, receiver) = async_mpsc::unbounded_channel();
        let job = Job {
            request,
            events,
            submitted: Instant::now(),
        };
        self.jobs.send(job).map_err(|_| EngineGone)?;
        Ok(receiver)
    }
```

A request creates its own event channel. The sending end travels to the engine inside the `Job`, with the time of submission (for the queueing and TTFT measurements); the receiving end goes back to the caller. `EngineHandle` holds only an `mpsc::Sender<Job>`, which is `Clone`, so every HTTP handler or client thread can have its own handle. If the engine thread has stopped, `send` fails and the caller gets `EngineGone` instead of waiting forever.

### 4.2 The engine loop

<!-- file: src/lib.rs -->
```rust
        .spawn(move || {
            // Everything the engine owns is created here, on its thread.
            let mut cache = KvCache::new(&model.config, context);
            let mut scratch = Scratch::new(&model.config, 256, context);
            // `recv` sleeps without using the CPU until a job arrives, and
            // fails once every handle is gone: that is the shutdown.
            while let Ok(job) = queue.recv() {
                // Pool workers spin while they wait for work: right while
                // requests run, a waste of every core while none do. So the
                // pool lives only while there is work (starting its threads
                // takes well under a millisecond).
                let mut pool = SpinPool::new(threads);
                run(&model, &mut pool, &mut cache, &mut scratch, job);
                while let Ok(job) = queue.try_recv() {
                    run(&model, &mut pool, &mut cache, &mut scratch, job);
                }
            }
        })
```

`move` transfers the model into the thread: from here on, no other code can reach it. The cache is allocated once for the engine's life and reused by every request (`generate` clears it at the start). The outer loop sleeps in `recv`; the inner loop drains whatever else is queued before the pool is dropped.

Shutdown needs no special message: when the last `EngineHandle` is dropped, the channel's sender count reaches zero, `recv` returns `Err` once the queue is empty, the loop ends, and the thread returns. Requests already queued are still served.

### 4.3 Running a request

<!-- file: src/lib.rs -->
```rust
        |token| {
            first_token.get_or_insert_with(|| submitted.elapsed());
            completion_tokens += 1;
            // A failed send means the receiver was dropped: the client has
            // gone, so stop spending compute on it.
            if events.send(Event::Token(token)).is_err() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
```

This is chapter 15's `generate` with a callback per token. The callback records the time of the first token, counts, and sends. `get_or_insert_with` fills the `Option` only the first time. After `generate` returns, `run` sends one final `Event::Done` with a `Summary` (finish reason, token counts, queueing delay, TTFT, total). A prompt that does not fit the context gets `Event::Rejected` instead of a panic on the engine thread, which would take the engine down for everyone.

### 4.4 Measuring the runtime's responsiveness

<!-- file: src/main.rs -->
```rust
async fn ticker(stop: Arc<AtomicBool>) -> Duration {
    let mut last = Instant::now();
    let mut longest = Duration::ZERO;
    while !stop.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(10)).await;
        longest = longest.max(last.elapsed());
        last = Instant::now();
    }
    longest
}
```

The ticker asks to be woken every 10 ms. If the runtime's thread is free, it is woken on time, and the longest gap is a little over 10 ms. If something holds the thread, the gap is however long that took.

## 5. Run it

```bash
cargo test -p ch21-engine-thread
cargo run --release -p ch21-engine-thread                 # all parts, about 15 seconds
cargo run --release -p ch21-engine-thread -- cancel       # or: stream, queue, idle, runtime
```

The tests use a small random model: tokens streamed through the engine equal those of direct generation, requests from several threads are all served, dropping a receiver cancels, an oversized prompt is rejected, dropping the handles stops the thread. Part 4 reads `/proc/self/stat` and prints "not measured" on systems without it.

## 6. The Rust behind it

**`Send` and `'static` for `thread::spawn`.** The closure given to `spawn` must be `Send` (it moves to another thread) and `'static` (the thread may outlive the caller's stack frame). Hence `W: Matrix + 'static`: `Matrix` already requires `Send + Sync` (chapter 14), and `'static` rules out a matrix type that borrows data it does not own. A model built on chapter 16's `mmap`ed weights qualifies: each matrix holds an `Arc` of the mapped file, so the mapping lives as long as any matrix does.

**Drop as a protocol.** Two of the three control signals are drops: dropping every `EngineHandle` stops the engine, dropping an event receiver cancels its request. Nothing has to remember to send "stop" or "cancel", and the signal is sent even if the client's code panics or returns early.

**`Result` from a channel send.** `send` returns `Err` when the other side is gone. `Result` is `#[must_use]`, so the compiler warns if it is silently ignored; ignoring it has to be written out: `let _ = events.send(...)` where it genuinely does not matter (the final `Done` to a client that left), a `Break` where it does.

**A panicking engine.** If the engine thread panics, its channel ends are dropped during unwinding. Clients waiting on events get `None` (and `collect` reports that the engine stopped); new submissions get `EngineGone`. `JoinHandle::join` returns the panic to whoever joins the thread.

**Two worlds, one type.** `tokio::sync::mpsc::UnboundedReceiver` has both `recv().await` and `blocking_recv()`. The latter panics if called from inside an async runtime, which prevents exactly the mistake of section 3.7 on the receiving side.

## 7. Mistakes you will make

- **Running the model inside an async handler.** Section 3.7. If you must run blocking work from async code, `tokio::task::spawn_blocking` moves it to a separate thread pool; for an inference engine, a dedicated thread is simpler and keeps the model's state in one place.
- **Wrapping the model in `Arc<Mutex<...>>`.** It works, and it serializes requests just like this engine does, but handlers block runtime threads while waiting for the lock and there is no single place to later add batching.
- **A bounded event channel with a blocking send.** One slow reader stalls the engine and every other request.
- **Ignoring a failed send.** The engine keeps generating tokens for a client who left. Part 3 shows what that costs the next client: 2.7 s instead of 0.2 s.
- **Spinning while idle.** An engine with a spinning pool uses three to four cores while serving no one (part 4). Measure CPU use at idle, not only speed under load.
- **Two spinning pools on one machine.** Each assumes it has the cores to itself. Together they can be many times slower than either alone (the 9 s of section 3.6).

## 8. How the professionals do it

- **vLLM** runs its engine core in a separate process with a busy loop over a scheduler; the API server (async Python) talks to it through ZeroMQ sockets, a cross-process version of this chapter's channels.
- **Text Generation Inference (TGI)** has a router written in Rust (tokio, axum) that queues and batches requests, and model shards that run the forward passes, connected by gRPC.
- **llama.cpp's server** keeps a task queue and one main loop that owns the model and processes "slots" (one per concurrent request). Its `--poll` option sets how long worker threads spin before sleeping, the same trade-off as section 3.6.
- **SGLang** separates the tokenizer, the scheduler with the model, and the detokenizer into processes connected by queues.

All of them cancel on client disconnect, and all of them batch (chapter 23).

## 9. Exercises

1. Add a limit on queued requests: `submit` returns an error when more than `N` jobs are waiting. (You need a shared counter: incremented in `submit`, decremented by the engine when it takes a job.) What should an HTTP server answer when it is full?
2. Add a deadline to `Request`: the engine stops the request when it passes, with a new finish reason. Where do you check it?
3. Make cancellation work during prefill: split a long prompt's prefill into chunks (chapter 14) and check between chunks whether the receiver is still there. `UnboundedSender::is_closed` tells you without sending anything.
4. Run part 2 with 8 clients and plot TTFT against position in the queue. What is the average TTFT, as a formula in the time of one request?

## 10. Check yourself

1. Why does the model live on one thread instead of behind a `Mutex`?
2. Which channel carries requests and which carries tokens, and why is each the kind it is?
3. How does the engine learn that a client has gone, and how quickly?
4. What does dropping the last `EngineHandle` do, step by step?
5. Why did an idle engine use three cores before the fix, and why is creating the pool per busy period cheap enough?
6. A forward pass takes 15 ms. On a single-threaded tokio runtime, what happens to a 1 ms timer while it runs inside an async task?

## 11. Recap

- One engine thread owns the model, the pool, the cache and the scratch space; everything else holds channels.
- Requests in over `std::sync::mpsc`, events out over one unbounded tokio channel per request: the engine never waits for a client.
- Streaming: first token after 89 ms, then about 80 tokens/s for SmolLM2-135M on 4 cores.
- One request at a time makes the fourth of four simultaneous clients wait 1.25 s to start. Chapter 23 fixes this with batching.
- Cancellation by drop: the engine notices within one decode step (11 ms); not cancelling cost the next client 2.7 s.
- A spinning pool burns 2.95 CPU-seconds per second at idle; the engine creates its pool only while it has work (0.2 ms to start).
- Never run the model on the async runtime: a 10 ms timer stalled for 548 ms.

## Answers

**Exercises**

1. A shared `Arc<AtomicUsize>` counts waiting jobs; `submit` increments it and refuses (returning a new error, or `Event::Rejected`) above `N`, the engine decrements when it takes a job. An HTTP server answers `429 Too Many Requests` or `503 Service Unavailable` with a `Retry-After` header, so clients back off instead of piling up. Chapter 22 does this.
2. In the token callback, compare `Instant::now()` with the deadline and return `Break`; add a `FinishReason` variant (or report `Stopped` with a flag in `Summary`). Also check before starting the request, since it may expire while queued, and reject it without running.
3. Instead of one `forward_last` over the whole prompt, prefill in chunks of, say, 256 tokens, checking `events.is_closed()` between chunks. The cancellation delay becomes one chunk's prefill time instead of the whole prompt's. Chapter 25 uses the same chunking to interleave prefill with other requests' decode steps.
4. With one request taking time `T` and all arriving together, client `i` (from 0) starts after `i × T` and gets its first token about `i × T + prefill`. The average over `n` clients is `(n − 1) / 2 × T + prefill`: it grows linearly with the number of waiting clients.

**Check yourself**

1. A `Mutex` gives the same one-at-a-time execution, plus handlers that block while waiting for the lock (bad for an async runtime) and no single place to schedule or batch requests. A thread that owns the state needs no lock at all.
2. Requests: `std::sync::mpsc`, because the engine is a synchronous thread that should sleep in `recv` until work arrives. Tokens: a tokio unbounded channel per request, because its send never blocks the engine and its receiver works in async handlers.
3. Its next `send` of a token fails because the receiver was dropped. At most one decode step later (11 ms measured), unless the request is still in prefill.
4. The job channel's last sender is dropped. The engine finishes the job it is running and any still queued, then `recv` returns `Err`, the loop ends, the pool is dropped (its workers are told to stop and joined), the thread returns, and `JoinHandle::join` returns.
5. The pool's three worker threads spin, then yield in a loop, waiting for the next job; yielding with nothing else to run returns immediately, so each keeps a core busy. Creating a pool starts three threads (about 0.2 ms), negligible next to a request of tens of milliseconds or more.
6. It does not fire until the forward pass ends: the runtime's only thread is busy, so the timer is late by up to 15 ms (and by the whole generation if generation runs in one task without `.await`).

## Further reading

- The Tokio tutorial, especially "Channels" and "Bridging with sync code" (tokio.rs).
- Alice Ryhl, "Async: What is blocking?" (2020), on why long computations stall async runtimes.
- The vLLM V1 architecture notes (the engine core and its process boundary), and llama.cpp's `tools/server/server.cpp`.
- Next: [Chapter 22: An HTTP server](../22-http-server/README.md). OpenAI-compatible endpoints, streaming over server-sent events, and saying no when full.
