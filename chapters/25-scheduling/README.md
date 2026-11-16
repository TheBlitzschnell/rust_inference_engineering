# Chapter 25: Scheduling and SLOs

> **In one sentence:** users judge a server by two latencies, the wait for the first token and the gaps between the tokens after it, as percentiles under real traffic; measuring them takes a load generator whose clients arrive independently of the server, and controlling them comes down to three decisions in the engine: how many prompt tokens a step may carry, which request goes next, and which requests to refuse.

**Where this fits:** chapters 23 and 24 built an engine that runs many requests at once, with a few fixed choices: decode tokens before prompt tokens, prompt chunks of up to `max_step_tokens` per step, admission in order of arrival, and a queue without a limit. This chapter measures what those choices do to the people waiting for answers, and changes them. Part VII then builds on the engine: speculative decoding (chapter 26) makes each request faster, structured output (chapter 27) constrains what it produces.

**You need:** chapter 22 (admission control, TTFT seen from a client), chapter 23 (the step plan, TPOT, throughput against per-client speed), chapter 24 (the paged engine this chapter extends).

**You will build:** latency metrics from per-token client timestamps (TTFT, TPOT, gaps between tokens, percentiles, SLO attainment and goodput); open-loop (Poisson) and closed-loop load generators that run hundreds of clients on one thread; and chapter 24's engine with a scheduling policy (first come first served, or shortest prompt first) and a bounded queue. Then measurements: latency against load, what the step budget does to stalls and first tokens, the two policies on a mix of short and long prompts, and a server at 1.7 times its capacity, with and without a queue limit.

---

## 1. The intuition

A restaurant kitchen at dinner time. Guests judge two things: how long until the first dish arrives, and whether the next ones keep coming without long pauses. The owner counts dinners per hour. Between them, the head chef makes three kinds of decision all evening:

- A party orders a twelve-course banquet. Cooking it all at once would stop every other table's next dish for half an hour; cooking it a few courses at a time, between other dishes, keeps everyone eating (chunked prefill).
- Three orders are waiting: the banquet, and two salads. Making the salads first costs the banquet five minutes and saves the salads half an hour (shortest first).
- The kitchen is already an hour behind and a new group comes in. Seating them does not cook more dinners; it makes every dinner later, including theirs. Better to say so at the door (admission control).

**Where the analogy breaks:** this kitchen gets faster per dish when it cooks many dishes at once (batching, chapter 23), up to a point, so a busy evening is not simply a slow one. And unlike a chef, the engine knows exactly how much work each starter is (the prompt's length), but not how many courses will follow (the answer's length).

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **TTFT** | Time to first token: from sending a request to receiving its first token. Waiting plus prefill. |
| **TPOT** | Time per output token: the average gap between a request's tokens after the first. |
| **Inter-token latency (ITL)** | One gap between two consecutive tokens of a request. Their average for a request is its TPOT; their tail shows stalls. |
| **Percentile** | The p-th percentile is the smallest value that p% of the measurements are at or below. p50 is the median; p99 is the latency one request in a hundred exceeds. |
| **SLO** | Service-level objective: the latencies a service promises itself, such as "first token within 1 s, then at most 100 ms per token". |
| **SLO attainment** | The share of requests that met the SLO. |
| **Goodput** | Requests per second that met the SLO. |
| **Closed loop** | A load generator with a fixed number of users, each sending its next request when the previous one is answered. |
| **Open loop** | A load generator that sends requests at times decided in advance, whatever the server is doing. |
| **Poisson process** | Arrivals at random moments, independent of each other, at a constant average rate; the gaps between them are exponentially distributed. |
| **Utilization** | The arrival rate divided by the rate at which the server can complete requests. |
| **Little's law** | `L = λW`: the average number of requests in a system equals their arrival rate times the average time each spends in it. |
| **Head-of-line blocking** | A long job at the front of a queue delaying every short job behind it. |
| **Chunked prefill** | Processing a prompt over several steps, a bounded number of tokens per step, alongside other requests' decode tokens. |
| **Admission control** | Deciding at arrival whether to accept a request at all. |

## 3. The concepts in depth

### 3.1 What users feel

A streamed answer has two latencies. The **TTFT** is how long the user stares at nothing; it is the time spent waiting for a place plus the prefill of the prompt. The **TPOT** is how fast the text then appears; it is the decode step time, which grows with the batch (chapter 23) and with any prompt work sharing those steps. The whole answer takes TTFT + (n − 1) × TPOT for n tokens.

What matters depends on the use. A chat user notices a TTFT above a second or two, and does not notice a TPOT faster than they read (about 5 words per second, roughly 7 tokens per second, so anything under about 140 ms per token). Code completion in an editor needs a TTFT of a few hundred milliseconds. An overnight job summarizing a million documents cares about neither, only about throughput. Throughput is the operator's number; latency is the user's; the engine trades one for the other, so a service states which latencies it promises: its **SLO**. This chapter uses:

> first token within 1 s, then at most 100 ms per token on average (10 tokens per second).

**Percentiles, not averages.** An average hides the slow requests, and slow requests are what people remember. A page that makes 10 requests sees the 90th-percentile latency of one request in 1 − 0.9¹⁰ = 65% of page loads; at scale, the tail is what somebody is always experiencing. Services therefore promise percentiles: "p90 TTFT under 1 s".

A percentile needs enough samples. By the nearest-rank definition used here (the smallest value with at least p% of the samples at or below it), the 99th percentile of 40 samples is their maximum: 0.99 × 40 = 39.6, rounded up to 40. You need at least 100 samples for a p99 that is not simply the worst one, and many more for one that is stable from run to run. The tables below show p50, p90 and the maximum for first tokens (tens of requests), and p99 for the gaps between tokens (thousands of them).

**Goodput.** Two numbers summarize a run against an SLO: **attainment**, the share of requests that met it, and **goodput**, the number of requests per second that met it (Zhong et al., "DistServe", 2024). Goodput is the number to maximize. Throughput that arrives too late does not count.

### 3.2 Generating load: closed and open loops

A **closed-loop** generator has `N` users; each sends a request, waits for the complete answer, perhaps "thinks" a while, and sends the next. Most quick benchmarks are closed loops ("concurrency 16"). Its defining property: the load adapts to the server. When the server slows down, the users send less, because they are waiting. There are never more than `N` requests in the system, so the queue cannot grow without bound, whatever the server does.

Real users do not wait for each other. Thousands of independent people, each sending a request now and then, produce arrivals that are well described by a **Poisson process**: random moments, independent of one another and of the server, at an average rate λ. The gaps between arrivals are exponentially distributed with mean 1/λ, which makes the traffic bursty: at 4 requests per second on average, 11% of one-second windows bring 7 or more. An **open-loop** generator fixes the arrival times in advance (here, from a Poisson process) and sends each request at its time, whatever the server is doing.

The difference matters most when the server struggles. A closed loop "coordinates" with the server: if the server stalls for two seconds, the users stall with it and the requests they would have sent during the stall are never sent, so the stall delays only the few requests in flight. Real users keep arriving during the stall and all of them wait. A closed-loop benchmark therefore reports better tail latencies than real users would see (Gil Tene calls this *coordinated omission*), and it can never show what happens when demand exceeds capacity.

### 3.3 Measured: latency against load

Part 1 sends chat requests (a passage of 20 to 60 words to summarize, prompts of 50 to 100 tokens, exactly 64 tokens of answer) to the engine (SmolLM2-135M, packed weights, at most 16 requests in the batch, step budget 128, 4 threads, the machine of chapter 17). First from closed loops of 1 and 16 users, then as Poisson arrivals at 1 to 5 requests per second for 12 seconds each:

```text
== 1. chat requests (prompts of 50-100 tokens, 64 tokens out), step budget 128
      SLO: first token within 1 s, then at most 100 ms per token on average
                    req/s tokens/s  TTFT p50     p90     max  TPOT p50    p90 met SLO goodput
   closed, 1 user     1.13      73     89 ms  107 ms  107 ms    12 ms  15 ms    100%   1.13
   closed, 16 users   3.69     236    483 ms 1382 ms 2095 ms    60 ms  65 ms     86%   3.17
   open, 1/s          1.19      76    136 ms  193 ms  197 ms    20 ms  23 ms    100%   1.19
   open, 2/s          1.84     118    123 ms  191 ms  297 ms    20 ms  32 ms    100%   1.84
   open, 3/s          2.48     159    150 ms  260 ms  298 ms    29 ms  40 ms    100%   2.48
   open, 4/s          3.31     212    184 ms  447 ms 1252 ms    41 ms  54 ms     98%   3.24
   open, 5/s          3.39     217    437 ms 3638 ms 4558 ms    66 ms  77 ms     57%   1.92
```

One user alone sees the engine's best: a first token after about 0.1 s (the prefill of 70 tokens) and 12 ms per token. Sixteen users get 3.7 requests per second through it, about its capacity. Little's law checks the measurement: with 16 requests always in the system and 3.69 completing per second, each spends 16 / 3.69 = 4.3 s in it, and indeed 0.5 s to the first token plus 63 × 60 ms is 4.3 s.

With Poisson arrivals, up to 3 per second, the first token barely moves (median 0.12 to 0.15 s): continuous batching absorbs the extra requests, and only TPOT grows, as the batch does. At 4 per second, about 90% of capacity, a tail appears (one first token after 1.25 s), but 98% of requests still meet the SLO. At 5 per second, more than the engine can complete, the 90th-percentile first token is 3.6 s, and it would be longer still in a longer run: the queue grows for as long as the overload lasts. Goodput, which rose with the load until 4 per second, falls from 3.24 to 1.92, although throughput went up.

Compare the closed loop of 16 users with the open loop at 5 per second. Both run the engine at its capacity (3.69 and 3.39 completed per second). The closed loop reports a 90th-percentile first token of 1.4 s, and would report the same however long it ran; the open loop shows 3.6 s and growing. Only the open loop tells you what overload looks like. (Across three runs: closed-loop p90 0.8-1.4 s; open loop at 5 per second, 3.0-3.6 s.)

### 3.4 Why the knee: a little queueing theory

**Utilization** ρ is the arrival rate divided by the rate the server can complete requests. For the simplest model of a server with random arrivals and random service times (the M/M/1 queue), the average wait before service is ρ / (1 − ρ) service times:

```text
utilization   50%   70%   80%   90%   95%   99%
wait           1    2.3    4     9    19    99    (in service times)
```

Nothing is wrong at 80% utilization, yet requests already wait four service times on average. The cause is randomness: with arrivals perfectly spaced and every request the same size, nobody would wait until 100%. With bursts, requests arrive while others are still being served, and the queue they leave takes time to drain. Above 100% there is no steady state at all: the queue grows by λ − μ requests per second (μ being the completion rate) for as long as the overload lasts.

A batching engine is not a server with a fixed speed: more requests in a step mean more throughput (chapter 23), so it absorbs bursts by batching them, which is why part 1's first tokens stay flat up to 3 per second. But the batch has limits (here 16 requests, elsewhere KV memory, chapter 24), and once they are reached the queue behaves like any other. The practical rule follows: plan capacity so that the expected peak is 70-80% of the throughput you measured, not 100% (chapter 30 does this).

### 3.5 The step budget: chunked prefill

Every step carries the decode token of every running request, then prompt tokens up to the budget `max_step_tokens` (chapter 23's plan). Beyond about 16 rows a step is compute-bound (chapter 23), so its duration grows with its tokens: about 2.3 ms per token on this machine, so 0.3 s with 128 prompt tokens and 2.5 s for a whole 1,050-token prompt. Every request that is decoding waits for the entire step. A long step is a stall in every stream.

Part 2 makes this visible. Eight requests stream answers of 200 tokens; three requests with 1,050-token prompts arrive at 1, 3 and 5 s. The budget varies:

```text
== 2. eight streams of 200 tokens; three 1,050-token prompts arrive at 1, 3 and 5 s
   step budget   gaps between the streams' tokens   long prompts' TTFT   all done
                    p50      p99      max             p50      max
       32 tokens     69 ms   185 ms   190 ms         5258 ms  6842 ms     13.6 s
       64 tokens     26 ms   200 ms   200 ms         3455 ms  3952 ms     12.3 s
      128 tokens     23 ms   329 ms   354 ms         2842 ms  3137 ms     11.9 s
      256 tokens     25 ms   661 ms   682 ms         2547 ms  2754 ms     12.3 s
     2048 tokens     23 ms  2571 ms  2638 ms         3218 ms  3692 ms     13.5 s
```

With a budget of 2048, each long prompt is prefilled in one step, and all eight streams freeze for 2.6 s, three times. The users see text appear, stop for nearly three seconds, and continue. Halving the budget roughly halves the longest stall: 0.68 s at 256, 0.35 s at 128, 0.20 s at 64. The median gap (23-26 ms, the decode step) is unaffected down to 64.

What does it cost the long prompts? Nothing measurable: across three runs, their first token came after 2.4-3.2 s with 128 or 256 tokens per step, and after 3.1-3.5 s with whole prompts in one step. Below 64 the cost appears: at 32 tokens per step, 8 of them go to the streams' decode tokens and only 24 to prompts, so the prompts take 5.3 s; and every step now carries prompt work, so even the median gap grows to 69 ms.

Why is chunking so cheap here? Because a step of 128 tokens is already compute-bound: the weights are read once per step whatever its size, and beyond the ridge point (about 13 rows on this machine, chapter 23) more rows do not make each row cheaper. A 128-token chunk costs per token what a 1,050-token prompt costs. On a GPU the ridge point is near 300 rows, so chunks there are 512 to 8,192 tokens; the principle is the same.

The rule: choose the largest budget whose stall your TPOT target can tolerate. Stall ≈ budget × cost per token, so with 2.3 ms per token, a budget of 128 bounds stalls near 0.3-0.4 s. The rest of this chapter uses 128.

Putting decode tokens first, as this engine does, protects the streams at the expense of new prompts: under heavy decode load, prompts get only what is left of the budget. Older engines did the opposite (vLLM before 2024 ran a prompt's whole prefill as soon as it arrived and paused every decode for it, the 2048 row above); Sarathi-Serve (Agrawal et al., 2024) introduced the decode-first, chunked plan, which it called *stall-free* batching.

### 3.6 Who goes next: head-of-line blocking

With first come, first served (FCFS), a long prompt that arrives just before a short one takes the step's prompt budget until it is done; the short one, which needs a single step, waits behind it for seconds. That is head-of-line blocking.

The classical cure is to serve the shortest job first. For a single server, **shortest remaining processing time** first minimizes the average waiting time over all orderings (Schrage, 1968). An inference engine knows the length of every prompt exactly (it does not know the length of the answers), so `Policy::ShortestFirst` orders by prompt length: when admitting waiting requests, and when handing out the step's prompt budget among admitted ones, the request with the fewest prompt tokens left goes first.

Part 3 sends 40 requests at 1.5 per second, Poisson; 4 of them have 1,050-token prompts, the rest are chat requests as in part 1:

```text
== 3. 40 requests at 1.5 per second; 4 of them have 1,050-token prompts
                       short prompts' TTFT         long prompts' TTFT   met SLO
                          p50      p90      max         p50      max
   FCFS, budget 512       151 ms  1591 ms  2218 ms     2405 ms  2709 ms       80%
   FCFS, budget 128       179 ms  1629 ms  2253 ms     2178 ms  2596 ms       80%
   shortest first, 128    168 ms   414 ms   600 ms     2382 ms  2782 ms       90%
```

With FCFS, the median short request is fine, but one in ten (those that arrive while a long prompt is being prefilled) waits 1.6 s or more. A smaller budget alone does not help: the long prompt, admitted first, still takes all of every step's prompt budget until it is done. Shortest first brings the short requests' 90th percentile from 1.6 s to 0.41 s, and costs the long prompts almost nothing: each short request they let through is one 128-token step, about 0.3 s, and their own prefill takes more than 2 s anyway.

The risk is **starvation**. Under sustained load, short requests keep arriving, and a long prompt may never get the budget. Production schedulers bound it, typically by *aging*: a request's priority grows with the time it has waited, so that eventually it goes first whatever its size (exercise 1). Other orders exist for other goals: priorities set by the client (interactive before batch jobs), fairness between users (so that one heavy user cannot crowd out the others), or a prediction of the answer's length.

### 3.7 Refusing work: admission control

Above capacity, every request accepted adds to the wait of everyone behind it. A batching engine has two places where requests accumulate, and two limits:

- **The batch** (`max_batch`, or KV memory): every running request slows every step a little, so the batch size sets TPOT. Chapter 23 measured that throughput stops growing near 16 requests on this machine, so this chapter's engine runs at most 16; more would only make every stream slower.
- **The queue**: requests waiting for a place in the batch. Their wait is TTFT. Chapter 24's queue had no limit.

Little's law sizes the queue. A request that finds `L` requests waiting before it waits about `L / μ`, μ being the rate at which requests complete and free their places. Part 1 measured μ ≈ 3.5 per second. With a TTFT target of 1 s, about 0.15 s of which is the request's own prefill, the queue should hold about 0.85 × 3.5 ≈ 3 requests; beyond that, a new request is rejected at once (`max_waiting: Some(3)`).

Part 4 sends 90 chat requests at 6 per second, about 1.7 times what the engine can complete, without and with that limit:

```text
== 4. overload: 90 chat requests at 6 per second, about 1.7 times capacity
   no limit: 90 served, 0 rejected, longest queue 26; 26 met the SLO; all done after 24.9 s
      served: TTFT p50 2904 ms, p90 6541 ms, max 7596 ms; TPOT p50 66 ms, p90 68 ms
      median TTFT by fifth of arrivals:  174 ms  1153 ms  2773 ms  4990 ms  6541 ms
   at most 3 waiting: 67 served, 23 rejected, longest queue 3; 58 met the SLO; all done after 18.8 s
      served: TTFT p50 507 ms, p90 1137 ms, max 1827 ms; TPOT p50 61 ms, p90 70 ms
      median TTFT by fifth of arrivals:  216 ms   396 ms   631 ms  1021 ms   554 ms
```

Without a limit, every request is served, eventually. The queue grows to 26, and the median first token grows through the run, from 0.17 s for the first fifth of arrivals to 6.5 s for the last: 26 waiting requests at 3.5 per second is 7 s. Only 26 requests meet the SLO, the early ones. With the limit, 23 requests are refused within a millisecond of arriving, the queue never exceeds 3, the first tokens stay near the target throughout, and 58 requests meet the SLO instead of 26. (Across three runs: 26-31 requests within the SLO without the limit, 52-60 with it.) TPOT is the same in both runs: the batch was full in both, and the batch limit, not the queue limit, sets TPOT.

The limit bounds the *average* wait; the 90th-percentile first token is still a little over the target (1.14 s), because the requests waiting ahead are of different sizes and a just-admitted request still waits its turn for the step's prompt budget. A smaller limit, or a rule that predicts each request's wait from the work ahead of it rather than counting requests, tightens the tail (exercise 2).

Refusing is kinder to clients than it looks. A rejection that arrives in a millisecond can be retried on another server, or shown to the user honestly. A first token after 6 s is, for most clients, a timeout, often followed by an automatic retry: the same work submitted again, which makes the overload worse. That feedback loop is called a *retry storm*, and an unbounded queue is how it starts.

## 4. The code

The metrics are in [`src/metrics.rs`](src/metrics.rs), the load generators in [`src/loadgen.rs`](src/loadgen.rs), the engine in [`src/engine.rs`](src/engine.rs) (chapter 24's, with the changes listed in its header), the demo in [`src/main.rs`](src/main.rs).

### 4.1 Percentiles and a client's trace

<!-- file: src/metrics.rs -->
```rust
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty(), "percentile of nothing");
    assert!(p > 0.0 && p <= 100.0, "percentile {p} out of range");
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}
```

Nearest rank: the value at position ⌈p/100 × n⌉ (counting from 1) of the sorted samples. It always returns a measured value, never an interpolation between two, and for p = 50 and an even count it returns the lower middle value. Libraries differ in their definitions; with thousands of samples the difference vanishes, with tens it does not, which is one more reason to report the sample count.

Each client records a `Trace`: when it sent its request, and when each token arrived, measured on the client's side, so that queueing anywhere between client and model is included. From it come TTFT (the first arrival), TPOT (the average gap after the first) and the gaps themselves:

<!-- file: src/metrics.rs -->
```rust
    pub fn met(&self, t: &Trace) -> bool {
        !t.rejected
            && t.ttft().is_some_and(|x| x <= self.ttft)
            && t.tpot().is_none_or(|x| x <= self.tpot)
    }
```

A request meets the SLO if it was served, its first token came in time, and its average gap was small enough (a one-token answer has no gaps, and passes that test).

### 4.2 Poisson arrivals

<!-- file: src/loadgen.rs -->
```rust
pub fn poisson_arrivals(rate: f64, n: usize, seed: u64) -> Vec<Duration> {
    let mut rng = Rng::new(seed);
    let mut t = 0.0;
    (0..n)
        .map(|_| {
            // Inverse transform: if u is uniform in (0, 1], -ln(u) / rate
            // is exponential with mean 1 / rate.
            let u = 1.0 - rng.next_f64();
            t += -u.ln() / rate;
            Duration::from_secs_f64(t)
        })
        .collect()
}
```

The exponential distribution's cumulative probability is `1 − e^(−λx)`; solving `u = e^(−λx)` for `x` turns a uniform random number into an exponential one. `next_f64` (chapter 15's seeded generator) returns values in `[0, 1)`; `1 − u` moves them to `(0, 1]`, so the logarithm never sees 0. The seed makes every run of the demo send the same arrivals. The tests check the mean gap, the share of gaps longer than the mean (e⁻¹ ≈ 37%) and that the standard deviation equals the mean, three properties of the exponential distribution.

### 4.3 Hundreds of clients on one thread

<!-- file: src/loadgen.rs -->
```rust
pub fn run_open(engine: &EngineHandle, plan: &[Planned]) -> Vec<Trace> {
    runtime().block_on(async {
        let start = Instant::now();
        let clients = plan.iter().map(|p| async move {
            tokio::time::sleep_until((start + p.at).into()).await;
            client(engine, p, start).await
        });
        join_all(clients).await
    })
}
```

Each planned request becomes a future that sleeps until its send time, then submits and records the arrival time of each token (`client`). `join_all` runs them all concurrently on a single-threaded tokio runtime: one thread, however many clients, so the load generator takes as little CPU as possible from the engine it measures. Sleeping *until* an absolute time, rather than *for* the gap since the previous request, means a late wake-up does not shift every later request.

The closed loop (`run_closed`) is `users` futures, each taking the next request from a shared counter, sending it, and awaiting its answer before taking another.

### 4.4 The policy in the engine

Two places in chapter 24's engine decide an order. Admission picks the next waiting request:

<!-- file: src/engine.rs -->
```rust
    fn next_waiting(&self) -> Option<usize> {
        if let Some(i) = self.waiting.iter().position(Active::started) {
            return Some(i);
        }
        match self.policy {
            Policy::Fcfs => (!self.waiting.is_empty()).then_some(0),
            Policy::ShortestFirst => {
                (0..self.waiting.len()).min_by_key(|&i| self.waiting[i].tokens.len())
            }
        }
    }
```

Requests that were preempted (chapter 24) come back first under both policies: they were admitted once and already hold their place in line. Then FCFS takes the front of the queue, and shortest first the shortest prompt; `min_by_key` returns the first of equal minimums, so ties keep their order of arrival.

The step plan hands out the prompt budget, after every decode token has been planned:

<!-- file: src/engine.rs -->
```rust
        // Prompt chunks: in admission order, or fewest tokens left first.
        let mut order: Vec<usize> = (0..self.active.len()).collect();
        if self.policy == Policy::ShortestFirst {
            order.sort_by_key(|&i| self.active[i].remaining());
        }
```

`remaining()` is the number of tokens still to be computed, so a long prompt that is nearly done counts as short: that is *remaining* processing time, not total. `sort_by_key` is a stable sort, so equal counts also keep admission order.

### 4.5 A bounded queue

<!-- file: src/engine.rs -->
```rust
        if self.max_waiting.is_some_and(|n| self.waiting.len() >= n) {
            Stats::add(&self.stats.rejected, 1);
            let _ = job.events.send(Event::Rejected(format!(
                "overloaded: {} requests are already waiting",
                self.waiting.len()
            )));
            return;
        }
```

The check happens when a job is received, before it is queued, so a rejected request costs the engine almost nothing. Chapter 22's server turns `Event::Rejected` into `503 Service Unavailable` for a whole answer; for a streamed one it has already sent `200 OK` and its headers by the time the engine answers, so the rejection arrives as an error event. A server that wants to answer overload with `429 Too Many Requests` and a `Retry-After` header has to wait for the engine's first event before sending headers (exercise 4).

## 5. Run it

```bash
cargo test -p ch25-scheduling
cargo run --release -p ch25-scheduling                  # all four parts, about 5 minutes
cargo run --release -p ch25-scheduling -- chunks        # or: load, policy, overload
```

The tests check the nearest-rank percentiles (including that a p99 of 10 values is their maximum); TTFT, TPOT and gaps from a trace; that the SLO counts rejections and both latencies; the Poisson generator's mean, spread and share of long gaps; that shortest first admits a short prompt before a long one that arrived earlier, and gives it the prompt budget when both are running (using the engine's own timestamps, which are exact, rather than the test threads'), with answers identical to chapter 14's; that a full queue rejects at once and the queued request is still served; and that both load generators record every token and send open-loop requests no earlier than planned.

## 6. The Rust behind it

**Concurrency without threads or `'static`.** `run_open` builds a future per client and runs them with `join_all` inside `block_on`. Nothing is spawned, so the futures may borrow `plan` and the engine handle from the caller; `tokio::spawn` would require `'static` futures, and therefore clones of everything they use.

**`Cell` and `RefCell` instead of `Mutex`.** The closed loop's users share a counter and the vector of traces. On a current-thread runtime they all run on one thread, so `Cell` (for a `Copy` value) and `RefCell` (for the vector) are enough, and the compiler checks that: those types are not `Sync`, so the futures could not be moved to a multi-threaded runtime by mistake.

**`Duration` cannot be negative.** `a - b` on two `Duration`s panics if `b > a`, and Clippy (`unchecked_time_subtraction`) asks you to say what should happen instead. The traces' times are ordered, so `saturating_sub` documents that the difference is never negative, without a panic path.

**`Option::is_some_and` and `is_none_or`.** `Slo::met` reads like its definition: the TTFT exists and is within the target; the TPOT does not exist or is within the target.

**A stable sort for policies.** `sort_by_key` keeps equal elements in their original order (it is a stable merge sort); `sort_unstable_by_key` does not. For a scheduling order that matters: requests with the same number of tokens left stay in arrival order.

## 7. Mistakes you will make

- **Benchmarking with a closed loop and calling it latency under load.** It cannot show overload, and it hides stalls (coordinated omission). Use arrival rates.
- **Averages instead of percentiles, and percentiles from too few samples.** A p99 from 50 requests is the maximum of 50 requests.
- **Sending every request at time 0.** That measures one burst, not a load.
- **A run too short to see overload.** Above capacity, latency grows with the run's duration; a 10-second test can look acceptable. Watch the latency by arrival time, as part 4 does.
- **Measuring latency inside the engine only.** It misses every queue in front of it. Timestamp on the client.
- **Tuning one number.** A tiny step budget minimizes stalls and wrecks first tokens; shortest first without aging can starve long prompts; a large batch maximizes throughput and breaks the TPOT target.
- **An unbounded queue.** It does not serve more requests; it serves all of them late.
- **Reporting throughput when users care about goodput.**

## 8. How the professionals do it

- **vLLM** schedules with a token budget per step (`max_num_batched_tokens`) and a batch limit (`max_num_seqs`); chunked prefill, decode tokens first, is the default in its V1 engine. It offers `fcfs` and `priority` scheduling policies. Its benchmark (`vllm bench serve`) sends Poisson arrivals at `--request-rate` (with `--burstiness` for burstier gamma-distributed gaps), reports TTFT, TPOT and ITL percentiles, and computes goodput against SLOs given with `--goodput`.
- **Sarathi-Serve** (Agrawal et al., OSDI 2024) introduced chunked prefills with stall-free, decode-first batching, measuring the trade-off of section 3.5 on GPUs.
- **Disaggregation.** DistServe (Zhong et al., OSDI 2024), Splitwise (Patel et al., ISCA 2024) and Mooncake (Moonshot AI's system for Kimi) run prefill and decode on different GPUs, so that prompts never stall streams at all, and send the KV cache from one to the other. DistServe introduced goodput as the metric to optimize.
- **Text Generation Inference** bounds prefill work per batch (`--max-batch-prefill-tokens`), the tokens in flight (`--max-batch-total-tokens`) and concurrent requests (`--max-concurrent-requests`, beyond which it answers 429).
- **Fairness.** The Virtual Token Counter (Sheng et al., OSDI 2024) schedules so that each client gets a fair share of tokens, whatever their request sizes.
- **Load testing.** NVIDIA's GenAI-Perf, Anyscale's LLMPerf and GuideLLM measure LLM servers with arrival rates and per-token timestamps. `wrk2` was written by Gil Tene to measure web servers without coordinated omission.
- **SLOs** as an engineering practice, with error budgets (the share of requests allowed to miss), come from Google's site reliability engineering.

## 9. Exercises

1. Make `ShortestFirst` starvation-free with aging: a request that has waited more than 2 s goes before any request that has not. First build a workload that starves a long prompt under plain `ShortestFirst` (a steady stream of short requests at about 80% of capacity), then show that aging bounds its wait.
2. Predictive admission: estimate a new request's wait as (prompt tokens of the requests ahead of it plus its own) × the measured cost per prompt token, plus the time until a place in the batch frees, and reject it if the estimate exceeds the TTFT target. Compare with `max_waiting = 3` in part 4: rejections, attainment, and the 90th-percentile first token.
3. Run part 1 with `max_batch` 32 and a TPOT target of 50 ms. Which latency now limits goodput, and at what rate?
4. Make chapter 22's server answer overload with `429 Too Many Requests` and `Retry-After`: wait for the engine's first event before sending the response headers, and map an overload rejection to 429. What does that wait cost a streamed request that is accepted?
5. Closed loop with thinking: 32 users who each wait 5 s after every answer before asking again. Predict the arrival rate with Little's law, measure it, and compare the latencies with an open loop at that rate.
6. Add a `priority` to requests (interactive or batch). Batch requests are admitted only when no interactive request waits, and get prompt budget only after interactive ones. Measure interactive TTFT with and without a background of batch requests that keeps the engine busy.

## 10. Check yourself

1. What does TTFT consist of, and what determines TPOT?
2. Why is the 99th percentile of 40 measurements their maximum?
3. A benchmark with 16 concurrent users reports 3.7 requests per second with a 90th-percentile TTFT of 1.4 s. Why can you not conclude that the server handles 3.7 requests per second of real traffic with that latency?
4. Why does waiting time grow sharply well before utilization reaches 100%?
5. What does the step budget trade against what, and why did a budget of 128 cost long prompts so little on this machine?
6. Why does shortest-first reduce the average wait, and what is its risk?
7. An engine completes 5 requests per second, and its TTFT target is 2 s, of which prefill takes 0.2 s. How many requests should be allowed to wait?
8. Why can a server that queues everything serve fewer requests within the SLO than one that rejects some?

## 11. Recap

- Users feel TTFT (waiting plus prefill) and TPOT (the decode step); services promise percentiles of them in an SLO and maximize goodput, the requests per second that meet it.
- Closed-loop load adapts to the server and hides overload; open-loop Poisson arrivals behave like independent users. At the same throughput, the closed loop reported a 1.4 s p90 TTFT and the open loop 3.6 s and growing.
- Latency is flat while batching absorbs the load (up to 3 requests per second here), then rises steeply near capacity (3.5 per second); above it, the queue grows without bound and goodput falls even as throughput rises.
- The step budget bounds stalls: with whole prompts in one step, streams froze for 2.6 s; with 128 tokens per step, for at most 0.35 s, and the long prompts got their first tokens no later, because a 128-token step is already compute-bound.
- Shortest prompt first cut the short requests' p90 TTFT from 1.6 s to 0.41 s with long prompts unaffected; without aging it can starve them.
- The batch limit sets TPOT; the queue limit sets TTFT. Sized by Little's law (3 waiting), it turned 1.7 times overload from 26 requests within the SLO into 58, by rejecting 23 at once.

## Answers

**Exercises**

1. Keep the arrival time of each waiting request; in `next_waiting` and in the plan's order, put requests that have waited more than the threshold first (oldest first), then the rest by length. With plain `ShortestFirst` and a steady stream of short requests, the long prompt gets the budget only when no short prompt is pending, which at 80% utilization is rarely, so its wait can grow to many seconds; with aging it waits at most about the threshold plus its own prefill.
2. The estimate uses information the count ignores: three waiting 1,000-token prompts are a 7 s wait, three 70-token prompts half a second. Expect fewer rejections when the waiting requests are short, more when they are long, and a 90th-percentile TTFT closer to the target. The estimate's weak point is the time until a place in the batch frees, which depends on answer lengths the engine does not know; the average completion interval (1/μ) is a reasonable stand-in.
3. With 32 requests per step, each decode step takes about twice as long as with 16 (chapter 23), so TPOT reaches 50 ms at a lower load (part 1 measured 41 ms at 4 requests per second with 16; expect 50 ms to be crossed around 3 per second with 32, and even at 2 per second during bursts). Goodput becomes limited by TPOT, not TTFT: the larger batch admits more requests, so first tokens improve, but streams slow down for everyone.
4. Read events until the first one arrives: `Token` means accepted (send 200 and stream), `Rejected` with an overload message means 429. An accepted stream's headers are delayed until its first token, so the client cannot tell "accepted, waiting" from "not yet answered"; that matters for long queues, where the SSE keep-alive comments of chapter 22 no longer reach the client while it waits. A bounded queue keeps that wait short.
5. Each user cycles through one request (about 0.15 + 63 × TPOT seconds) and 5 s of thinking. With TPOT around 30 ms, a cycle is about 7 s, so 32 users send about 32 / 7 ≈ 4.6 requests per second, near capacity; as the server slows, the cycles get longer and the rate drops, which the open loop at 4.6 per second does not do. Expect the closed loop to show lower tail latencies at the same average rate.
6. A field on `Request` (or a separate queue per class) and two orders: in `next_waiting`, interactive first; in the plan, interactive prompts first. With a background of batch work, interactive TTFT should approach its unloaded value plus at most one step, while the batch requests absorb the waiting. Note that running batch requests still slow every step (TPOT), which only preemption or a lower batch limit for them prevents.

**Check yourself**

1. The time waiting for a place in the batch (queueing), then the prefill of the prompt, possibly spread over several steps; TPOT is the duration of the decode steps, which grows with the number of requests in the batch and with any prompt work sharing those steps.
2. With the nearest-rank definition, the p-th percentile is the value at rank ⌈p/100 × n⌉; for p = 99 and n = 40, that is rank ⌈39.6⌉ = 40, the largest value.
3. In a closed loop, the 16 users wait for their answers, so no more than 16 requests are ever in the system and requests that would have arrived during a stall are not sent. Real traffic keeps arriving in bursts regardless; at that rate (near capacity), the open loop measured a 90th-percentile TTFT of 3.6 s, growing with the run.
4. Because arrivals are random: bursts bring requests while others are still in service, and the queue they leave takes time to drain. The closer the utilization to 100%, the less spare capacity there is to drain it; for the M/M/1 queue the wait is ρ / (1 − ρ) service times, 4 at 80% and 9 at 90%.
5. A larger budget finishes prompts in fewer, longer steps, giving their first tokens sooner but stalling every decoding stream for the length of the step; a smaller budget bounds the stalls but takes more steps per prompt. On this machine, a step of 128 tokens is already compute-bound (past the ridge point of about 13 rows), so 128-token chunks cost per token what a whole prompt costs, and the long prompt's prefill takes about as long either way.
6. Serving short jobs first means few requests wait behind long ones: each long job delays many short ones under FCFS, while under shortest-first each short job delays a long one by only its own small size, so the total (and average) waiting falls. The risk is starvation: with a steady supply of short requests, a long one may wait indefinitely unless its priority grows with its waiting time.
7. The wait budget is 2 − 0.2 = 1.8 s; at 5 completions per second, `L = λW` gives 1.8 × 5 = 9 requests.
8. A queue with no limit makes every request wait for all the ones before it; above capacity that wait grows until nearly every request misses the TTFT target, so very few count as good. Rejecting the excess at once keeps the wait of those admitted within the target, so all of them can count. The rejected requests were going to miss the target anyway, and they learn it in a millisecond instead of after seconds.

## Further reading

- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI 2024.
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024.
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", ISCA 2024.
- Sheng et al., "Fairness in Serving Large Language Models", OSDI 2024.
- Dean and Barroso, "The Tail at Scale", Communications of the ACM, 2013.
- Harchol-Balter, *Performance Modeling and Design of Computer Systems: Queueing Theory in Action*, 2013. Queues, Little's law, and why shortest-remaining-time-first works.
- Schrage, "A Proof of the Optimality of the Shortest Remaining Processing Time Discipline", Operations Research, 1968.
- Gil Tene, "How NOT to Measure Latency" (talk, 2015): coordinated omission.
- Beyer, Jones, Petoff and Murphy (eds.), *Site Reliability Engineering*, O'Reilly 2016, chapter 4, "Service Level Objectives".
- Next: [Chapter 26: Speculative decoding](../26-speculative-decoding/README.md). A small model drafts tokens and the large one checks several at once.
