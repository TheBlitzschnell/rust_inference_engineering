# Chapter 30: Inference in production

> **In one sentence:** running a model as a service adds problems that no single request shows: knowing whether the numbers are right (compare hidden states layer by layer with a trusted implementation), knowing how the service is doing (counters and histograms, and the percentiles estimated from them), having enough of it (capacity planned from measured throughput and queueing theory, with headroom), and staying up when something goes wrong (timeouts, retries and queues that recover after a stall instead of amplifying it).

**Where this fits:** this is the last chapter. Chapters 21 to 25 built a server; chapters 26 to 29 made it faster and more capable. This chapter is about operating it: the checks, measurements and policies that decide whether a model in production is right, observed, sized and reliable.

**You need:** chapter 16 (SmolLM2 and the PyTorch reference fixtures), chapter 19 (4-bit weights), chapter 22 (the server, 429s), chapter 25 (the scheduling engine, SLOs, Little's law, queueing), chapter 28 (the one-token layer loop).

**You will build:** a tracer that records the hidden state entering every layer and a comparison that finds where two implementations part; a metrics registry with counters, gauges and histograms in Prometheus's text format, and its bucket-based quantile estimate; a capacity planner based on the M/M/c queue (Erlang's C formula); and a simulation of timeouts and retries around a 5-second stall. Then: SmolLM2-135M against PyTorch layer by layer, an injected bug and 4-bit weights; metrics from chapter 25's engine under load; a replica count for a forecast; and four retry policies through an outage.

---

## 1. The intuition

Running a restaurant chain, not cooking one dinner. Four new questions come up.

- **Is the food right?** When a dish tastes wrong, tasting the final plate tells you *that* something is wrong. Tasting at every station, next to the same dish from a kitchen you trust, tells you *where*.
- **How are we doing?** No one can watch every table. The manager reads counts: dishes served, complaints, and how many waited under five minutes, under ten, under twenty.
- **Do we have enough kitchens?** Enough for the Friday-night peak, with room for a rush and for one kitchen being closed.
- **What happens when something breaks?** If the kitchen stops for five minutes, a queue forms; if the waiters keep re-taking orders from impatient guests, the kitchen restarts to a pile of orders whose guests have left, cooks them, and every new guest waits behind them. The kitchen is back, and the restaurant is still failing.

**Where the analogy breaks:** a model's arithmetic has no taste. A wrong answer can look perfectly fluent, which is why the checks must be numerical, not by reading the output.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Reference implementation** | A trusted version of the model (here PyTorch) whose intermediate values are the expected ones. |
| **Cosine similarity** | `a·b / (|a| |b|)`: 1 when two vectors point the same way. |
| **Counter, gauge** | A metric that only goes up (requests served), and one that goes up and down (requests waiting). |
| **Histogram** | Counts of observations per bucket (`le`, "less than or equal"), with their sum and count. |
| **Exposition format** | The text a server returns at `/metrics` for Prometheus to scrape. |
| **Headroom** | Capacity kept unused at the expected peak, for bursts and failures. |
| **M/M/c queue, Erlang C** | Poisson arrivals, `c` servers with exponential service times; Erlang's C formula gives the probability of waiting. |
| **N+1** | One more replica than the load needs, so that one can fail or be upgraded. |
| **Timeout, retry, backoff, jitter** | A client gives up on an attempt after a timeout and tries again, waiting longer each time (backoff), by a random amount (jitter). |
| **Metastable failure** | An overload that sustains itself after its trigger is gone. |
| **Deadline propagation** | Passing the client's deadline with the request, so the server can skip work nobody will receive. |
| **Canary** | Sending a small share of traffic to a new version first, and comparing. |

## 3. The concepts in depth

### 3.1 Is the model right? Layer by layer

Most bugs in an inference engine do not crash it. A norm weight not loaded, a RoPE dimension paired wrongly, a transposed matrix, keys stored at the wrong position: the model still produces fluent text, a little worse, or very different. Comparing final logits with a reference says *that* something is wrong. Comparing the hidden state **entering every layer** says *where*: the layer whose output first departs from the reference is the layer to read.

Chapter 16's fixture, made with PyTorch and Hugging Face's implementation, holds exactly that: for one chat prompt, the hidden state at the last position entering each of the 30 layers, and after the final norm. Part 1 records the same values with our model (`trace_last`, which runs chapter 28's one-token layer loop and keeps a copy before each layer) and compares, with relative difference `|ours − theirs| / |theirs|`:

```text
== 1. hidden state entering each layer (last prompt position), relative difference
   layer:                            0       1       5      10      12      13      14      29   final   first jump
   ours vs PyTorch               0.0e0  1.2e-6  1.0e-6  1.0e-6  9.2e-7  9.9e-7  1.0e-6  1.4e-6  1.8e-6   none
   layer 12 norm bug vs ours     0.0e0   0.0e0   0.0e0   0.0e0   0.0e0   7.8e0   7.9e0   1.4e0   1.3e0   input of layer 13
   4-bit weights vs ours         0.0e0  8.1e-2  1.7e-1  2.3e-1  1.9e-1  1.9e-1  2.0e-1  2.8e-1  3.8e-1   input of layer 1
   (ours vs PyTorch: lowest cosine similarity 1.0000000)
   next token: PyTorch "The", ours "The", with the bug " \"", 4-bit "The"
```

Our `f32` model matches PyTorch to about 10⁻⁶ at every layer: rounding, from adding in a different order (chapter 2), with no growth worth the name. That is what "correct" looks like.

Then two changed models, compared with ours:

- **A bug.** Layer 12's MLP norm weights are replaced by ones, as if they had not been loaded. Nothing differs until the input of layer 13, then everything does (a relative difference of 7.8). `first_jump`, which looks for a difference more than 10 times anything before it, points at the input of layer 13: the bug is in layer 12. The next token changes from "The" to a quotation mark.
- **Drift.** Every layer's weights quantized to 4 bits and back (chapter 19), the embedding untouched. The first quantized layer already moves the hidden state by 8%; then the difference grows slowly, to 38% after the final norm, without a jump. And the next token is still "The". Quantization error is spread over every layer and, for most tokens, harmless; a bug sits in one place and is not.

A practical routine follows. Keep reference activations for a few prompts, from the original implementation. Compare after every change to kernels, formats or loading code. Look for the first layer where the difference jumps; drift that grows smoothly is a precision question (chapters 18-19), a jump is a bug.

### 3.2 How is it doing? Metrics

A service is observed through numbers it exports continuously: Prometheus (or a compatible system) fetches `GET /metrics` every few seconds and stores the time series. Three kinds:

- **Counters** only go up: requests by status, tokens generated. Their rate of increase is what matters (requests per second, errors per second).
- **Gauges** go up and down: requests waiting, requests running, KV blocks in use.
- **Histograms** count observations per bucket: how many first tokens took at most 0.1 s, at most 0.25 s, and so on, with the sum and count. Latency percentiles are *estimated* from them later, by the monitoring system.

A server cannot keep every request's latency; a histogram keeps a dozen counts. The price: a percentile is estimated by finding the bucket that contains it and assuming the observations are spread evenly within it (Prometheus's `histogram_quantile`). It is only as precise as the buckets around it.

Part 2 runs chapter 25's engine (shortest-prompt-first, 128-token step budget, at most 3 waiting) with 70 chat requests at 3.5 per second, slightly more than this machine can serve, and records what a server would export:

```text
== 2. metrics from 70 chat requests at 3.5 per second (chapter 25's engine and SLO)
   generated_tokens_total 3584
   requests_total{status="200"} 56
   requests_total{status="429"} 14
   ttft_seconds_bucket{le="0.005"} 0
   ttft_seconds_bucket{le="0.01"} 0
   ttft_seconds_bucket{le="0.025"} 0
   ttft_seconds_bucket{le="0.05"} 0
   ttft_seconds_bucket{le="0.1"} 0
   ttft_seconds_bucket{le="0.25"} 22
   ttft_seconds_bucket{le="0.5"} 35
   ttft_seconds_bucket{le="1"} 47
   ttft_seconds_bucket{le="2.5"} 54
   ttft_seconds_bucket{le="5"} 56
   ttft_seconds_bucket{le="10"} 56
   ttft_seconds_bucket{le="+Inf"} 56
   ttft_seconds_sum 32.810598155
   ttft_seconds_count 56
   TTFT percentile     exact   default buckets   chosen buckets
   p50                   317 ms            365 ms            329 ms
   p90                  1299 ms           1729 ms           1425 ms
   p99                  2743 ms           4300 ms           2813 ms
   (56 served, 14 rejected; 67% met the SLO)
```

The exposition is what Prometheus reads: `requests_total` by status (14 requests were refused with 429 by the bounded queue), tokens generated, and the TTFT histogram's cumulative buckets. With Prometheus's default buckets (5 ms to 10 s, spaced for web requests), the estimated 99th percentile is 4.3 s against an exact 2.7 s: the true value lies in the bucket from 2.5 to 5 s, and interpolation guesses within that wide bucket. Buckets chosen for this service (finer between 100 ms and 3 s, around the SLO) give 2.8 s. Choose buckets where the decisions are made: around the SLO's thresholds.

What an inference service should export, at least: requests by status (and by finish reason), tokens in and out, TTFT and time per output token as histograms, requests running and waiting, batch size, KV-cache blocks in use, preemptions (chapter 24), prefix-cache hit rate. Alert on the SLO itself (the share of requests over the TTFT target, from the histogram), on error rates, and on a queue that stays full.

### 3.3 Do we have enough? Capacity planning

Capacity planning turns a forecast ("40 requests per second at the evening peak") into a number of replicas. Two inputs: what one replica can do, measured under realistic load (chapter 25), and a model of how waiting grows with load.

Part 3 measures one replica of this machine at saturation (16 closed-loop users, as in chapter 25), then applies the **M/M/c** model: Poisson arrivals at `λ`, `c` replicas each completing `μ` requests per second. Erlang's C formula gives the probability that a request finds every replica busy and must wait, and the mean wait follows:

```text
== 3. capacity: one replica (this machine) completes 3.12 chat requests per second at saturation
   forecast peak: 40 requests per second (M/M/c model, not measured)
   replicas   utilization   P(wait)   mean wait
         13           99%       94%      1672 ms
         14           92%       67%       181 ms
         15           85%       46%        68 ms
         16           80%       31%        31 ms
         17           75%       20%        15 ms
         18           71%       12%         8 ms
         19           67%        8%         4 ms
   for a mean wait under 100 ms and P(wait) under 20%: 17 replicas, 18 with one spare
```

Thirteen replicas would be 99% busy: requests wait 1.7 s on average, and 94% of them wait. Two more bring the wait to 68 ms; 17 meet a target of under 100 ms with under 20% of requests waiting, at 75% utilization. Then add one spare (N+1), because a replica will fail or be redeployed during the peak.

Two lessons in the table. First, the waiting curve is steep near full utilization (chapter 25's knee, now for a fleet): 13 to 15 replicas is the difference between broken and fine. Second, **pooling**: a fleet can run hotter than a single server for the same wait. One replica at 75% utilization makes a request wait with probability 0.75; seventeen at 75% make it wait with probability 0.20, because a request only waits when *all* of them are busy. That is why large shared deployments are cheaper per request than many small dedicated ones.

The model is rough: a batching replica is not one server with exponential service times, and real traffic has daily cycles and bursts larger than Poisson. Use it to choose a starting point, then verify with an open-loop load test at the forecast rate (chapter 25), and keep the headroom.

### 3.4 Staying up: timeouts, retries and metastable failures

Clients protect themselves with timeouts: give up on an attempt after, say, one second, and retry. Each client is right to do so. Together, they can keep a service down long after its problem has gone.

Part 4 simulates it (no model involved): requests arrive at 80 per second, the server can complete 100 per second from a FIFO queue, clients time out after 1 s and make at most 4 attempts. At 10 s the server stalls for 5 seconds (a garbage collection, a failover, a GPU reset):

```text
== 4. a 5-second stall at 10 s: 80 requests/s, capacity 100/s, 1 s client timeout, 4 attempts (simulated)
   policy                       good requests per second       wasted   lost   longest
                              5-10 s  15-20 s  20-30 s  30-40 s  work            queue
   retry at once                  80        0        0        0     2500   2074     6609
   backoff with jitter            80        0        0        0     2500   1933     6389
   drop expired requests          80      100       86       80        0    245     1109
   all three, queue limit 100     80      100       83       80        0    268      100
```

**Retrying at once:** during the stall, the queue fills with 400 new requests and their retries. When the server comes back at 15 s, it serves the queue in order, and every request it serves has waited more than a second: its client has already given up. All the work is wasted. Meanwhile every new request also waits behind the stale ones, times out and retries, adding up to four attempts per request: 320 attempts per second for a server that completes 100. The queue grows to 6,600, and 25 seconds after the stall ended, the server is fully busy and has not answered a single request in time. Nothing is broken any more; the overload sustains itself. That is a **metastable failure** (Bronson et al., 2021), and it has caused real, long outages at large services.

**Backoff with jitter** alone does not help here: it spreads retries over time, but every request still makes its four attempts, every attempt still waits behind stale work, and the queue still grows. Backoff prevents synchronized retry waves; it does not reduce the total load of retries when everything is failing. For that, clients need a **retry budget** (for example, retries limited to 10% of requests, as Google's SRE practice recommends) or a circuit breaker that stops retrying when most attempts fail.

**Dropping expired requests** fixes it at the server: if each request carries its client's deadline, the server can skip, at no cost, every queued attempt whose client has already given up. After the stall, it discards the stale queue in an instant, serves fresh requests, and is back to 80 good requests per second immediately (100 in the first seconds, clearing the backlog of recent attempts). A **bounded queue** (chapter 25's admission control) adds a second guarantee: the queue never exceeds 100, about one timeout's worth of work, so it cannot hold work that will be stale when served.

The rule: a server should never do work that no one will receive. Propagate deadlines, bound queues, and make retries bounded, delayed and randomized.

### 3.5 Changing it safely

Most production incidents follow a change. For an inference service the risky changes are new kernels, new quantization, new model versions and new engine versions, and their failures are often numerical: no errors, slightly different answers. Three practices:

- **Golden tests.** A set of prompts with the expected greedy outputs (chapter 16's `generation.json`) or, better, expected logits within a tolerance, run before every deployment. Section 3.1's layer comparison is the tool for when they fail.
- **Canaries.** Send a small share of real traffic to the new version, and compare its metrics (latency, errors, output lengths, finish reasons) with the old one before sending more.
- **Shadow evaluation.** For model changes, run quality evaluations (perplexity, chapter 18; task benchmarks) on the new version before it serves anyone.

## 4. The code

The tracer and comparison are in [`src/trace.rs`](src/trace.rs), the metrics in [`src/metrics.rs`](src/metrics.rs), the capacity model in [`src/capacity.rs`](src/capacity.rs), the outage simulation in [`src/outage.rs`](src/outage.rs), the demo in [`src/main.rs`](src/main.rs).

### 4.1 Recording every layer's input

<!-- file: src/trace.rs -->
```rust
    for (pos, &token) in tokens.iter().enumerate() {
        let last = pos + 1 == tokens.len();
        model.embed.row_to_f32(token as usize, &mut x);
        for l in 0..c.num_layers {
            if last {
                rows.push(x.clone());
            }
            run_layers(model, l..l + 1, pool, &mut x, pos, &mut cache, &mut buf);
        }
        cache.advance(1);
    }
```

Every token goes through every layer, one layer at a time, so that the hidden state can be copied before each; only the last token's copies are kept, because that is what the fixture holds. Chapter 28 wrote `run_layers` for pipeline stages, which needed to run a range of layers on a hidden state; here the range is one layer. The test checks the recording against chapter 14's forward pass: the last row through the LM head gives its logits.

### 4.2 A percentile from buckets

<!-- file: src/metrics.rs -->
```rust
    pub fn quantile(&self, q: f64) -> f64 {
        let rank = q * self.count as f64;
        let mut below = 0u64;
        for (i, &c) in self.counts.iter().enumerate() {
            if (below + c) as f64 >= rank && c > 0 {
                let Some(&upper) = self.bounds.get(i) else {
                    return *self.bounds.last().expect("at least one bound");
                };
                let lower = if i == 0 { 0.0 } else { self.bounds[i - 1] };
                return lower + (upper - lower) * (rank - below as f64) / c as f64;
            }
            below += c;
        }
        f64::NAN
    }
```

The rank `q · count` falls in some bucket; the estimate places it proportionally between the bucket's bounds. A rank in the last (`+Inf`) bucket returns the largest finite bound: the histogram cannot say more, which is another reason to put a bound above the slowest latency you care about.

### 4.3 Erlang C

<!-- file: src/capacity.rs -->
```rust
pub fn erlang_c(c: usize, lambda: f64, mu: f64) -> f64 {
    let a = lambda / mu; // offered load, in servers
    let rho = a / c as f64;
    assert!(rho < 1.0, "the replicas cannot keep up: utilization {rho}");
    // Σ a^k / k! for k < c, computed term by term to avoid overflow.
    let mut term = 1.0;
    let mut sum = 0.0;
    for k in 0..c {
        sum += term;
        term *= a / (k + 1) as f64;
    }
    // `term` is now a^c / c!.
    let top = term / (1.0 - rho);
    top / (sum + top)
}
```

The formula is `(aᶜ/c! · 1/(1 − ρ)) / (Σₖ₌₀^{c−1} aᵏ/k! + aᶜ/c! · 1/(1 − ρ))`, with `a = λ/μ` and `ρ = a/c`. Computing `aᵏ` and `k!` separately overflows for a few hundred servers; building each term from the previous one (`term *= a / (k + 1)`) never does. The tests check it against the M/M/1 formula and known values.

### 4.4 The server's loop, and the fix

<!-- file: src/outage.rs -->
```rust
                let Some((r, deadline)) = queue.pop_front() else {
                    budget = 0.0; // idle capacity is not saved up
                    break;
                };
                if s.drop_expired && deadline < now {
                    continue; // skipped without doing the work
                }
                budget -= 1.0;
                if deadline >= now && !requests[r].done {
                    requests[r].done = true;
                    sec.good += 1;
                } else {
                    sec.wasted += 1;
                }
```

Each 10 ms step, the server gains a tenth of its per-second capacity and serves from the head of the queue. An attempt served after its client's deadline is wasted work; with `drop_expired`, it is skipped without spending capacity. That one `if`, made possible by sending the deadline with the request, is the difference between part 4's first and third rows.

## 5. Run it

```bash
cargo test -p ch30-production
cargo run --release -p ch30-production                   # all four parts, about 2 minutes
cargo run --release -p ch30-production -- outage         # or: debug, metrics, capacity
```

The tests check that the recorded last row gives chapter 14's logits and the first row is the token's embedding; that a jump is found and drift and rounding noise are not; Prometheus-style interpolation inside a bucket, the `+Inf` bucket and values on a bound; the exposition format's lines (cumulative buckets included); Erlang C against M/M/1 and known values, pooling, and a replica count; and the outage simulation: everything served without an outage, no good requests at the end with immediate retries, and recovery when expired requests are dropped.

Parts 1 to 3 measured this machine; part 3's replica counts and part 4's outage are model and simulation results, as their headings say.

## 6. The Rust behind it

**`f64` for comparisons of `f32` values.** `diff` converts each element to `f64` before squaring and summing: the sums of 576 squares of values near 1 would lose the digits that a 10⁻⁶ difference lives in, if kept in `f32`.

**`partition_point` for buckets.** `bounds.partition_point(|&b| b < v)` is a binary search returning the first bound not below `v`: the bucket `v` belongs to under Prometheus's "less than or equal" rule.

**`BTreeMap` for stable output.** The registry keeps metrics in `BTreeMap`s, so the exposition lists them in the same order every time, which makes it testable line by line (a `HashMap`'s order changes from run to run).

**`std::fmt::Write` for building text.** `writeln!(out, ...)` appends formatted text to a `String` without a temporary allocation per line; the `Result` is ignored with `let _ =`, since writing to a `String` cannot fail.

**`BinaryHeap<Reverse<_>>` as a timer queue.** The simulation keeps scheduled retries in a heap of `Reverse((time, request))`: a max-heap of reversed keys pops the earliest time first.

## 7. Mistakes you will make

- **Debugging from the final output.** Fluent text and plausible logits say nothing about which layer is wrong. Compare layer by layer.
- **Mistaking drift for a bug, or a bug for drift.** A smooth growth over layers is precision; a jump is a bug.
- **Averages in dashboards.** Use histograms, and alert on percentiles against the SLO.
- **Default histogram buckets.** Put bucket bounds around your thresholds, or the percentiles you alert on will be coarse guesses.
- **Planning for the average load.** Plan for the peak, at 70-80% utilization, plus a spare.
- **Unbounded queues and retries.** Bound the queue, propagate deadlines, limit retries (a budget, not only backoff).
- **Treating a model update as a code update.** It changes outputs; use golden tests, canaries and evaluations.

## 8. How the professionals do it

- **Numerical debugging:** frameworks capture intermediate activations with hooks (PyTorch's forward hooks), and engines provide tensor dumps (llama.cpp's eval-callback example prints every tensor of a forward pass). Porting a model to a new engine is routinely checked layer by layer against the Hugging Face implementation, as in chapter 16.
- **Metrics:** vLLM, SGLang, TGI and TensorRT-LLM's Triton server export Prometheus metrics: TTFT and inter-token latency histograms, running and waiting requests, KV-cache usage, preemptions, prefix-cache hits. Traces (OpenTelemetry) follow single requests through gateways and engines.
- **Capacity:** cloud providers autoscale replicas on queue length or utilization; because model replicas take minutes to start (loading tens of GB of weights), capacity is planned ahead of peaks rather than added during them.
- **Reliability:** deadline propagation is built into gRPC (deadlines travel with calls); Envoy and gRPC support retry budgets; exponential backoff with jitter is standard in cloud SDKs (Brooker, "Exponential Backoff and Jitter", 2015). Google's SRE books describe load shedding, retry budgets and cascading failures.
- **Change safety:** canary deployments, A/B comparisons of output quality, and shadow traffic are standard for model rollouts.

## 9. Exercises

1. **Find a RoPE bug.** In a copy of chapter 28's `run_layers`, apply RoPE with the wrong position (`pos + 1`) in one layer only. Where does part 1's comparison jump, and how large is the jump compared with the norm bug?
2. **Per-token comparison.** Extend `trace_last` to keep every position, not just the last, and compare the int8 model (chapter 18) with ours. Are some tokens much more affected than others?
3. **An alert.** From the TTFT histogram, compute the share of requests over 1 s (the SLO's TTFT target) over the last minute, from bucket counts alone. Which bucket bound must exist for this to be exact?
4. **A retry budget.** Add to the simulation a budget: a client retries only while its retries in the last 10 seconds are fewer than 10% of its requests. Run part 4's first scenario with it.
5. **Plan with a daily cycle.** Traffic follows `λ(t) = 25 + 15 · sin(2πt / 24 h)` requests per second. How many replicas at the peak and at the trough, and how much does scaling between them save, compared with a fixed fleet, if starting a replica takes 5 minutes?
6. **Golden test.** Write a test that runs chapter 25's engine on chapter 16's `generation.json` prompts and compares the greedy outputs with PyTorch's. What tolerance would you use for logits, and why are exact token matches not guaranteed in general?

## 10. Check yourself

1. Why compare hidden states layer by layer instead of only the final logits?
2. In part 1, how do the bug and the 4-bit weights look different?
3. What is the difference between a counter, a gauge and a histogram?
4. Why did the default buckets give a 99th percentile of 4.3 s when the exact value was 2.7 s?
5. Why can 17 replicas run at 75% utilization with only 20% of requests waiting, while one replica at 75% makes 75% wait?
6. What sustains the overload after the stall ends, in part 4's first scenario?
7. Why does backoff with jitter not fix it, and what does?
8. What is N+1, and why plan for it?

## 11. Recap

- Comparing the hidden state entering each layer with a reference locates errors: our model matched PyTorch to about 10⁻⁶ everywhere; a norm bug in layer 12 appeared as a jump to 7.8 at layer 13's input; 4-bit weights drifted smoothly from 8% to 38%.
- Services export counters, gauges and histograms; percentiles are estimated from histogram buckets and are only as good as the buckets: default buckets put the 99th-percentile TTFT at 4.3 s, buckets chosen around the SLO at 2.8 s, exact 2.7 s.
- Capacity planning: one replica measured at 3.1 requests per second; Erlang C says 17 replicas (75% utilization) keep the mean wait under 100 ms at 40 requests per second, plus one spare. Waiting explodes near full utilization, and larger pools can run hotter.
- A 5-second stall with immediate retries became a lasting outage (no request answered in time for the next 25 seconds); backoff alone did not help; dropping expired requests, made possible by propagating deadlines, recovered at once; a bounded queue capped the damage.
- Changes are the main source of incidents: golden tests, canaries and evaluations catch numerical ones.

## Answers

**Exercises**

1. Only that layer's keys and queries are rotated wrongly, so the difference appears at the input of the next layer. Its size depends on the attention pattern: a shift of one position changes the angles slightly for every key, so expect a jump much smaller than the norm bug's (perhaps 10⁻² to 10⁻¹ instead of 7.8), but still a clear jump against the 10⁻⁶ before it. Small bugs are exactly why the comparison looks at jumps relative to what came before, not at absolute sizes.
2. Expect most positions to differ by similar amounts, and some by much more: tokens whose hidden states have large outlier channels (chapter 18), often the first token and some punctuation, amplify quantization error. Per-position comparisons find them.
3. The count of the bucket `le="1"` divided by the total count gives the share at or under 1 s; the share over it is one minus that. It is exact only if 1 s is a bucket bound, which is one reason to put bounds at the SLO's thresholds. Prometheus computes it over a window with `rate()` on the bucket counters.
4. With the budget, retries during the stall stop after the first few, and after the stall the arrivals are mostly fresh requests: about 80 attempts per second plus a small share of retries, below capacity. The stale queue is still served (no deadline dropping), so recovery takes as long as draining it: the queue at the end of the stall, at 20 requests per second of spare capacity. Expect recovery tens of seconds after the stall instead of never.
5. At the peak (40 per second), 17 replicas plus a spare; at the trough (10 per second), Erlang C gives about 6, plus a spare. A fixed fleet sized for the peak runs 18 all day; following the cycle averages roughly 12, a saving near a third. Starting a replica takes 5 minutes, so scaling must lead the forecast by that much, and the headroom must cover the forecast's error.
6. Compare logits within about 10⁻³ (relative), the agreement chapter 16 measured, and greedy tokens until the first position where the two reference logits of the top two tokens are closer than that tolerance: from there, a legitimate rounding difference can pick the other token, and the continuations diverge. Exact matches over long outputs are not guaranteed by any engine whose kernels differ from PyTorch's.

**Check yourself**

1. Many bugs leave the output fluent and plausible; the logits say only that something differs. Comparing every layer's input shows the first layer whose output departs, which is where the bug is.
2. The bug: zero difference up to layer 12, then a jump of seven orders of magnitude at layer 13's input. The 4-bit weights: a difference from the first quantized layer on, growing gradually, with no jump.
3. A counter only increases (requests served); a gauge is a current value that goes up and down (requests waiting); a histogram counts observations per bucket of value, plus their sum and count.
4. The exact value lay in the bucket from 2.5 s to 5 s, and the estimate interpolates within that bucket as if the observations were spread evenly across it; with so wide a bucket, the estimate can be far from the truth.
5. A request waits only when every replica is busy. With one replica at 75%, that is 75% of the time; with 17, all 17 are busy at once much less often, and a request that finds one free does not wait. Pooling smooths the bursts.
6. The queue, full of attempts whose clients have given up: the server spends its capacity on them, so new requests wait more than the timeout too, time out, and retry, adding more attempts than the server can complete.
7. Backoff spreads the retries in time but does not reduce their number: every request still makes all its attempts, and they still wait behind stale work. Dropping expired requests (with deadlines propagated) removes the stale work; a bounded queue keeps it from accumulating; a retry budget limits the extra attempts.
8. One replica more than the load requires, so that the service still meets its targets with one replica down (failure, deployment, maintenance); without it, every routine restart at peak is an overload.

## Further reading

- Beyer, Jones, Petoff and Murphy (eds.), *Site Reliability Engineering*, O'Reilly 2016: chapters on service level objectives, monitoring, handling overload and cascading failures.
- Bronson, Aghayev, Charapko and Zhu, "Metastable Failures in Distributed Systems", HotOS 2021; Huang et al., "Metastable Failures in the Wild", OSDI 2022.
- Brooker, "Exponential Backoff and Jitter", AWS Architecture Blog, 2015.
- Harchol-Balter, *Performance Modeling and Design of Computer Systems*, 2013: the M/M/c queue and Erlang C.
- The Prometheus documentation: metric types, histograms and `histogram_quantile`, and naming best practices.
- This is the last chapter. The [course README](../../README.md) lists every chapter; the [glossary](../../GLOSSARY.md) collects every term.
