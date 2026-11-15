//! Pipeline parallelism: consecutive layers on different devices.
//!
//! Stage `s` of `n` holds a contiguous block of layers. A token's hidden
//! state enters stage 0, goes through its layers, is sent to stage 1, and
//! so on; the last stage computes the logits and sends the chosen token
//! back to stage 0. Each stage keeps the KV cache of its own layers.
//!
//! One sequence alone keeps one stage busy at a time: the others wait (the
//! pipeline "bubble"). Several sequences in flight fill the stages, which
//! is the only way pipelining raises throughput.

use crate::comm::{Comm, run};
use ch07_threads::SpinPool;
use ch08_operators::{add_inplace, rms_norm, softmax, swiglu};
use ch14_kv_cache::{KvCache, Matrix, Model};
use std::ops::Range;
use std::time::{Duration, Instant};

/// Buffers for one token's pass through layers.
pub struct Buffers {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    scores: Vec<f32>,
    matmul: Vec<f32>,
}

impl Buffers {
    pub fn new<W: Matrix>(model: &Model<W>, context: usize) -> Self {
        let c = &model.config;
        Self {
            normed: vec![0.0; c.hidden_size],
            q: vec![0.0; c.q_dim()],
            k: vec![0.0; c.kv_dim()],
            v: vec![0.0; c.kv_dim()],
            attn: vec![0.0; c.q_dim()],
            proj: vec![0.0; c.hidden_size],
            gate: vec![0.0; c.intermediate_size],
            up: vec![0.0; c.intermediate_size],
            act: vec![0.0; c.intermediate_size],
            scores: vec![0.0; context],
            matmul: Vec::new(),
        }
    }
}

/// Runs one token, whose hidden state is `x`, through `layers` of `model`
/// at position `pos`, storing its keys and values in `cache` (which keeps
/// the positions of these layers only). Chapter 14's layer, for one token.
pub fn run_layers<W: Matrix>(
    model: &Model<W>,
    layers: Range<usize>,
    pool: &mut SpinPool,
    x: &mut [f32],
    pos: usize,
    cache: &mut KvCache,
    b: &mut Buffers,
) {
    let c = &model.config;
    let d = c.head_dim;
    let heads = c.heads();
    let scale = 1.0 / (d as f32).sqrt();
    for l in layers {
        let layer = &model.layers[l];
        rms_norm(x, &layer.attn_norm, c.rms_norm_eps, &mut b.normed);
        layer.wq.matmul(pool, &b.normed, &mut b.q, 1, &mut b.matmul);
        layer.wk.matmul(pool, &b.normed, &mut b.k, 1, &mut b.matmul);
        layer.wv.matmul(pool, &b.normed, &mut b.v, 1, &mut b.matmul);
        model.rope().apply_heads(&mut b.q, pos);
        model.rope().apply_heads(&mut b.k, pos);
        cache.store(l, pos, &b.k, &b.v);
        for head in 0..c.num_heads {
            let kvh = heads.kv_head(head);
            let query = &b.q[head * d..(head + 1) * d];
            let scores = &mut b.scores[..=pos];
            for (s, key) in scores
                .iter_mut()
                .zip(cache.keys(l, kvh, pos + 1).chunks_exact(d))
            {
                *s = query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            softmax(scores);
            let out = &mut b.attn[head * d..(head + 1) * d];
            out.fill(0.0);
            for (&p, value) in scores
                .iter()
                .zip(cache.values(l, kvh, pos + 1).chunks_exact(d))
            {
                for (o, v) in out.iter_mut().zip(value) {
                    *o += p * v;
                }
            }
        }
        layer
            .wo
            .matmul(pool, &b.attn, &mut b.proj, 1, &mut b.matmul);
        add_inplace(x, &b.proj);
        rms_norm(x, &layer.mlp_norm, c.rms_norm_eps, &mut b.normed);
        layer
            .w_gate
            .matmul(pool, &b.normed, &mut b.gate, 1, &mut b.matmul);
        layer
            .w_up
            .matmul(pool, &b.normed, &mut b.up, 1, &mut b.matmul);
        swiglu(&b.gate, &b.up, &mut b.act);
        layer
            .w_down
            .matmul(pool, &b.act, &mut b.proj, 1, &mut b.matmul);
        add_inplace(x, &b.proj);
    }
}

/// The most likely next token after the hidden state `x`.
fn next_token<W: Matrix>(model: &Model<W>, pool: &mut SpinPool, x: &[f32], b: &mut Buffers) -> u32 {
    let c = &model.config;
    rms_norm(x, &model.final_norm, c.rms_norm_eps, &mut b.normed);
    let mut logits = vec![0.0; c.vocab_size];
    model
        .lm_head()
        .matmul(pool, &b.normed, &mut logits, 1, &mut b.matmul);
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// What a pipelined generation produced and how busy each stage was.
pub struct PipelineRun {
    /// The new tokens of each sequence.
    pub outputs: Vec<Vec<u32>>,
    /// When each new token reached stage 0, from the start.
    pub arrivals: Vec<Vec<Duration>>,
    pub elapsed: Duration,
    /// Time each stage spent computing (not waiting).
    pub busy: Vec<Duration>,
    /// Bytes each stage sent.
    pub bytes: Vec<usize>,
}

impl PipelineRun {
    /// Decoding only (after every sequence's first new token): new tokens
    /// per second over all sequences, and the average time between two
    /// tokens of one sequence.
    pub fn decode(&self) -> (f64, Duration) {
        let first = self.arrivals.iter().filter_map(|a| a.first()).min();
        let last = self.arrivals.iter().filter_map(|a| a.last()).max();
        let (Some(&first), Some(&last)) = (first, last) else {
            return (0.0, Duration::ZERO);
        };
        let tokens: usize = self
            .arrivals
            .iter()
            .map(|a| a.len().saturating_sub(1))
            .sum();
        let gaps: Duration = self
            .arrivals
            .iter()
            .filter_map(|a| Some(a.last()?.saturating_sub(*a.first()?)))
            .sum();
        let rate = tokens as f64 / last.saturating_sub(first).as_secs_f64();
        (rate, gaps / tokens.max(1) as u32)
    }
}

/// The `i`-th of `n` contiguous blocks of `0..len`.
fn block(len: usize, i: usize, n: usize) -> Range<usize> {
    i * len / n..(i + 1) * len / n
}

/// A message between stages: `[sequence, position, wants logits]`, then
/// the hidden state; or `[sequence, token]` from the last stage back to
/// the first; or `[-1]` to stop.
const STOP: f32 = -1.0;

/// Greedy decoding of `new` tokens after each prompt, over `stages`
/// devices of one core each. Stage 0 feeds the sequences' tokens in turn
/// (prompt tokens as fast as it can, a generated token once the last stage
/// has sent it back), so that several sequences can be in the pipeline at
/// once.
pub fn generate<W: Matrix>(
    model: &Model<W>,
    stages: usize,
    prompts: &[Vec<u32>],
    new: usize,
) -> PipelineRun {
    let c = &model.config;
    let h = c.hidden_size;
    let context = prompts.iter().map(Vec::len).max().unwrap_or(0) + new;
    let start = Instant::now();
    let results = run(stages, |comm: &Comm| {
        let s = comm.rank();
        let layers = block(c.num_layers, s, stages);
        let last = stages - 1;
        let mut pool = SpinPool::new(1);
        let mut buf = Buffers::new(model, context);
        let mut caches: Vec<KvCache> = prompts.iter().map(|_| KvCache::new(c, context)).collect();
        let mut busy = Duration::ZERO;
        let mut x = vec![0.0; h];
        // Runs this stage's layers for one token; returns the next token if
        // this is the last stage and logits are wanted.
        let mut work = |seq: usize, pos: usize, want: bool, x: &mut [f32], busy: &mut Duration| {
            let t = Instant::now();
            run_layers(
                model,
                layers.clone(),
                &mut pool,
                x,
                pos,
                &mut caches[seq],
                &mut buf,
            );
            caches[seq].advance(1);
            let token = (s == last && want).then(|| next_token(model, &mut pool, x, &mut buf));
            *busy += t.elapsed();
            token
        };
        if s > 0 {
            // A middle or last stage: process what arrives, in order.
            loop {
                let msg = comm.recv(s - 1);
                if msg[0] == STOP {
                    if s < last {
                        comm.send(s + 1, &[STOP]);
                    }
                    break;
                }
                let (seq, pos, want) = (msg[0] as usize, msg[1] as usize, msg[2] > 0.0);
                x.copy_from_slice(&msg[3..]);
                match work(seq, pos, want, &mut x, &mut busy) {
                    Some(token) => comm.send(0, &[seq as f32, token as f32]),
                    None if s < last => {
                        let mut out = vec![seq as f32, pos as f32, msg[2]];
                        out.extend_from_slice(&x);
                        comm.send(s + 1, &out);
                    }
                    None => {}
                }
            }
            return (Vec::new(), Vec::new(), busy, comm.stats().bytes);
        }
        // Stage 0 drives: each sequence's tokens so far, and how many of
        // them have been fed into the pipeline. A sequence whose tokens have
        // all been fed is waiting for its next token to come back.
        let mut tokens: Vec<Vec<u32>> = prompts.to_vec();
        let mut fed = vec![0; prompts.len()];
        let mut arrivals = vec![Vec::new(); prompts.len()];
        let done = |tokens: &[Vec<u32>]| {
            tokens
                .iter()
                .zip(prompts)
                .all(|(t, p)| t.len() == p.len() + new)
        };
        let mut turn = 0;
        while !done(&tokens) {
            // The next sequence (round robin) with a token ready to feed.
            let ready = (0..prompts.len())
                .map(|i| (turn + i) % prompts.len())
                .find(|&i| fed[i] < tokens[i].len() && tokens[i].len() < prompts[i].len() + new);
            let Some(seq) = ready else {
                // Nothing to feed: wait for a token to come back.
                let msg = comm.recv(last);
                tokens[msg[0] as usize].push(msg[1] as u32);
                arrivals[msg[0] as usize].push(start.elapsed());
                continue;
            };
            turn = seq + 1;
            let pos = fed[seq];
            fed[seq] += 1;
            let want = fed[seq] == tokens[seq].len();
            model.embed.row_to_f32(tokens[seq][pos] as usize, &mut x);
            if let Some(token) = work(seq, pos, want, &mut x, &mut busy) {
                tokens[seq].push(token); // a single stage does everything
                arrivals[seq].push(start.elapsed());
            } else if stages > 1 {
                let mut out = vec![seq as f32, pos as f32, f32::from(u8::from(want))];
                out.extend_from_slice(&x);
                comm.send(1, &out);
            }
        }
        if stages > 1 {
            comm.send(1, &[STOP]);
        }
        let outputs = tokens
            .iter()
            .zip(prompts)
            .map(|(t, p)| t[p.len()..].to_vec())
            .collect();
        (outputs, arrivals, busy, comm.stats().bytes)
    });
    let elapsed = start.elapsed();
    let mut results = results.into_iter();
    let (outputs, arrivals, busy0, bytes0) = results.next().expect("stage 0");
    let mut busy = vec![busy0];
    let mut bytes = vec![bytes0];
    for (_, _, b, n) in results {
        busy.push(b);
        bytes.push(n);
    }
    PipelineRun {
        outputs,
        arrivals,
        elapsed,
        busy,
        bytes,
    }
}
