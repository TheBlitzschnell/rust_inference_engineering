//! Mixture of experts, and expert parallelism.
//!
//! A mixture-of-experts (MoE) layer replaces one MLP by many ("experts")
//! and a router: for each token, the router scores every expert, and only
//! the `top_k` best run on that token, their outputs weighted by the
//! router's softmax over those `k` scores. The model has the parameters of
//! all experts but each token pays for `k` of them.
//!
//! With expert parallelism, device `r` holds experts `r·E/n ..`; tokens are
//! sent to the devices holding their experts (an all-to-all), processed,
//! and sent back (a second all-to-all). An expert can accept at most a
//! fixed number of tokens per batch (its capacity); tokens beyond it are
//! dropped by that expert, as fixed-size buffers on accelerators require.

use crate::comm::Comm;
use ch07_threads::SpinPool;
use ch08_operators::swiglu;
use ch14_kv_cache::{DenseF32, Matrix};
use ch15_sampling::Rng;
use std::ops::Range;

/// One expert: a SwiGLU MLP, like a layer's.
pub struct Expert {
    pub gate: DenseF32,
    pub up: DenseF32,
    pub down: DenseF32,
}

/// The experts of one MoE layer, and its router.
pub struct Moe {
    pub hidden: usize,
    pub router: DenseF32,
    pub experts: Vec<Expert>,
    pub top_k: usize,
}

/// Normal-ish random values (a sum of uniforms), scaled.
fn random(n: usize, scale: f32, rng: &mut Rng) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let s: f64 = (0..4).map(|_| rng.next_f64() - 0.5).sum();
            s as f32 * scale
        })
        .collect()
}

/// For one token: the chosen experts, with their weights.
pub type Route = Vec<(usize, f32)>;

impl Moe {
    /// A layer with random weights: `experts` SwiGLU experts of width
    /// `ffn` over `hidden` features, each token using `top_k` of them.
    pub fn random(hidden: usize, ffn: usize, experts: usize, top_k: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let w = |rows: usize, cols: usize, rng: &mut Rng| {
            DenseF32::new(
                &random(rows * cols, 1.0 / (cols as f32).sqrt(), rng),
                rows,
                cols,
            )
        };
        Self {
            hidden,
            router: w(experts, hidden, &mut rng),
            experts: (0..experts)
                .map(|_| Expert {
                    gate: w(ffn, hidden, &mut rng),
                    up: w(ffn, hidden, &mut rng),
                    down: w(hidden, ffn, &mut rng),
                })
                .collect(),
            top_k,
        }
    }

    /// The `top_k` experts of each of the `m` tokens in `x`, with the
    /// softmax of their router scores.
    pub fn route(&self, pool: &mut SpinPool, x: &[f32], m: usize) -> Vec<Route> {
        let e = self.experts.len();
        let mut scores = vec![0.0; m * e];
        self.router.matmul(pool, x, &mut scores, m, &mut Vec::new());
        scores
            .chunks_exact(e)
            .map(|row| {
                let mut order: Vec<usize> = (0..e).collect();
                order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                let best = &order[..self.top_k];
                let max = row[best[0]];
                let w: Vec<f32> = best.iter().map(|&i| (row[i] - max).exp()).collect();
                let sum: f32 = w.iter().sum();
                best.iter().zip(w).map(|(&i, w)| (i, w / sum)).collect()
            })
            .collect()
    }

    /// Expert `e` applied to `m` rows of `x`.
    pub fn expert(&self, e: usize, pool: &mut SpinPool, x: &[f32], m: usize) -> Vec<f32> {
        let ex = &self.experts[e];
        let ffn = ex.gate.rows();
        let mut scratch = Vec::new();
        let (mut gate, mut up, mut act) =
            (vec![0.0; m * ffn], vec![0.0; m * ffn], vec![0.0; m * ffn]);
        ex.gate.matmul(pool, x, &mut gate, m, &mut scratch);
        ex.up.matmul(pool, x, &mut up, m, &mut scratch);
        swiglu(&gate, &up, &mut act);
        let mut y = vec![0.0; m * self.hidden];
        ex.down.matmul(pool, &act, &mut y, m, &mut scratch);
        y
    }

    /// The layer on one device: every token, every expert. `capacity`
    /// limits the tokens each expert takes (in token order); the others
    /// are dropped by it. Returns the output and the tokens each expert
    /// processed.
    pub fn forward(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        m: usize,
        capacity: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let h = self.hidden;
        let routes = self.route(pool, x, m);
        let mut y = vec![0.0; m * h];
        let mut load = Vec::new();
        for e in 0..self.experts.len() {
            // This expert's (token, weight) pairs, in token order.
            let mine: Vec<(usize, f32)> = routes
                .iter()
                .enumerate()
                .filter_map(|(t, r)| r.iter().find(|&&(ex, _)| ex == e).map(|&(_, w)| (t, w)))
                .take(capacity)
                .collect();
            load.push(mine.len());
            let rows: Vec<f32> = mine
                .iter()
                .flat_map(|&(t, _)| x[t * h..(t + 1) * h].iter().copied())
                .collect();
            let out = self.expert(e, pool, &rows, mine.len());
            for (&(t, w), o) in mine.iter().zip(out.chunks_exact(h)) {
                for (yi, oi) in y[t * h..(t + 1) * h].iter_mut().zip(o) {
                    *yi += w * oi;
                }
            }
        }
        (y, load)
    }
}

/// The `i`-th of `n` contiguous blocks of `0..len`.
pub fn block(len: usize, i: usize, n: usize) -> Range<usize> {
    i * len / n..(i + 1) * len / n
}

/// What one device did in an expert-parallel layer.
#[derive(Clone, Debug, Default)]
pub struct EpStats {
    /// Tokens processed by each of this device's experts.
    pub load: Vec<usize>,
    /// (token, expert) pairs dropped by this device's experts.
    pub dropped: usize,
    /// Time computing this device's experts.
    pub expert_time: std::time::Duration,
}

/// Expert parallelism over `comm.size()` devices. Device `r` holds the
/// experts `block(E, r, n)` and the tokens `block(m, r, n)` of the batch
/// (`x_local`, its rows); it returns those tokens' outputs.
pub fn forward_ep(
    moe: &Moe,
    comm: &Comm,
    pool: &mut SpinPool,
    x_local: &[f32],
    capacity: usize,
) -> (Vec<f32>, EpStats) {
    let (h, n, experts) = (moe.hidden, comm.size(), moe.experts.len());
    let m = x_local.len() / h;
    let owner = |e: usize| {
        (0..n)
            .find(|&r| block(experts, r, n).contains(&e))
            .expect("an owner")
    };
    // 1. Route the local tokens and send each (token, expert) pair to the
    //    expert's device: [expert, x...] per pair, in token order.
    let routes = moe.route(pool, x_local, m);
    let mut outgoing = vec![Vec::new(); n];
    let mut sent: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for (t, route) in routes.iter().enumerate() {
        for &(e, w) in route {
            let d = owner(e);
            outgoing[d].push(e as f32);
            outgoing[d].extend_from_slice(&x_local[t * h..(t + 1) * h]);
            sent[d].push((t, w));
        }
    }
    let incoming = comm.all_to_all(&outgoing);

    // 2. Run each local expert on the pairs it received, in the order of
    //    the senders' ranks (that is, global token order), up to capacity.
    let mut stats = EpStats::default();
    let pairs: Vec<Vec<usize>> = incoming
        .iter()
        .map(|msg| msg.chunks_exact(h + 1).map(|p| p[0] as usize).collect())
        .collect();
    let mut results: Vec<Vec<f32>> = incoming.iter().map(|msg| vec![0.0; msg.len()]).collect();
    let start = std::time::Instant::now();
    for e in block(experts, comm.rank(), n) {
        let mut rows = Vec::new();
        let mut at = Vec::new(); // (sender, index of the pair)
        for (src, list) in pairs.iter().enumerate() {
            for (i, &ex) in list.iter().enumerate() {
                if ex == e {
                    if at.len() < capacity {
                        rows.extend_from_slice(&incoming[src][i * (h + 1) + 1..(i + 1) * (h + 1)]);
                        at.push((src, i));
                    } else {
                        stats.dropped += 1;
                    }
                }
            }
        }
        stats.load.push(at.len());
        let out = moe.expert(e, pool, &rows, at.len());
        for (&(src, i), o) in at.iter().zip(out.chunks_exact(h)) {
            // [kept, y...]: a dropped pair's slot stays [0, 0...].
            results[src][i * (h + 1)] = 1.0;
            results[src][i * (h + 1) + 1..(i + 1) * (h + 1)].copy_from_slice(o);
        }
    }
    stats.expert_time = start.elapsed();

    // 3. Send the results back and add them up, weighted.
    let returned = comm.all_to_all(&results);
    let mut y = vec![0.0; m * h];
    for (d, msg) in returned.iter().enumerate() {
        for (&(t, w), r) in sent[d].iter().zip(msg.chunks_exact(h + 1)) {
            if r[0] > 0.0 {
                for (yi, ri) in y[t * h..(t + 1) * h].iter_mut().zip(&r[1..]) {
                    *yi += w * ri;
                }
            }
        }
    }
    (y, stats)
}
