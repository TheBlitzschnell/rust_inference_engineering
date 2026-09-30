//! FlashAttention-style attention against chapter 14's KV cache.
//!
//! Work is split into tasks. A task is one KV head, one block of query
//! tokens and one part of the key range:
//!
//! - **one KV head, all its query heads**: with GQA, the 3 query heads that
//!   share a KV head read the same keys and values, so they are processed
//!   together and each key tile is loaded once for all three;
//! - **a block of query tokens**: during prefill, a tile of keys is used by
//!   every token of the block while it is in the cache;
//! - **part of the keys (split-KV)**: during decode there is one query
//!   token, and only 3 KV heads, too few tasks for the threads. Splitting
//!   the context into parts gives each thread a share; the partial results
//!   are merged afterwards, exactly (see [`crate::online`]).
//!
//! Within a task, each (token, head) keeps an online-softmax state and the
//! keys are visited once, in tiles.

#![expect(
    clippy::inline_always,
    reason = "multiversioning: the generic task body must be inlined into the #[target_feature] function to be compiled for AVX-512"
)]

use ch06_simd::dot;
use ch07_threads::SpinPool;
use ch08_operators::exp_fast;
use ch14_kv_cache::AttentionInput;

/// Tuning knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashOptions {
    /// Query tokens per task.
    pub query_block: usize,
    /// Keys per tile (at most [`MAX_TILE`]).
    pub key_block: usize,
    /// Parts each task's key range is split into; 0 chooses enough parts
    /// to give every thread the same number of tasks.
    pub key_splits: usize,
    /// Use the AVX-512 kernels when available (heads of 64 dimensions).
    /// `false` measures what the loop structure alone is worth.
    pub simd: bool,
}

impl Default for FlashOptions {
    fn default() -> Self {
        Self {
            query_block: 16,
            key_block: 64,
            key_splits: 0,
            simd: true,
        }
    }
}

/// The largest key tile (the scores of one tile live on the stack).
pub const MAX_TILE: usize = 256;

/// Where one task works.
#[derive(Clone, Copy, Debug)]
struct Task {
    kv_head: usize,
    /// Query tokens `t0..t1` of the chunk.
    t0: usize,
    t1: usize,
    /// Keys `k0..k1`.
    k0: usize,
    k1: usize,
}

/// Causal attention for the chunk in `input`, written to `out`
/// (`[m × q_dim]`).
pub fn flash_attention(
    pool: &mut SpinPool,
    input: &AttentionInput<'_>,
    out: &mut [f32],
    opts: &FlashOptions,
) {
    let c = input.config;
    let (d, q_dim, m, start) = (c.head_dim, c.q_dim(), input.m, input.start);
    let group = c.num_heads / c.num_kv_heads;
    assert!(
        opts.key_block > 0 && opts.key_block <= MAX_TILE,
        "bad key block"
    );
    // No bigger than the chunk: a decode step has one query token.
    let qb = opts.query_block.clamp(1, m.max(1));
    let blocks = m.div_ceil(qb);
    let splits = if opts.key_splits > 0 {
        opts.key_splits
    } else {
        // `for_each_chunk_mut` gives every thread the same number of tasks
        // only if the task count is a multiple of the thread count. Split
        // the keys just enough for that (4 parts for decode's 3 KV heads
        // on 4 threads), but never into parts shorter than a tile.
        let base = c.num_kv_heads * blocks;
        let threads = pool.threads();
        let wanted = threads / gcd(base, threads);
        wanted.min((start + m).div_ceil(opts.key_block)).max(1)
    };
    // One state per (token, head of the group): max, sum, then d values.
    let stride = d + 2;
    let per_task = qb * group * stride;
    let tasks = c.num_kv_heads * blocks * splits;
    let task_of = |id: usize| {
        let s = id % splits;
        let b = (id / splits) % blocks;
        let kv_head = id / (splits * blocks);
        let (t0, t1) = (b * qb, ((b + 1) * qb).min(m));
        // Keys visible to the block's last token, split evenly.
        let keys = start + t1;
        let part = keys.div_ceil(splits);
        Task {
            kv_head,
            t0,
            t1,
            k0: (s * part).min(keys),
            k1: ((s + 1) * part).min(keys),
        }
    };

    let mut states = vec![0.0f32; tasks * per_task];
    pool.for_each_chunk_mut(&mut states, per_task, |first, mine| {
        for (i, state) in mine.chunks_exact_mut(per_task).enumerate() {
            run_task(input, task_of(first / per_task + i), group, state, opts);
        }
    });

    // Merge the parts of each (KV head, block) and write the outputs.
    for kv_head in 0..c.num_kv_heads {
        for b in 0..blocks {
            let first = (kv_head * blocks + b) * splits;
            let task = task_of(first);
            for t in task.t0..task.t1 {
                for g in 0..group {
                    let at = ((t - task.t0) * group + g) * stride;
                    let head = kv_head * group + g;
                    let o = &mut out[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                    merge_parts(&states, first, splits, per_task, at, o);
                }
            }
        }
    }
}

/// Decode attention for many sequences at once, each with one new token
/// (chapter 23's batches): `out` receives one `q_dim` row per input, in
/// order. All sequences' tasks go to the pool in one parallel pass, and
/// the per-task states live in `states`, reused from call to call.
pub fn flash_decode_many(
    pool: &mut SpinPool,
    inputs: &[AttentionInput<'_>],
    out: &mut [f32],
    opts: &FlashOptions,
    states: &mut Vec<f32>,
) {
    let Some(first) = inputs.first() else {
        return;
    };
    let c = first.config;
    let (d, q_dim) = (c.head_dim, c.q_dim());
    let group = c.num_heads / c.num_kv_heads;
    assert!(inputs.iter().all(|i| i.m == 1), "one token per sequence");
    assert_eq!(out.len(), inputs.len() * q_dim, "one output row per input");
    // Split keys only if there are too few (sequence, KV head) tasks to
    // give every thread the same number: the same rule as above.
    let base = inputs.len() * c.num_kv_heads;
    let threads = pool.threads();
    let shortest = inputs.iter().map(|i| i.start + 1).min().unwrap_or(1);
    let splits = (threads / gcd(base, threads))
        .min(shortest.div_ceil(opts.key_block))
        .max(1);
    let stride = d + 2;
    let per_task = group * stride;
    let task_of = |id: usize| {
        let (seq, rest) = (
            id / (c.num_kv_heads * splits),
            id % (c.num_kv_heads * splits),
        );
        let (kv_head, s) = (rest / splits, rest % splits);
        let keys = inputs[seq].start + 1;
        let part = keys.div_ceil(splits);
        (
            seq,
            Task {
                kv_head,
                t0: 0,
                t1: 1,
                k0: (s * part).min(keys),
                k1: ((s + 1) * part).min(keys),
            },
        )
    };
    states.clear();
    states.resize(base * splits * per_task, 0.0);
    pool.for_each_chunk_mut(states, per_task, |first, mine| {
        for (i, state) in mine.chunks_exact_mut(per_task).enumerate() {
            let (seq, task) = task_of(first / per_task + i);
            run_task(&inputs[seq], task, group, state, opts);
        }
    });
    // Merge each (sequence, head)'s parts, as in `flash_attention`.
    for (seq, row) in out.chunks_exact_mut(q_dim).enumerate() {
        for kv_head in 0..c.num_kv_heads {
            let first = (seq * c.num_kv_heads + kv_head) * splits;
            for g in 0..group {
                let head = kv_head * group + g;
                let o = &mut row[head * d..(head + 1) * d];
                merge_parts(states, first, splits, per_task, g * stride, o);
            }
        }
    }
}

/// Merges `splits` partial states (tasks `first..first + splits`, each
/// state at offset `at` within its task) into `o`, normalized.
fn merge_parts(
    states: &[f32],
    first: usize,
    splits: usize,
    per_task: usize,
    at: usize,
    o: &mut [f32],
) {
    let stride = o.len() + 2;
    let (mut mx, mut sum) = (f32::NEG_INFINITY, 0.0f32);
    o.fill(0.0);
    for s in 0..splits {
        let part = &states[(first + s) * per_task + at..][..stride];
        let (pm, ps, pacc) = (part[0], part[1], &part[2..]);
        if pm == f32::NEG_INFINITY {
            continue;
        }
        let new = mx.max(pm);
        let (a, b) = (exp_fast(mx - new), exp_fast(pm - new));
        sum = sum * a + ps * b;
        for (x, &y) in o.iter_mut().zip(pacc) {
            *x = *x * a + y * b;
        }
        mx = new;
    }
    let inv = 1.0 / sum;
    for x in o.iter_mut() {
        *x *= inv;
    }
}

/// Runs one task, with AVX-512 kernels when the CPU has them and heads have
/// 64 dimensions (SmolLM2, Llama and most models), portable ones otherwise.
fn run_task(
    input: &AttentionInput<'_>,
    task: Task,
    group: usize,
    state: &mut [f32],
    opts: &FlashOptions,
) {
    #[cfg(target_arch = "x86_64")]
    if opts.simd && input.config.head_dim == 64 && std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512F was just detected.
        unsafe { x86::run_task_avx512(input, task, group, state, opts.key_block) };
        return;
    }
    run_task_with::<Portable>(input, task, group, state, opts.key_block);
}

/// The two inner operations of attention, per implementation.
trait Kernels {
    /// `q · k`.
    fn dot(q: &[f32], k: &[f32]) -> f32;
    /// `acc = acc · rescale`, then `acc += Σ p_j · v_j` over the tile.
    fn accumulate(acc: &mut [f32], rescale: f32, p: &[f32], values: &[f32]);
}

struct Portable;

impl Kernels for Portable {
    #[inline(always)]
    fn dot(q: &[f32], k: &[f32]) -> f32 {
        dot(q, k)
    }

    #[inline(always)]
    fn accumulate(acc: &mut [f32], rescale: f32, p: &[f32], values: &[f32]) {
        for a in acc.iter_mut() {
            *a *= rescale;
        }
        for (&pj, v) in p.iter().zip(values.chunks_exact(acc.len())) {
            for (a, &x) in acc.iter_mut().zip(v) {
                *a += pj * x;
            }
        }
    }
}

/// Every query token of the task's block, every query head of its KV
/// head, over its key range in tiles. Generic over the kernels, and
/// inlined into each caller, so that each instantiation is compiled for
/// its caller's instruction set.
#[inline(always)]
fn run_task_with<K: Kernels>(
    input: &AttentionInput<'_>,
    task: Task,
    group: usize,
    state: &mut [f32],
    key_block: usize,
) {
    let c = input.config;
    let (d, q_dim, start) = (c.head_dim, c.q_dim(), input.start);
    let stride = d + 2;
    let scale = 1.0 / (d as f32).sqrt();
    for s in state.chunks_exact_mut(stride) {
        s[0] = f32::NEG_INFINITY;
        s[1] = 0.0;
        s[2..].fill(0.0);
    }
    let keys = input.cache.keys(input.layer, task.kv_head, task.k1);
    let values = input.cache.values(input.layer, task.kv_head, task.k1);
    let mut scores = [0.0f32; MAX_TILE];
    let mut k = task.k0;
    while k < task.k1 {
        let tile_end = (k + key_block).min(task.k1);
        for t in task.t0..task.t1 {
            // Causal: token t sees positions up to start + t.
            let end = tile_end.min(start + t + 1);
            if end <= k {
                continue;
            }
            let n = end - k;
            for g in 0..group {
                let head = task.kv_head * group + g;
                let q = &input.q[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                let tile = &mut scores[..n];
                for (sc, key) in tile.iter_mut().zip(keys[k * d..end * d].chunks_exact(d)) {
                    *sc = K::dot(q, key) * scale;
                }
                let st = &mut state[((t - task.t0) * group + g) * stride..][..stride];
                online_update::<K>(st, tile, &values[k * d..end * d]);
            }
        }
        k = tile_end;
    }
}

/// One tile of scores into an online-softmax state `[max, sum, acc...]`.
/// The scores are overwritten with their weights `exp(s − max)`.
#[inline(always)]
fn online_update<K: Kernels>(state: &mut [f32], scores: &mut [f32], values: &[f32]) {
    let tile_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let new_max = state[0].max(tile_max);
    let rescale = exp_fast(state[0] - new_max);
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = exp_fast(*s - new_max);
        sum += *s;
    }
    let (head, acc) = state.split_at_mut(2);
    head[1] = head[1] * rescale + sum;
    K::accumulate(acc, rescale, scores, values);
    head[0] = new_max;
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! The same task, compiled for AVX-512 with kernels for 64-dimension
    //! heads: a head is exactly four 16-lane vectors.

    use super::{Kernels, Task, run_task_with};
    use ch14_kv_cache::AttentionInput;
    use std::arch::x86_64::{
        __m512, _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps,
        _mm512_reduce_add_ps, _mm512_set1_ps, _mm512_storeu_ps,
    };

    pub(super) struct Avx512D64;

    impl Kernels for Avx512D64 {
        #[inline(always)]
        fn dot(q: &[f32], k: &[f32]) -> f32 {
            assert!(q.len() == 64 && k.len() == 64, "64-dimension heads only");
            // SAFETY: both slices hold 64 floats, 16 at 0, 16, 32, 48.
            // This function is only instantiated inside `run_task_avx512`.
            unsafe {
                let mut acc =
                    [_mm512_mul_ps(_mm512_loadu_ps(q.as_ptr()), _mm512_loadu_ps(k.as_ptr())); 2];
                acc[1] = _mm512_mul_ps(
                    _mm512_loadu_ps(q.as_ptr().add(16)),
                    _mm512_loadu_ps(k.as_ptr().add(16)),
                );
                acc[0] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(q.as_ptr().add(32)),
                    _mm512_loadu_ps(k.as_ptr().add(32)),
                    acc[0],
                );
                acc[1] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(q.as_ptr().add(48)),
                    _mm512_loadu_ps(k.as_ptr().add(48)),
                    acc[1],
                );
                _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]))
            }
        }

        #[inline(always)]
        fn accumulate(acc: &mut [f32], rescale: f32, p: &[f32], values: &[f32]) {
            assert!(
                acc.len() == 64 && values.len() == p.len() * 64,
                "64-dimension heads only"
            );
            // SAFETY: `acc` holds 64 floats and each value row 64; this
            // function is only instantiated inside `run_task_avx512`.
            unsafe {
                let r = _mm512_set1_ps(rescale);
                let mut a: [__m512; 4] = std::array::from_fn(|i| {
                    _mm512_mul_ps(_mm512_loadu_ps(acc.as_ptr().add(16 * i)), r)
                });
                for (j, &pj) in p.iter().enumerate() {
                    let w = _mm512_set1_ps(pj);
                    let v = values.as_ptr().add(64 * j);
                    for (i, ai) in a.iter_mut().enumerate() {
                        *ai = _mm512_fmadd_ps(w, _mm512_loadu_ps(v.add(16 * i)), *ai);
                    }
                }
                for (i, ai) in a.iter().enumerate() {
                    _mm512_storeu_ps(acc.as_mut_ptr().add(16 * i), *ai);
                }
            }
        }
    }

    /// # Safety
    ///
    /// The CPU must support AVX-512F, and heads must have 64 dimensions.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn run_task_avx512(
        input: &AttentionInput<'_>,
        task: Task,
        group: usize,
        state: &mut [f32],
        key_block: usize,
    ) {
        run_task_with::<Avx512D64>(input, task, group, state, key_block);
    }
}
