//! Attention over the paged cache.
//!
//! The same online softmax as chapter 20, over tiles that are now the
//! sequence's blocks: each block's keys for one (layer, head) are
//! contiguous, but consecutive blocks can be anywhere in the pool. The
//! block table says where; `attend_tile` (chapter 20) does the arithmetic,
//! with the same AVX-512 kernels.

use crate::blocks::{BlockPool, BlockTable};
use ch07_threads::SpinPool;
use ch14_kv_cache::Config;
use ch20_flash_attention::{FlashOptions, attend_tile, merge_parts, reset_state};

/// One sequence's attention input for one layer: queries for positions
/// `start..start + m`, whose keys and values are already stored.
pub struct PagedInput<'a> {
    pub table: &'a BlockTable,
    pub start: usize,
    pub m: usize,
    /// `[m × q_dim]`.
    pub q: &'a [f32],
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// Decode attention for many sequences, one new token each (`m == 1`), in
/// one parallel pass. `out` gets one `q_dim` row per input; `states` is
/// reused from call to call.
#[expect(
    clippy::too_many_arguments,
    reason = "the pieces of one attention call, kept explicit"
)]
pub fn paged_decode_many(
    pool: &mut SpinPool,
    kv: &BlockPool,
    config: &Config,
    layer: usize,
    inputs: &[PagedInput<'_>],
    out: &mut [f32],
    opts: &FlashOptions,
    states: &mut Vec<f32>,
) {
    if inputs.is_empty() {
        return;
    }
    let (d, q_dim, bs) = (config.head_dim, config.q_dim(), kv.block_size());
    let (kv_heads, group) = (config.num_kv_heads, config.num_heads / config.num_kv_heads);
    assert!(inputs.iter().all(|i| i.m == 1), "one token per sequence");
    assert_eq!(out.len(), inputs.len() * q_dim, "one output row per input");
    // As in chapter 20: split a sequence's blocks into parts only when there
    // are too few (sequence, KV head) tasks for every thread to get the
    // same number, and never into more parts than the shortest has blocks.
    let base = inputs.len() * kv_heads;
    let threads = pool.threads();
    let shortest = inputs
        .iter()
        .map(|i| (i.start + 1).div_ceil(bs))
        .min()
        .unwrap_or(1);
    let splits = (threads / gcd(base, threads)).min(shortest).max(1);
    let stride = d + 2;
    let per_task = group * stride;
    states.clear();
    states.resize(base * splits * per_task, 0.0);
    pool.for_each_chunk_mut(states, per_task, |first, mine| {
        for (i, state) in mine.chunks_exact_mut(per_task).enumerate() {
            let id = first / per_task + i;
            let (seq, rest) = (id / (kv_heads * splits), id % (kv_heads * splits));
            let (kv_head, part) = (rest / splits, rest % splits);
            let input = &inputs[seq];
            let keys = input.start + 1;
            let blocks = keys.div_ceil(bs);
            let per_part = blocks.div_ceil(splits);
            for s in state.chunks_exact_mut(stride) {
                reset_state(s);
            }
            for b in (part * per_part).min(blocks)..((part + 1) * per_part).min(blocks) {
                let n = (keys - b * bs).min(bs);
                let block = input.table.blocks[b];
                let k = kv.keys(block, layer, kv_head, n);
                let v = kv.values(block, layer, kv_head, n);
                for (g, s) in state.chunks_exact_mut(stride).enumerate() {
                    let head = kv_head * group + g;
                    attend_tile(&input.q[head * d..(head + 1) * d], k, v, s, opts.simd);
                }
            }
        }
    });
    for (seq, row) in out.chunks_exact_mut(q_dim).enumerate() {
        for kv_head in 0..kv_heads {
            let first = (seq * kv_heads + kv_head) * splits;
            for g in 0..group {
                let head = kv_head * group + g;
                let o = &mut row[head * d..(head + 1) * d];
                merge_parts(states, first, splits, per_task, g * stride, o);
            }
        }
    }
}

/// Causal attention for a chunk of `m` tokens of one sequence (a prompt
/// being prefilled). Tasks are (KV head, block of query tokens); each
/// walks the sequence's blocks once for all its tokens.
#[expect(
    clippy::too_many_arguments,
    reason = "the pieces of one attention call, kept explicit"
)]
pub fn paged_chunk(
    pool: &mut SpinPool,
    kv: &BlockPool,
    config: &Config,
    layer: usize,
    input: &PagedInput<'_>,
    out: &mut [f32],
    opts: &FlashOptions,
    states: &mut Vec<f32>,
) {
    let (d, q_dim, bs) = (config.head_dim, config.q_dim(), kv.block_size());
    let (kv_heads, group) = (config.num_kv_heads, config.num_heads / config.num_kv_heads);
    let (m, start) = (input.m, input.start);
    assert_eq!(out.len(), m * q_dim, "one output row per token");
    let qb = opts.query_block.clamp(1, m);
    let q_blocks = m.div_ceil(qb);
    let stride = d + 2;
    let per_task = qb * group * stride;
    states.clear();
    states.resize(kv_heads * q_blocks * per_task, 0.0);
    pool.for_each_chunk_mut(states, per_task, |first, mine| {
        for (i, state) in mine.chunks_exact_mut(per_task).enumerate() {
            let id = first / per_task + i;
            let (kv_head, qblock) = (id / q_blocks, id % q_blocks);
            let (t0, t1) = (qblock * qb, ((qblock + 1) * qb).min(m));
            for s in state.chunks_exact_mut(stride) {
                reset_state(s);
            }
            // The block's last token sees the most keys.
            let last_keys = start + t1;
            for b in 0..last_keys.div_ceil(bs) {
                let block = input.table.blocks[b];
                for t in t0..t1 {
                    // Causal: token t sees positions up to start + t.
                    let visible = start + t + 1;
                    if visible <= b * bs {
                        continue;
                    }
                    let n = (visible - b * bs).min(bs);
                    let k = kv.keys(block, layer, kv_head, n);
                    let v = kv.values(block, layer, kv_head, n);
                    for g in 0..group {
                        let head = kv_head * group + g;
                        let q = &input.q[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                        let s = &mut state[((t - t0) * group + g) * stride..][..stride];
                        attend_tile(q, k, v, s, opts.simd);
                    }
                }
            }
        }
    });
    for kv_head in 0..kv_heads {
        for qblock in 0..q_blocks {
            let task = kv_head * q_blocks + qblock;
            for t in qblock * qb..((qblock + 1) * qb).min(m) {
                for g in 0..group {
                    let head = kv_head * group + g;
                    let at = ((t - qblock * qb) * group + g) * stride;
                    let o = &mut out[t * q_dim + head * d..t * q_dim + (head + 1) * d];
                    // One part: this only normalizes.
                    merge_parts(states, task, 1, per_task, at, o);
                }
            }
        }
    }
}
