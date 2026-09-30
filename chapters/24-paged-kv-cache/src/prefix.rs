//! Prefix caching: reusing the KV blocks of earlier requests that started
//! with the same tokens.
//!
//! Many requests share a beginning: the chat template, a system prompt,
//! the documents of a retrieval pipeline, the earlier turns of a
//! conversation. Their keys and values for those positions are identical,
//! because a position's keys and values depend only on the tokens up to it.
//! So a full block computed for one request can serve every later request
//! whose tokens up to the end of that block are the same.
//!
//! A block is found by a hash chained over the blocks before it: block `i`
//! is keyed by `hash(key of block i − 1, tokens of block i)`, so equal keys
//! mean equal tokens from the very start. The entry also keeps the block's
//! tokens and its parent's key, and a lookup compares them: a hash
//! collision can cost a cache miss, never a wrong answer.

use crate::blocks::{BlockPool, BlockTable};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// The key of the block holding `tokens`, following the block keyed `parent`
/// (0 for the first block).
pub fn block_key(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    parent.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

/// The keys of every full block of `tokens`, in order.
pub fn block_keys(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let mut parent = 0;
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            parent = block_key(parent, block);
            parent
        })
        .collect()
}

struct Entry {
    block: u32,
    parent: u64,
    tokens: Vec<u32>,
    /// Position of the block in its chain (0 for a sequence's first).
    depth: usize,
    /// For eviction: when the entry was last used.
    last_used: u64,
}

/// Full blocks kept after their requests, found by their keys.
///
/// The cache owns one reference to every block it keeps (chapter's
/// `BlockPool` counts it). A block used by no request, only by the cache,
/// can be evicted when memory is needed.
#[derive(Default)]
pub struct PrefixCache {
    entries: HashMap<u64, Entry>,
    clock: u64,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Blocks kept.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The longest run of cached blocks at the start of `tokens`, at most
    /// `max_blocks`. Each block returned has been retained for the caller.
    pub fn lookup(&mut self, pool: &mut BlockPool, tokens: &[u32], max_blocks: usize) -> Vec<u32> {
        self.clock += 1;
        let mut found = Vec::new();
        let mut parent = 0;
        for block_tokens in tokens.chunks_exact(pool.block_size()).take(max_blocks) {
            let key = block_key(parent, block_tokens);
            let Some(e) = self.entries.get_mut(&key) else {
                break;
            };
            if e.parent != parent || e.tokens != block_tokens {
                break; // a hash collision: treat as a miss
            }
            e.last_used = self.clock;
            pool.retain(e.block);
            found.push(e.block);
            parent = key;
        }
        found
    }

    /// Records the full blocks of a sequence whose first `table.len`
    /// positions hold the keys and values of `tokens[..table.len]`. Blocks
    /// already cached are left alone; new ones are retained by the cache.
    pub fn insert(&mut self, pool: &mut BlockPool, tokens: &[u32], table: &BlockTable) {
        self.clock += 1;
        let bs = pool.block_size();
        let mut parent = 0;
        for (i, block_tokens) in tokens[..table.len].chunks_exact(bs).enumerate() {
            let key = block_key(parent, block_tokens);
            let block = table.blocks[i];
            let clock = self.clock;
            let e = self.entries.entry(key).or_insert_with(|| {
                pool.retain(block);
                Entry {
                    block,
                    parent,
                    tokens: block_tokens.to_vec(),
                    depth: i,
                    last_used: clock,
                }
            });
            e.last_used = clock;
            parent = key;
        }
    }

    /// Frees up to `wanted` blocks that only the cache holds, least
    /// recently used first; among equally recent ones, the end of a chain
    /// before its beginning (a block is only reachable through its
    /// parents, so evicting a parent first would strand its children).
    /// Returns how many were freed.
    pub fn evict(&mut self, pool: &mut BlockPool, wanted: usize) -> usize {
        let mut idle: Vec<(u64, std::cmp::Reverse<usize>, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| pool.refs(e.block) == 1)
            .map(|(&key, e)| (e.last_used, std::cmp::Reverse(e.depth), key))
            .collect();
        idle.sort_unstable();
        let mut freed = 0;
        for (_, _, key) in idle.into_iter().take(wanted) {
            let e = self.entries.remove(&key).expect("listed above");
            pool.release(e.block);
            freed += 1;
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch14_kv_cache::Config;

    #[test]
    fn a_shared_prefix_is_found_block_by_block() {
        let mut pool = BlockPool::new(&Config::tiny(), 8, 4);
        let mut cache = PrefixCache::new();
        let first: Vec<u32> = (0..10).collect();
        let mut table = BlockTable::default();
        assert!(table.reserve(&mut pool, 10));
        table.len = 10;
        cache.insert(&mut pool, &first, &table);
        assert_eq!(cache.len(), 2, "two full blocks of 4");

        // Same first 6 tokens: only the first block matches.
        let second = [0, 1, 2, 3, 4, 5, 99, 99, 99];
        assert_eq!(cache.lookup(&mut pool, &second, 9), vec![table.blocks[0]]);
        // Same first 8 tokens: both blocks, unless capped.
        let third = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(cache.lookup(&mut pool, &third, 9).len(), 2);
        assert_eq!(cache.lookup(&mut pool, &third, 1).len(), 1);
        // A different first token: nothing, even with equal later blocks.
        let fourth = [9, 1, 2, 3, 4, 5, 6, 7];
        assert!(cache.lookup(&mut pool, &fourth, 9).is_empty());
    }

    #[test]
    fn only_blocks_nobody_uses_are_evicted_oldest_first() {
        let mut pool = BlockPool::new(&Config::tiny(), 4, 2);
        let mut cache = PrefixCache::new();
        let tokens = [1, 2, 3, 4];
        let mut table = BlockTable::default();
        assert!(table.reserve(&mut pool, 4));
        table.len = 4;
        cache.insert(&mut pool, &tokens, &table);
        // The sequence still holds both blocks: nothing can be evicted.
        assert_eq!(cache.evict(&mut pool, 2), 0);
        table.release(&mut pool);
        assert_eq!(pool.free_blocks(), 2, "the cache keeps the other two");
        assert_eq!(cache.evict(&mut pool, 1), 1);
        assert_eq!((cache.len(), pool.free_blocks()), (1, 3));
        // The chain's second block went first: the first is still found.
        assert_eq!(cache.lookup(&mut pool, &tokens, 2).len(), 1);
    }
}
