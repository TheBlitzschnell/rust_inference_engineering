//! The paged KV cache: one pool of fixed-size blocks shared by every
//! sequence, and a block table per sequence.
//!
//! Chapter 23 gave each request a cache for the whole context, used or not.
//! Here memory is handed out a block (`block_size` positions) at a time, as
//! a sequence grows, and returned when it ends: the same idea as virtual
//! memory pages, and the reason vLLM called it *paged* attention.

use ch14_kv_cache::Config;

/// Keys and values for `num_blocks` blocks of `block_size` positions.
///
/// Layout, for keys and values separately:
/// `[block][layer][kv_head][position in block][head_dim]`. The positions of
/// one (block, layer, head) are contiguous, so attention reads a block of
/// keys as one run of memory, like a tile in chapter 20.
pub struct BlockPool {
    k: Vec<f32>,
    v: Vec<f32>,
    block_size: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    /// How many owners each block has: sequences using it, plus the prefix
    /// cache if it keeps it. 0 means free.
    refs: Vec<u32>,
    free: Vec<u32>,
}

impl BlockPool {
    pub fn new(config: &Config, num_blocks: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "blocks need at least one position");
        let size = num_blocks * Self::floats_per_block(config, block_size);
        Self {
            k: vec![0.0; size],
            v: vec![0.0; size],
            block_size,
            layers: config.num_layers,
            kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            refs: vec![0; num_blocks],
            // Popped from the end: block 0 is handed out first.
            free: (0..num_blocks as u32).rev().collect(),
        }
    }

    fn floats_per_block(config: &Config, block_size: usize) -> usize {
        config.num_layers * config.num_kv_heads * block_size * config.head_dim
    }

    /// Bytes of keys and values in one block.
    pub fn bytes_per_block(config: &Config, block_size: usize) -> usize {
        2 * Self::floats_per_block(config, block_size) * size_of::<f32>()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.refs.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// A free block with one owner (the caller), or `None` if none is left.
    pub fn allocate(&mut self) -> Option<u32> {
        let b = self.free.pop()?;
        self.refs[b as usize] = 1;
        Some(b)
    }

    /// One more owner for block `b`.
    pub fn retain(&mut self, b: u32) {
        assert!(self.refs[b as usize] > 0, "retaining a free block");
        self.refs[b as usize] += 1;
    }

    /// One owner fewer; the last one frees the block.
    pub fn release(&mut self, b: u32) {
        let r = &mut self.refs[b as usize];
        assert!(*r > 0, "releasing a free block");
        *r -= 1;
        if *r == 0 {
            self.free.push(b);
        }
    }

    pub fn refs(&self, b: u32) -> u32 {
        self.refs[b as usize]
    }

    /// Start of the `[block_size × head_dim]` run of (block, layer, head).
    fn offset(&self, block: u32, layer: usize, head: usize) -> usize {
        ((block as usize * self.layers + layer) * self.kv_heads + head)
            * self.block_size
            * self.head_dim
    }

    /// Stores one position's keys and values (all KV heads) for a layer, at
    /// index `slot` of `block`.
    pub fn store(&mut self, block: u32, slot: usize, layer: usize, k_row: &[f32], v_row: &[f32]) {
        assert!(slot < self.block_size, "slot beyond the block");
        let d = self.head_dim;
        for head in 0..self.kv_heads {
            let at = self.offset(block, layer, head) + slot * d;
            self.k[at..at + d].copy_from_slice(&k_row[head * d..(head + 1) * d]);
            self.v[at..at + d].copy_from_slice(&v_row[head * d..(head + 1) * d]);
        }
    }

    /// The first `n` keys of (block, layer, head), as `[n × head_dim]`.
    pub fn keys(&self, block: u32, layer: usize, head: usize, n: usize) -> &[f32] {
        let at = self.offset(block, layer, head);
        &self.k[at..at + n * self.head_dim]
    }

    /// The first `n` values of (block, layer, head), as `[n × head_dim]`.
    pub fn values(&self, block: u32, layer: usize, head: usize, n: usize) -> &[f32] {
        let at = self.offset(block, layer, head);
        &self.v[at..at + n * self.head_dim]
    }
}

/// Which blocks hold a sequence's positions, in order: position `p` is at
/// index `p % block_size` of block `blocks[p / block_size]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockTable {
    pub blocks: Vec<u32>,
    /// Positions stored so far.
    pub len: usize,
}

impl BlockTable {
    /// Positions the table's blocks can hold.
    pub fn capacity(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }

    /// Adds blocks from `pool` until the table can hold `len` positions.
    /// Returns false (keeping the blocks it got) if the pool runs out.
    pub fn reserve(&mut self, pool: &mut BlockPool, len: usize) -> bool {
        while self.capacity(pool.block_size()) < len {
            match pool.allocate() {
                Some(b) => self.blocks.push(b),
                None => return false,
            }
        }
        true
    }

    /// Returns every block to the pool (or drops this table's share of it).
    pub fn release(&mut self, pool: &mut BlockPool) {
        for &b in &self.blocks {
            pool.release(b);
        }
        self.blocks.clear();
        self.len = 0;
    }

    /// Stores one position (`self.len` is not changed: the forward pass
    /// advances it after the last layer).
    pub fn store(
        &self,
        pool: &mut BlockPool,
        pos: usize,
        layer: usize,
        k_row: &[f32],
        v_row: &[f32],
    ) {
        let bs = pool.block_size();
        pool.store(self.blocks[pos / bs], pos % bs, layer, k_row, v_row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_counted_shared_and_returned() {
        let config = Config::tiny();
        let mut pool = BlockPool::new(&config, 4, 8);
        let mut a = BlockTable::default();
        assert!(a.reserve(&mut pool, 17));
        assert_eq!((a.blocks.len(), pool.free_blocks()), (3, 1));
        // A second table shares a's first block (as a prefix hit would).
        let mut b = BlockTable::default();
        pool.retain(a.blocks[0]);
        b.blocks.push(a.blocks[0]);
        assert_eq!(pool.refs(a.blocks[0]), 2);
        // Only one block left: reserving 3 more positions' worth fails.
        assert!(!b.reserve(&mut pool, 24));
        assert_eq!(pool.free_blocks(), 0);
        a.release(&mut pool);
        assert_eq!(pool.free_blocks(), 2, "the shared block stays with b");
        b.release(&mut pool);
        assert_eq!(pool.free_blocks(), 4);
    }

    #[test]
    fn a_position_lands_in_its_block() {
        let config = Config::tiny();
        let d = config.head_dim;
        let kv = config.kv_dim();
        let mut pool = BlockPool::new(&config, 3, 4);
        let mut t = BlockTable::default();
        assert!(t.reserve(&mut pool, 9));
        let row: Vec<f32> = (0..kv).map(|i| i as f32).collect();
        t.store(&mut pool, 6, 1, &row, &row);
        // Position 6 is slot 2 of the table's second block.
        let keys = pool.keys(t.blocks[1], 1, 1, 3);
        assert_eq!(&keys[2 * d..3 * d], &row[d..2 * d]);
    }
}
