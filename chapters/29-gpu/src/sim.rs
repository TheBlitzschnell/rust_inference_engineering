//! A small SIMT simulator: GPU kernels, run on the CPU.
//!
//! A GPU runs a kernel as a **grid** of **blocks**, each a group of
//! **threads** that can share fast on-chip memory and wait for each other
//! at **barriers**; threads execute in **warps** of 32 that issue each
//! instruction together. This module reproduces that model closely enough
//! to run real kernel code and check it:
//!
//! - A block's code between two barriers is a **segment** ([`Block::each`]):
//!   the simulator runs it for thread 0, then thread 1, and so on. On a GPU
//!   they run at the same time; any order is a legal schedule as long as no
//!   thread depends on another's work within the segment, which is exactly
//!   what a barrier is for. The simulator checks it: shared memory accesses
//!   that conflict between threads without a barrier in between are
//!   reported as **races**.
//! - Warp shuffles ([`Block::warp_sum`]) exchange registers between the 32
//!   lanes of a warp, all at once.
//! - Global memory loads are grouped by warp and by instruction (the `n`-th
//!   load of each lane in a segment), and counted in 32-byte **sectors**,
//!   the unit GPU memory delivers: 32 lanes reading 32 consecutive floats
//!   need 4 sectors, 32 lanes reading floats 1,000 apart need 32.
//! - Shared memory accesses are counted in **wavefronts**: shared memory has
//!   32 banks of 4-byte words, and lanes that hit different words in the
//!   same bank are served one after another (a bank conflict).
//!
//! Per-thread registers that live across barriers are arrays indexed by
//! thread, kept by the kernel outside its segments.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Threads in a warp.
pub const WARP: usize = 32;
/// Bytes in a global memory sector.
pub const SECTOR: usize = 32;
/// Shared memory banks (4-byte words).
pub const BANKS: usize = 32;

/// Global memory: a read-only array at some byte address.
pub struct Global<'a> {
    pub data: &'a [f32],
    /// Byte address of `data[0]`; arrays of one launch must not overlap.
    pub base: usize,
}

impl<'a> Global<'a> {
    pub fn new(data: &'a [f32], base: usize) -> Self {
        Self { data, base }
    }
}

/// What a launch did, in the units a GPU's profiler reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Global loads (one per lane per load), and the bytes they asked for.
    pub loads: usize,
    pub requested_bytes: usize,
    /// 32-byte sectors the loads touched, counted per warp instruction.
    pub sectors: usize,
    /// Shared memory accesses (warp instructions), and the wavefronts they
    /// took (1 each without bank conflicts).
    pub shared_instructions: usize,
    pub shared_wavefronts: usize,
    pub barriers: usize,
    /// Shared memory races found.
    pub races: usize,
}

impl Stats {
    /// Bytes fetched from memory (sectors × 32).
    pub fn fetched_bytes(&self) -> usize {
        self.sectors * SECTOR
    }

    /// The share of fetched bytes that were asked for.
    pub fn load_efficiency(&self) -> f64 {
        self.requested_bytes as f64 / self.fetched_bytes().max(1) as f64
    }

    fn add(&mut self, o: &Self) {
        self.loads += o.loads;
        self.requested_bytes += o.requested_bytes;
        self.sectors += o.sectors;
        self.shared_instructions += o.shared_instructions;
        self.shared_wavefronts += o.shared_wavefronts;
        self.barriers += o.barriers;
        self.races += o.races;
    }
}

/// Accesses recorded during one segment.
#[derive(Default)]
struct Recorder {
    /// (warp, n-th load of the lane) → sectors.
    global: HashMap<(usize, usize), HashSet<usize>>,
    /// (warp, n-th shared access of the lane) → word addresses.
    shared: HashMap<(usize, usize), HashSet<usize>>,
    /// Per thread: loads and shared accesses so far in this segment.
    counts: HashMap<usize, (usize, usize)>,
    loads: usize,
}

/// One thread's view during a segment.
pub struct Thread<'r> {
    /// `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x`.
    pub index: usize,
    pub block: usize,
    pub block_dim: usize,
    pub grid_dim: usize,
    rec: &'r RefCell<Recorder>,
    shared: &'r RefCell<Shared>,
}

impl Thread<'_> {
    /// `blockIdx.x * blockDim.x + threadIdx.x`.
    pub fn global(&self) -> usize {
        self.block * self.block_dim + self.index
    }

    pub fn lane(&self) -> usize {
        self.index % WARP
    }

    /// The warp within the block.
    pub fn warp(&self) -> usize {
        self.index / WARP
    }

    /// Loads `m.data[i]` from global memory.
    pub fn load(&self, m: &Global<'_>, i: usize) -> f32 {
        let mut rec = self.rec.borrow_mut();
        let n = {
            let c = rec.counts.entry(self.index).or_default();
            c.0 += 1;
            c.0 - 1
        };
        let sector = (m.base + i * 4) / SECTOR;
        rec.global
            .entry((self.warp(), n))
            .or_default()
            .insert(sector);
        rec.loads += 1;
        m.data[i]
    }

    /// Reads shared memory word `i`.
    pub fn shared_read(&self, i: usize) -> f32 {
        self.note_shared(i);
        self.shared.borrow_mut().read(self.index, i)
    }

    /// Writes shared memory word `i`.
    pub fn shared_write(&self, i: usize, v: f32) {
        self.note_shared(i);
        self.shared.borrow_mut().write(self.index, i, v);
    }

    fn note_shared(&self, i: usize) {
        let mut rec = self.rec.borrow_mut();
        let n = {
            let c = rec.counts.entry(self.index).or_default();
            c.1 += 1;
            c.1 - 1
        };
        rec.shared.entry((self.warp(), n)).or_default().insert(i);
    }
}

/// A block's shared memory, with the race detector's bookkeeping since
/// the last barrier.
struct Shared {
    words: Vec<f32>,
    writer: Vec<Option<usize>>,
    /// `Some(t)`: read by thread `t` only; `Some(usize::MAX)`: by several.
    reader: Vec<Option<usize>>,
    races: usize,
}

impl Shared {
    fn new(n: usize) -> Self {
        Self {
            words: vec![0.0; n],
            writer: vec![None; n],
            reader: vec![None; n],
            races: 0,
        }
    }

    fn read(&mut self, t: usize, i: usize) -> f32 {
        if self.writer[i].is_some_and(|w| w != t) {
            self.races += 1; // read after another thread's write
        }
        self.reader[i] = match self.reader[i] {
            None => Some(t),
            Some(r) if r == t => Some(t),
            Some(_) => Some(usize::MAX),
        };
        self.words[i]
    }

    fn write(&mut self, t: usize, i: usize, v: f32) {
        let other_writer = self.writer[i].is_some_and(|w| w != t);
        let other_reader = self.reader[i].is_some_and(|r| r != t);
        if other_writer || other_reader {
            self.races += 1;
        }
        self.writer[i] = Some(t);
        self.words[i] = v;
    }

    fn barrier(&mut self) {
        self.writer.fill(None);
        self.reader.fill(None);
    }
}

/// One block during a launch.
pub struct Block {
    pub index: usize,
    pub dim: usize,
    pub grid_dim: usize,
    rec: RefCell<Recorder>,
    shared: RefCell<Shared>,
    stats: Stats,
}

impl Block {
    /// Runs `f` once for every thread of the block, in order: code that
    /// all threads execute between two barriers.
    pub fn each(&mut self, mut f: impl FnMut(&Thread<'_>)) {
        for index in 0..self.dim {
            let t = Thread {
                index,
                block: self.index,
                block_dim: self.dim,
                grid_dim: self.grid_dim,
                rec: &self.rec,
                shared: &self.shared,
            };
            f(&t);
        }
        self.flush();
    }

    /// `__syncthreads()`: every thread waits here until all have arrived.
    pub fn sync_threads(&mut self) {
        self.stats.barriers += 1;
        self.shared.get_mut().barrier();
    }

    /// The sum over each warp of `v` (one value per thread), left in every
    /// lane: five rounds of `v += shuffle_xor(v, offset)` with offsets 16,
    /// 8, 4, 2, 1, exactly as a GPU's lanes would compute it.
    pub fn warp_sum(&self, v: &mut [f32]) {
        for warp in v.chunks_mut(WARP) {
            let mut offset = WARP / 2;
            while offset > 0 {
                let before = warp.to_vec();
                for (lane, x) in warp.iter_mut().enumerate() {
                    *x += before[lane ^ offset];
                }
                offset /= 2;
            }
        }
    }

    /// Adds this segment's accesses to the statistics.
    fn flush(&mut self) {
        let rec = std::mem::take(self.rec.get_mut());
        self.stats.loads += rec.loads;
        self.stats.requested_bytes += rec.loads * 4;
        self.stats.sectors += rec.global.values().map(HashSet::len).sum::<usize>();
        for words in rec.shared.values() {
            let mut per_bank = [0usize; BANKS];
            for &w in words {
                per_bank[w % BANKS] += 1;
            }
            self.stats.shared_instructions += 1;
            self.stats.shared_wavefronts += per_bank.iter().copied().max().unwrap_or(1).max(1);
        }
    }
}

/// Launches `kernel` on a grid of `grid` blocks of `block` threads, each
/// with `shared` words (4 bytes each) of shared memory. Blocks are
/// independent and run one after another.
pub fn launch(
    grid: usize,
    block: usize,
    shared: usize,
    mut kernel: impl FnMut(&mut Block),
) -> Stats {
    let mut total = Stats::default();
    for index in 0..grid {
        let mut b = Block {
            index,
            dim: block,
            grid_dim: grid,
            rec: RefCell::default(),
            shared: RefCell::new(Shared::new(shared)),
            stats: Stats::default(),
        };
        kernel(&mut b);
        b.stats.races += b.shared.get_mut().races;
        total.add(&b.stats);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesced_loads_touch_four_sectors_per_warp() {
        let data: Vec<f32> = (0..32 * 64).map(|i| i as f32).collect();
        let g = Global::new(&data, 0);
        // Lane i reads element i: 128 contiguous bytes.
        let coalesced = launch(1, 32, 0, |b| {
            b.each(|t| {
                t.load(&g, t.index);
            });
        });
        assert_eq!((coalesced.loads, coalesced.sectors), (32, 4));
        assert!((coalesced.load_efficiency() - 1.0).abs() < 1e-9);
        // Lane i reads element 64·i: every lane its own sector.
        let strided = launch(1, 32, 0, |b| {
            b.each(|t| {
                t.load(&g, t.index * 64);
            });
        });
        assert_eq!(strided.sectors, 32);
        assert!((strided.load_efficiency() - 0.125).abs() < 1e-9);
        // All lanes read the same element: one sector (a broadcast).
        let same = launch(1, 32, 0, |b| {
            b.each(|t| {
                t.load(&g, 5);
            });
        });
        assert_eq!(same.sectors, 1);
    }

    #[test]
    fn bank_conflicts_and_races_are_counted() {
        // Stride 1: 32 banks, one wavefront. Stride 32: one bank, 32.
        for (stride, wavefronts) in [(1, 1), (2, 2), (32, 32)] {
            let s = launch(1, 32, 32 * 32, |b| {
                b.each(|t| t.shared_write(t.index * stride, 1.0));
            });
            assert_eq!(s.shared_wavefronts, wavefronts, "stride {stride}");
            assert_eq!(s.races, 0);
        }
        // Thread i reads what thread i + 1 wrote, without a barrier: a race.
        let racy = launch(1, 32, 32, |b| {
            b.each(|t| t.shared_write(t.index, 1.0));
            b.each(|t| {
                t.shared_read((t.index + 1) % 32);
            });
        });
        assert!(racy.races > 0);
        // With a barrier in between, no race.
        let safe = launch(1, 32, 32, |b| {
            b.each(|t| t.shared_write(t.index, 1.0));
            b.sync_threads();
            b.each(|t| {
                t.shared_read((t.index + 1) % 32);
            });
        });
        assert_eq!((safe.races, safe.barriers), (0, 1));
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "sums of small integers are exact in f32, in any order"
    )]
    fn warp_sum_adds_each_warp() {
        let mut v: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b = Block {
            index: 0,
            dim: 64,
            grid_dim: 1,
            rec: RefCell::default(),
            shared: RefCell::new(Shared::new(0)),
            stats: Stats::default(),
        };
        b.warp_sum(&mut v);
        assert!(v[..32].iter().all(|&x| x == (0..32).sum::<i32>() as f32));
        assert!(v[32..].iter().all(|&x| x == (32..64).sum::<i32>() as f32));
    }
}
