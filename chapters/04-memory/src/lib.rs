//! Chapter 4: measuring the memory hierarchy, and the roofline model.
//!
//! Every function here is a small probe that isolates one property of the
//! machine: how fast it reads, how long one memory access takes, what jumping
//! around in memory costs, how many floating-point operations per second one
//! core can do. The roofline model at the bottom combines two of those
//! numbers to predict the speed limit of any kernel.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Sums a slice with eight independent running totals, so the loop is
/// limited by how fast memory delivers data rather than by the chain of
/// additions (chapter 6 explains why eight totals help).
pub fn sum_fast(data: &[f32]) -> f32 {
    let (chunks, rest) = data.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for chunk in chunks {
        for lane in 0..8 {
            sums[lane] += chunk[lane];
        }
    }
    sums.iter().sum::<f32>() + rest.iter().sum::<f32>()
}

/// Runs `f` several times and returns the fastest run.
///
/// For measuring what the *hardware* can do, the minimum is the right
/// statistic: every source of noise (other processes, interrupts) only ever
/// makes a run slower, never faster. For measuring what *users* experience,
/// you want percentiles instead (chapter 1).
pub fn best_of(runs: usize, mut f: impl FnMut()) -> Duration {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .min()
        .expect("runs must be at least 1")
}

/// Read bandwidth in GB/s for a buffer of `bytes` bytes that has already
/// been touched once (so it is in whatever cache level it fits in).
pub fn read_bandwidth(bytes: usize) -> f64 {
    let data = vec![1.0f32; bytes / 4];
    // Read at least 1 GB in total per measurement so tiny buffers are timed
    // over many passes and the timer's resolution does not matter.
    let passes = (1usize << 30).div_ceil(bytes).max(1);
    black_box(sum_fast(&data)); // warm-up: pull the buffer into cache
    let best = best_of(3, || {
        for _ in 0..passes {
            black_box(sum_fast(black_box(&data)));
        }
    });
    (bytes * passes) as f64 / best.as_secs_f64() / 1e9
}

/// Write bandwidth in GB/s: fills a buffer of `bytes` bytes repeatedly.
pub fn write_bandwidth(bytes: usize) -> f64 {
    let mut data = vec![0.0f32; bytes / 4];
    let passes = (1usize << 30).div_ceil(bytes).max(1);
    data.fill(1.0); // warm-up
    let best = best_of(3, || {
        for p in 0..passes {
            data.fill(p as f32);
            black_box(&data);
        }
    });
    (bytes * passes) as f64 / best.as_secs_f64() / 1e9
}

/// Read bandwidth with several threads, each summing its own share of one
/// large buffer. Chapter 7 covers threads properly; here we only need to
/// know how much bandwidth the whole machine has, not just one core.
pub fn parallel_read_bandwidth(bytes: usize, threads: usize) -> f64 {
    let data = vec![1.0f32; bytes / 4];
    let chunk = data.len().div_ceil(threads);
    let read_all = || {
        std::thread::scope(|s| {
            for part in data.chunks(chunk) {
                s.spawn(move || black_box(sum_fast(black_box(part))));
            }
        });
    };
    read_all(); // warm-up
    let best = best_of(5, read_all);
    bytes as f64 / best.as_secs_f64() / 1e9
}

/// Builds a random single cycle over `n` slots: following `next[i]` from any
/// start visits every slot exactly once before coming back.
///
/// Shuffle the order in which slots will be visited (Fisher-Yates), then
/// link each slot to the one after it in that order, and the last back to
/// the first. The randomness defeats the CPU's prefetcher, which would
/// otherwise guess the next address and hide the latency we want to see.
pub fn random_cycle(n: usize, seed: u64) -> Vec<u32> {
    assert!(n >= 2 && u32::try_from(n).is_ok());
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = seed.max(1);
    for i in (1..n).rev() {
        // xorshift64: a tiny, fast pseudo-random generator.
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut next = vec![0u32; n];
    for w in 0..n {
        next[order[w] as usize] = order[(w + 1) % n];
    }
    next
}

/// Follows the links for `steps` hops. Each load depends on the previous
/// one, so the CPU cannot overlap them: the time per hop is the latency of
/// one memory access.
pub fn chase(next: &[u32], steps: usize) -> u32 {
    let mut i = 0u32;
    for _ in 0..steps {
        i = next[i as usize];
    }
    i
}

/// Average nanoseconds per dependent load, for a working set of `bytes`.
pub fn load_latency_ns(bytes: usize) -> f64 {
    let n = (bytes / 4).max(2);
    let next = random_cycle(n, 0x9E37_79B9_7F4A_7C15);
    let steps = 10_000_000;
    black_box(chase(&next, n.min(steps))); // warm-up
    let best = best_of(3, || {
        black_box(chase(black_box(&next), steps));
    });
    best.as_secs_f64() * 1e9 / steps as f64
}

/// Sums the first float of every block of `STRIDE` floats.
///
/// With `STRIDE` of 16 or more, every float read comes from a different
/// 64-byte cache line. The stride is a compile-time constant (a const
/// generic) so that for small strides the compiler can turn the loop into
/// fast vector code: we want to measure memory, not loop overhead.
pub fn strided_sum<const STRIDE: usize>(data: &[f32]) -> f32 {
    let (blocks, _) = data.as_chunks::<STRIDE>();
    let (groups, rest) = blocks.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for group in groups {
        for lane in 0..8 {
            sums[lane] += group[lane][0];
        }
    }
    sums.iter().sum::<f32>() + rest.iter().map(|block| block[0]).sum::<f32>()
}

/// Time for one pass of `strided_sum::<STRIDE>` over `data`.
pub fn strided_pass<const STRIDE: usize>(data: &[f32]) -> Duration {
    black_box(strided_sum::<STRIDE>(data)); // warm-up
    best_of(3, || {
        black_box(strided_sum::<STRIDE>(black_box(data)));
    })
}

/// A loop of multiply-adds on 128 independent accumulators that live in
/// registers: no memory traffic at all, so it measures arithmetic speed.
///
/// Written as `a * m + c`, which Rust never fuses into a single FMA
/// instruction on its own (fusing changes rounding, so it needs explicit
/// `mul_add` or intrinsics: chapter 6).
pub fn arithmetic_kernel(iterations: usize) -> f32 {
    let mut acc = [0.0f32; 128];
    for (i, a) in acc.iter_mut().enumerate() {
        *a = i as f32 * 0.001;
    }
    let m = black_box(0.999_f32);
    let c = black_box(0.001_f32);
    for _ in 0..iterations {
        for a in &mut acc {
            *a = *a * m + c;
        }
    }
    acc.iter().sum()
}

/// Single-core arithmetic throughput in GFLOP/s (one multiply plus one add
/// counts as two operations).
pub fn peak_gflops_one_core() -> f64 {
    let iterations = 2_000_000;
    black_box(arithmetic_kernel(1000));
    let best = best_of(3, || {
        black_box(arithmetic_kernel(black_box(iterations)));
    });
    (2 * 128 * iterations) as f64 / best.as_secs_f64() / 1e9
}

/// Measures the cost of touching freshly allocated memory for the first
/// time, versus touching it again. Returns (first pass, second pass).
pub fn first_touch(bytes: usize) -> (Duration, Duration) {
    let mut data: Vec<f32> = vec![0.0; bytes / 4];
    let start = Instant::now();
    data.fill(1.0);
    black_box(&data);
    let first = start.elapsed();
    let start = Instant::now();
    data.fill(2.0);
    black_box(&data);
    (first, start.elapsed())
}

/// The roofline model: a kernel can go no faster than the machine's peak
/// arithmetic rate, and no faster than memory can feed it.
#[derive(Debug, Clone, Copy)]
pub struct Roofline {
    /// Peak arithmetic, in GFLOP/s.
    pub peak_gflops: f64,
    /// Memory bandwidth, in GB/s.
    pub bandwidth_gbs: f64,
}

impl Roofline {
    /// The best achievable GFLOP/s for a kernel doing `intensity` FLOPs per
    /// byte moved.
    pub fn attainable(&self, intensity: f64) -> f64 {
        self.peak_gflops.min(intensity * self.bandwidth_gbs)
    }

    /// The intensity where the two limits meet. Kernels below it are
    /// memory-bound, kernels above it are compute-bound.
    pub fn ridge_point(&self) -> f64 {
        self.peak_gflops / self.bandwidth_gbs
    }

    pub fn is_memory_bound(&self, intensity: f64) -> bool {
        intensity < self.ridge_point()
    }
}

/// Arithmetic intensity of `y = W x` (matrix-vector) for weights of
/// `bytes_per_weight` bytes, processing `batch` inputs at once.
///
/// Each weight is read once and used in `batch` multiply-adds (2 FLOPs
/// each). Inputs and outputs are tiny next to the weights and are ignored.
pub fn linear_layer_intensity(bytes_per_weight: f64, batch: usize) -> f64 {
    2.0 * batch as f64 / bytes_per_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_cycle_visits_every_slot_once() {
        for n in [2, 3, 10, 1000] {
            let next = random_cycle(n, 42);
            let mut seen = vec![false; n];
            let mut i = 0usize;
            for _ in 0..n {
                assert!(!seen[i], "slot {i} visited twice");
                seen[i] = true;
                i = next[i] as usize;
            }
            assert_eq!(i, 0, "must return to the start after n hops");
            assert!(seen.iter().all(|&s| s));
        }
    }

    #[test]
    fn strided_sum_reads_the_right_elements() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        // Blocks of 7: the first float of each complete block (14 blocks).
        let expected: f32 = (0..98).step_by(7).map(|i| i as f32).sum();
        assert!((strided_sum::<7>(&data) - expected).abs() < 1e-3);
        assert!((strided_sum::<1>(&data) - 4950.0).abs() < 1e-3);
        assert!((sum_fast(&data) - 4950.0).abs() < 1e-3);
    }

    #[test]
    fn roofline_takes_the_lower_limit() {
        let r = Roofline {
            peak_gflops: 100.0,
            bandwidth_gbs: 20.0,
        };
        assert!((r.ridge_point() - 5.0).abs() < 1e-12);
        assert!((r.attainable(0.5) - 10.0).abs() < 1e-12);
        assert!((r.attainable(50.0) - 100.0).abs() < 1e-12);
        assert!(r.is_memory_bound(1.0));
        assert!(!r.is_memory_bound(10.0));
    }

    #[test]
    fn linear_layer_intensity_scales_with_batch() {
        assert!((linear_layer_intensity(4.0, 1) - 0.5).abs() < 1e-12);
        assert!((linear_layer_intensity(2.0, 1) - 1.0).abs() < 1e-12);
        assert!((linear_layer_intensity(2.0, 16) - 16.0).abs() < 1e-12);
    }
}
