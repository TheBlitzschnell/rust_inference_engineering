//! Chapter 17: profiling the real model, and acting on what it shows.
//!
//! - [`timed`]: a wrapper that times every matrix multiplication of a
//!   running model, per kind of matrix.
//! - [`kernels`]: `dot4_bf16`, four weight rows against one activation
//!   vector, and `tile_bf16`, four weight rows against four activations.
//! - [`matmul`]: parallel matmuls built on them, and [`matvec_many`],
//!   several matrices in one parallel pass.
//! - [`Rows4Bf16`] and [`FusedBf16`]: two ideas for decode, which the
//!   measurements reject.
//! - [`TiledBf16`]: the tile kernel for prefill, which they support.
//! - [`Stats`], [`measure`] and [`compare`]: timing on a noisy machine.

use ch02_numbers::Bf16;
use ch07_threads::SpinPool;
use ch14_kv_cache::{Layer, Matrix, Model};
use ch16_real_model::DenseBf16;
use std::time::{Duration, Instant};

pub mod kernels;
pub mod matmul;
pub mod timed;

pub use kernels::{dot4_bf16, dot4_bf16_with, tile_bf16, tile_bf16_with};
pub use matmul::{matmul_rows4, matmul_tiled, matvec_many};
pub use timed::{SLOTS, SlotTotal, Timed, Timings, instrument};

/// Chapter 16's `bf16` matrix, multiplied with the four-row kernel. An
/// experiment: the lesson measures why it does not help SmolLM2.
pub struct Rows4Bf16(pub DenseBf16);

impl Matrix for Rows4Bf16 {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn bytes(&self) -> usize {
        self.0.bytes()
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        self.0.row_to_f32(r, out);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        matmul_rows4(
            pool,
            x,
            self.0.values(),
            y,
            m,
            self.0.cols(),
            self.0.rows(),
            scratch,
            ch06_simd::dot_bf16,
            dot4_bf16,
        );
    }
}

/// Chapter 16's `bf16` matrix, with the matrices that share an input (Q, K
/// and V; gate and up) multiplied in one parallel pass when decoding.
pub struct FusedBf16(pub DenseBf16);

impl Matrix for FusedBf16 {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn bytes(&self) -> usize {
        self.0.bytes()
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        self.0.row_to_f32(r, out);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        self.0.matmul(pool, x, y, m, scratch);
    }

    fn matmul_many(
        pool: &mut SpinPool,
        matrices: &[&Self],
        x: &[f32],
        outputs: &mut [&mut [f32]],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        if m > 1 {
            // Prefill multiplies large blocks and is compute-bound: fusing
            // buys little there, so keep the separate products.
            for (w, y) in matrices.iter().zip(outputs.iter_mut()) {
                w.matmul(pool, x, y, m, scratch);
            }
            return;
        }
        let parts: Vec<&[Bf16]> = matrices.iter().map(|w| w.0.values()).collect();
        matvec_many(
            pool,
            &parts,
            x.len(),
            x,
            outputs,
            scratch,
            ch06_simd::dot_bf16,
        );
    }
}

/// Chapter 16's `bf16` matrix with a tiled kernel for prefill. Decode is
/// unchanged: it is memory-bound, and the profile found nothing to gain.
pub struct TiledBf16(pub DenseBf16);

impl Matrix for TiledBf16 {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn bytes(&self) -> usize {
        self.0.bytes()
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        self.0.row_to_f32(r, out);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        matmul_tiled(
            pool,
            x,
            self.0.values(),
            y,
            m,
            self.0.cols(),
            self.0.rows(),
            scratch,
            ch06_simd::dot_bf16,
            tile_bf16,
        );
    }
}

/// Rebuilds a model around different matrices, converting each with `f`.
pub fn map_matrices<W: Matrix, V: Matrix>(model: Model<W>, mut f: impl FnMut(W) -> V) -> Model<V> {
    let (config, embed, layers, final_norm, lm_head) = model.into_parts();
    let layers = layers
        .into_iter()
        .map(|l| Layer {
            attn_norm: l.attn_norm,
            wq: f(l.wq),
            wk: f(l.wk),
            wv: f(l.wv),
            wo: f(l.wo),
            mlp_norm: l.mlp_norm,
            w_gate: f(l.w_gate),
            w_up: f(l.w_up),
            w_down: f(l.w_down),
        })
        .collect();
    let embed = f(embed);
    let lm_head = lm_head.map(&mut f);
    Model::new(config, embed, layers, final_norm, lm_head)
}

/// Summary of repeated measurements.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub runs: usize,
    pub min: Duration,
    pub p10: Duration,
    pub median: Duration,
    pub p90: Duration,
}

impl Stats {
    pub fn from_samples(samples: &mut [Duration]) -> Self {
        assert!(!samples.is_empty(), "no samples");
        samples.sort_unstable();
        let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
        Self {
            runs: samples.len(),
            min: samples[0],
            p10: at(0.1),
            median: at(0.5),
            p90: at(0.9),
        }
    }
}

/// Runs `f` `warmup` times untimed, then `runs` times timed.
///
/// Warm-up runs fault in memory, fill caches and let the thread pool spin
/// up, so the timed runs measure the steady state.
pub fn measure(warmup: usize, runs: usize, mut f: impl FnMut()) -> Stats {
    for _ in 0..warmup {
        f();
    }
    let mut samples: Vec<Duration> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    Stats::from_samples(&mut samples)
}

/// The result of an interleaved A/B comparison.
#[derive(Clone, Copy, Debug)]
pub struct Comparison {
    pub a: Stats,
    pub b: Stats,
    /// Median over pairs of `time(a) / time(b)`: above 1 means B is faster.
    pub ratio: f64,
    /// 10th and 90th percentiles of the per-pair ratios.
    pub ratio_p10: f64,
    pub ratio_p90: f64,
}

/// Compares two versions of the same work on a noisy machine.
///
/// Runs `a` and `b` alternately, `pairs` times each (in the order AB, BA,
/// AB... so neither always goes first), and compares each run with its
/// neighbour. Load from other programs changes slowly compared with one
/// pair, so it affects both halves of a pair almost equally and cancels in
/// their ratio. Both closures receive the same thread pool: two pools would
/// spin against each other.
pub fn compare(
    pool: &mut SpinPool,
    pairs: usize,
    mut a: impl FnMut(&mut SpinPool),
    mut b: impl FnMut(&mut SpinPool),
) -> Comparison {
    a(pool);
    b(pool);
    let time = |f: &mut dyn FnMut(&mut SpinPool), pool: &mut SpinPool| {
        let start = Instant::now();
        f(pool);
        start.elapsed()
    };
    let (mut ta, mut tb, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..pairs {
        let (x, y) = if i % 2 == 0 {
            let x = time(&mut a, pool);
            (x, time(&mut b, pool))
        } else {
            let y = time(&mut b, pool);
            (time(&mut a, pool), y)
        };
        ratios.push(x.as_secs_f64() / y.as_secs_f64());
        ta.push(x);
        tb.push(y);
    }
    ratios.sort_by(f64::total_cmp);
    let at = |q: f64| ratios[((ratios.len() - 1) as f64 * q).round() as usize];
    Comparison {
        a: Stats::from_samples(&mut ta),
        b: Stats::from_samples(&mut tb),
        ratio: at(0.5),
        ratio_p10: at(0.1),
        ratio_p90: at(0.9),
    }
}
