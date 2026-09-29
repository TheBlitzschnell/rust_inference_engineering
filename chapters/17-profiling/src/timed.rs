//! Timing every matrix multiplication of a running model.

use ch07_threads::SpinPool;
use ch14_kv_cache::{Layer, Matrix, Model};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The kinds of matrix in a Llama layer, plus the LM head.
pub const SLOTS: [&str; 8] = ["q", "k", "v", "o", "gate", "up", "down", "lm_head"];

/// Accumulated calls, time and bytes per slot. Atomics, because `matmul`
/// takes `&self` and the model is shared.
pub struct Timings {
    calls: [AtomicU64; 8],
    nanos: [AtomicU64; 8],
    bytes: [AtomicU64; 8],
}

/// One slot's totals.
#[derive(Clone, Copy, Debug)]
pub struct SlotTotal {
    pub name: &'static str,
    pub calls: u64,
    pub time: Duration,
    /// Weight bytes read (each call reads the whole matrix once).
    pub bytes: u64,
}

impl Timings {
    fn new() -> Self {
        Self {
            calls: Default::default(),
            nanos: Default::default(),
            bytes: Default::default(),
        }
    }

    pub fn reset(&self) {
        for a in self.calls.iter().chain(&self.nanos).chain(&self.bytes) {
            a.store(0, Ordering::Relaxed);
        }
    }

    pub fn totals(&self) -> Vec<SlotTotal> {
        SLOTS
            .iter()
            .enumerate()
            .map(|(i, &name)| SlotTotal {
                name,
                calls: self.calls[i].load(Ordering::Relaxed),
                time: Duration::from_nanos(self.nanos[i].load(Ordering::Relaxed)),
                bytes: self.bytes[i].load(Ordering::Relaxed),
            })
            .collect()
    }
}

/// A matrix that records how long each of its multiplications takes.
pub struct Timed<W> {
    inner: W,
    slot: usize,
    timings: Arc<Timings>,
}

impl<W: Matrix> Matrix for Timed<W> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn bytes(&self) -> usize {
        self.inner.bytes()
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        self.inner.row_to_f32(r, out);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        let start = Instant::now();
        self.inner.matmul(pool, x, y, m, scratch);
        let t = &self.timings;
        t.nanos[self.slot].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        t.calls[self.slot].fetch_add(1, Ordering::Relaxed);
        t.bytes[self.slot].fetch_add(self.inner.bytes() as u64, Ordering::Relaxed);
    }

    /// Forwards to the inner type's `matmul_many` (so a fused pass stays
    /// fused) and splits the time between the matrices in proportion to
    /// their bytes.
    fn matmul_many(
        pool: &mut SpinPool,
        matrices: &[&Self],
        x: &[f32],
        outputs: &mut [&mut [f32]],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        let inner: Vec<&W> = matrices.iter().map(|t| &t.inner).collect();
        let start = Instant::now();
        W::matmul_many(pool, &inner, x, outputs, m, scratch);
        let nanos = start.elapsed().as_nanos() as u64;
        let total: u64 = matrices.iter().map(|t| t.inner.bytes() as u64).sum();
        for t in matrices {
            let bytes = t.inner.bytes() as u64;
            let timings = &t.timings;
            timings.nanos[t.slot].fetch_add(nanos * bytes / total.max(1), Ordering::Relaxed);
            timings.calls[t.slot].fetch_add(1, Ordering::Relaxed);
            timings.bytes[t.slot].fetch_add(bytes, Ordering::Relaxed);
        }
    }
}

/// Wraps every matrix of `model` in a timer. The embedding, which is also
/// the LM head when they are tied, goes into the `lm_head` slot: looking up
/// embedding rows is not a multiplication and is not timed.
pub fn instrument<W: Matrix>(model: Model<W>) -> (Model<Timed<W>>, Arc<Timings>) {
    let timings = Arc::new(Timings::new());
    let wrap = |inner: W, slot: usize| Timed {
        inner,
        slot,
        timings: Arc::clone(&timings),
    };
    let (config, embed, layers, final_norm, lm_head) = model.into_parts();
    let layers = layers
        .into_iter()
        .map(|l| Layer {
            attn_norm: l.attn_norm,
            wq: wrap(l.wq, 0),
            wk: wrap(l.wk, 1),
            wv: wrap(l.wv, 2),
            wo: wrap(l.wo, 3),
            mlp_norm: l.mlp_norm,
            w_gate: wrap(l.w_gate, 4),
            w_up: wrap(l.w_up, 5),
            w_down: wrap(l.w_down, 6),
        })
        .collect();
    let model = Model::new(
        config,
        wrap(embed, 7),
        layers,
        final_norm,
        lm_head.map(|h| wrap(h, 7)),
    );
    (model, timings)
}
