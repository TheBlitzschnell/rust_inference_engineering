//! Chapter 10: a complete, small model from training to serving.
//!
//! The model is a multi-layer perceptron (MLP): a stack of linear layers with
//! a ReLU between them. It classifies 2-D points from the "spirals" dataset
//! into one of three interleaved spiral arms. The pieces:
//!
//! - [`spirals`] generates the data,
//! - [`train`] fits the weights with plain gradient descent (the one part of
//!   this course that is about training, kept as short as possible),
//! - [`Mlp::save`] and [`Mlp::load`] go through chapter 9's safetensors code,
//! - [`Mlp::forward`] is the inference path: allocation-free, batched, and
//!   parallel through chapter 7's [`SpinPool`].

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use ch06_simd::AlignedVec;
use ch07_threads::{SpinPool, matmul_nt_pool};
use ch08_operators::{relu, softmax};
use ch09_safetensors::{Dtype, MappedFile, SafeTensors, TensorToWrite, f32_bytes, serialize};

pub mod train;

/// A small deterministic random number generator (xorshift64*).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximately standard normal (sum of 12 uniforms, minus 6).
    pub fn normal(&mut self) -> f32 {
        (0..12).map(|_| self.uniform()).sum::<f32>() - 6.0
    }
}

/// Points on `classes` interleaved spiral arms, `per_class` points each.
/// Returns the points (`[n × 2]`, row-major) and their labels.
pub fn spirals(per_class: usize, classes: usize, noise: f32, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut points = Vec::with_capacity(per_class * classes * 2);
    let mut labels = Vec::with_capacity(per_class * classes);
    for class in 0..classes {
        for i in 0..per_class {
            let r = i as f32 / per_class as f32;
            let angle = class as f32 * std::f32::consts::TAU / classes as f32
                + r * 4.0
                + rng.normal() * noise;
            points.push(r * angle.cos());
            points.push(r * angle.sin());
            labels.push(class);
        }
    }
    (points, labels)
}

/// One fully connected layer: `y = W x + b`, with `W` stored as
/// `[out × in]`, one output per row, as PyTorch stores it.
#[derive(Clone)]
pub struct Linear {
    pub weight: AlignedVec<f32>,
    pub bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    /// Random weights scaled by `sqrt(2 / in)` ("He" initialization), which
    /// keeps activations from growing or shrinking through ReLU layers.
    pub fn random(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        Self {
            weight: AlignedVec::from_fn(in_dim * out_dim, |_| rng.normal() * scale),
            bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    pub fn params(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

/// Everything that can go wrong when loading a model file.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Format(ch09_safetensors::Error),
    Shape(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "cannot read model file: {e}"),
            LoadError::Format(e) => write!(f, "invalid model file: {e}"),
            LoadError::Shape(why) => write!(f, "model file has the wrong shapes: {why}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<ch09_safetensors::Error> for LoadError {
    fn from(e: ch09_safetensors::Error) -> Self {
        LoadError::Format(e)
    }
}

/// Reusable buffers for one forward pass: two "ping-pong" activation buffers
/// big enough for the widest layer at the largest batch. Allocated once,
/// reused for every request.
pub struct Workspace {
    a: Vec<f32>,
    b: Vec<f32>,
    max_batch: usize,
}

impl Workspace {
    pub fn new(model: &Mlp, max_batch: usize) -> Self {
        let widest = model
            .layers
            .iter()
            .map(|l| l.in_dim.max(l.out_dim))
            .max()
            .unwrap_or(0);
        Self {
            a: vec![0.0; widest * max_batch],
            b: vec![0.0; widest * max_batch],
            max_batch,
        }
    }
}

/// A multi-layer perceptron: linear layers with ReLU between them (not after
/// the last one, whose outputs are the logits).
#[derive(Clone)]
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    /// A randomly initialized network with the given layer widths, e.g.
    /// `[2, 64, 64, 3]`.
    pub fn random(widths: &[usize], seed: u64) -> Self {
        assert!(
            widths.len() >= 2,
            "need at least an input and an output width"
        );
        let mut rng = Rng::new(seed);
        let layers = widths
            .windows(2)
            .map(|w| Linear::random(w[0], w[1], &mut rng))
            .collect();
        Self { layers }
    }

    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }

    pub fn output_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].out_dim
    }

    pub fn params(&self) -> usize {
        self.layers.iter().map(Linear::params).sum()
    }

    /// Runs `batch` inputs (`x`, `[batch × input_dim]`) through the network
    /// and returns the logits (`[batch × output_dim]`), which live in the
    /// workspace. The activations never allocate; chapter 7's parallel
    /// matmul still makes two small temporary allocations per layer (its
    /// chunk list, and a transposed buffer when `batch > 1`), which the
    /// engine in chapter 14 removes.
    pub fn forward<'w>(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        batch: usize,
        ws: &'w mut Workspace,
    ) -> &'w [f32] {
        assert!(batch <= ws.max_batch, "batch larger than the workspace");
        assert_eq!(
            x.len(),
            batch * self.input_dim(),
            "input has the wrong size"
        );
        let Workspace { a, b, .. } = ws;
        a[..x.len()].copy_from_slice(x);
        let (mut input, mut output) = (a, b);
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let x_in = &input[..batch * layer.in_dim];
            let y = &mut output[..batch * layer.out_dim];
            matmul_nt_pool(
                pool,
                x_in,
                &layer.weight,
                y,
                batch,
                layer.in_dim,
                layer.out_dim,
            );
            for row in y.chunks_exact_mut(layer.out_dim) {
                for (v, b) in row.iter_mut().zip(&layer.bias) {
                    *v += b;
                }
                if i != last {
                    relu(row);
                }
            }
            std::mem::swap(&mut input, &mut output);
        }
        &input[..batch * self.output_dim()]
    }

    /// Saves the weights as `layers.{i}.weight` / `layers.{i}.bias`.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let names: Vec<(String, String)> = (0..self.layers.len())
            .map(|i| (format!("layers.{i}.weight"), format!("layers.{i}.bias")))
            .collect();
        let bytes: Vec<(Vec<u8>, Vec<u8>)> = self
            .layers
            .iter()
            .map(|l| (f32_bytes(&l.weight), f32_bytes(&l.bias)))
            .collect();
        let shapes: Vec<([usize; 2], [usize; 1])> = self
            .layers
            .iter()
            .map(|l| ([l.out_dim, l.in_dim], [l.out_dim]))
            .collect();
        let mut tensors = Vec::new();
        for ((names, bytes), shapes) in names.iter().zip(&bytes).zip(&shapes) {
            tensors.push(TensorToWrite {
                name: &names.0,
                dtype: Dtype::F32,
                shape: &shapes.0,
                data: &bytes.0,
            });
            tensors.push(TensorToWrite {
                name: &names.1,
                dtype: Dtype::F32,
                shape: &shapes.1,
                data: &bytes.1,
            });
        }
        let mut meta = BTreeMap::new();
        meta.insert("architecture".to_string(), "mlp-relu".to_string());
        meta.insert("layers".to_string(), self.layers.len().to_string());
        std::fs::write(path, serialize(&tensors, &meta))
    }

    /// Loads a model saved by [`Mlp::save`], checking every shape.
    ///
    /// The weights are copied out of the memory map into aligned buffers
    /// owned by the model (the "owned copies" design of chapter 9), so the
    /// model has no lifetime parameter and the file can be closed.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let file = MappedFile::open(path)?;
        let st = SafeTensors::parse(file.bytes())?;
        let count: usize = st
            .metadata()
            .get("layers")
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| LoadError::Shape("missing \"layers\" metadata".into()))?;
        let mut layers: Vec<Linear> = Vec::with_capacity(count);
        for i in 0..count {
            let w = st.tensor(&format!("layers.{i}.weight"))?;
            let b = st.tensor(&format!("layers.{i}.bias"))?;
            let [out_dim, in_dim] = w.shape[..] else {
                return Err(LoadError::Shape(format!("layer {i} weight is not 2-D")));
            };
            if b.shape != [out_dim] {
                return Err(LoadError::Shape(format!(
                    "layer {i} bias has shape {:?}, expected [{out_dim}]",
                    b.shape
                )));
            }
            if let Some(prev) = layers.last()
                && prev.out_dim != in_dim
            {
                return Err(LoadError::Shape(format!(
                    "layer {i} takes {in_dim} inputs but layer {} produces {}",
                    i - 1,
                    prev.out_dim
                )));
            }
            layers.push(Linear {
                weight: AlignedVec::from_slice(&w.to_f32_vec()?),
                bias: b.to_f32_vec()?,
                in_dim,
                out_dim,
            });
        }
        if layers.is_empty() {
            return Err(LoadError::Shape("no layers".into()));
        }
        Ok(Self { layers })
    }
}

/// Index of the largest value (the predicted class).
pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
            if x > best.1 { (i, x) } else { best }
        })
        .0
}

/// Turns one row of logits into probabilities, in place.
pub fn probabilities(logits: &mut [f32]) {
    softmax(logits);
}

/// Fraction of `points` whose predicted class matches `labels`.
pub fn accuracy(model: &Mlp, pool: &mut SpinPool, points: &[f32], labels: &[usize]) -> f32 {
    let n = labels.len();
    let mut ws = Workspace::new(model, n);
    let logits = model.forward(pool, points, n, &mut ws);
    let correct = logits
        .chunks_exact(model.output_dim())
        .zip(labels)
        .filter(|(row, label)| argmax(row) == **label)
        .count();
    correct as f32 / n as f32
}

/// Where the trained spiral classifier is stored: `models/spiral-mlp.safetensors`
/// at the repository root (or under `$INFER_MODELS`).
pub fn model_path() -> std::path::PathBuf {
    let models = std::env::var_os("INFER_MODELS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"),
        std::path::PathBuf::from,
    );
    models.join("spiral-mlp.safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_forward_matches_one_at_a_time() {
        let model = Mlp::random(&[5, 16, 7, 3], 1);
        let mut pool = SpinPool::new(3);
        let batch = 9;
        let mut rng = Rng::new(2);
        let x: Vec<f32> = (0..batch * 5).map(|_| rng.normal()).collect();
        let mut ws = Workspace::new(&model, batch);
        let all = model.forward(&mut pool, &x, batch, &mut ws).to_vec();
        let mut one = Workspace::new(&model, 1);
        for i in 0..batch {
            let row = model.forward(&mut pool, &x[i * 5..(i + 1) * 5], 1, &mut one);
            for (a, b) in row.iter().zip(&all[i * 3..(i + 1) * 3]) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn save_and_load_round_trip() {
        let model = Mlp::random(&[2, 8, 3], 5);
        let dir = std::env::temp_dir().join(format!("ch10-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.safetensors");
        model.save(&path).unwrap();
        let loaded = Mlp::load(&path).unwrap();
        for (a, b) in model.layers.iter().zip(&loaded.layers) {
            assert_eq!(&a.weight[..], &b.weight[..]);
            assert_eq!(a.bias, b.bias);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_rejects_a_broken_layer_chain() {
        // Hand-build a file whose second layer expects 5 inputs but the first
        // produces 4.
        let w0 = f32_bytes(&[0.0; 8]); // [4 × 2]
        let b0 = f32_bytes(&[0.0; 4]);
        let w1 = f32_bytes(&[0.0; 15]); // [3 × 5]
        let b1 = f32_bytes(&[0.0; 3]);
        let mut meta = BTreeMap::new();
        meta.insert("layers".into(), "2".into());
        let bytes = serialize(
            &[
                TensorToWrite {
                    name: "layers.0.weight",
                    dtype: Dtype::F32,
                    shape: &[4, 2],
                    data: &w0,
                },
                TensorToWrite {
                    name: "layers.0.bias",
                    dtype: Dtype::F32,
                    shape: &[4],
                    data: &b0,
                },
                TensorToWrite {
                    name: "layers.1.weight",
                    dtype: Dtype::F32,
                    shape: &[3, 5],
                    data: &w1,
                },
                TensorToWrite {
                    name: "layers.1.bias",
                    dtype: Dtype::F32,
                    shape: &[3],
                    data: &b1,
                },
            ],
            &meta,
        );
        let path =
            std::env::temp_dir().join(format!("ch10-bad-{}.safetensors", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let err = Mlp::load(&path).err().expect("must be rejected");
        assert!(matches!(err, LoadError::Shape(_)), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn training_learns_the_spirals() {
        let (x, y) = spirals(100, 3, 0.2, 7);
        let mut model = Mlp::random(&[2, 32, 32, 3], 3);
        let mut pool = SpinPool::new(2);
        let before = accuracy(&model, &mut pool, &x, &y);
        train::train(&mut model, &x, &y, 600, 0.2, |_, _| {});
        let after = accuracy(&model, &mut pool, &x, &y);
        assert!(after > 0.9, "accuracy {before} -> {after}");
    }

    #[test]
    fn argmax_picks_the_largest() {
        assert_eq!(argmax(&[0.1, 3.0, -1.0, 2.9]), 1);
    }
}
