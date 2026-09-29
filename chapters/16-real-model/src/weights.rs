//! Loading `model.safetensors` into chapter 14's engine.

use crate::Error;
use crate::config::{ModelInfo, read_config};
use ch02_numbers::{Bf16, bf16_to_f32_slice};
use ch06_simd::{AlignedVec, dot_bf16};
use ch07_threads::SpinPool;
use ch09_safetensors::{Dtype, MappedFile, SafeTensors, reinterpret};
use ch14_kv_cache::{DenseF32, Layer, Matrix, Model, matmul_pooled};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// Where a `bf16` matrix's numbers live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Straight from the memory-mapped file: nothing is copied, pages are
    /// read from disk on first use and shared with every other process that
    /// maps the file. The data is wherever the file puts it.
    Mapped,
    /// Copied into 64-byte-aligned memory owned by the process.
    Aligned,
}

enum Bf16Data {
    Mapped { file: Arc<MappedFile>, start: usize },
    Aligned(AlignedVec<Bf16>),
}

/// A `bf16` weight matrix. Activations stay `f32`; each weight is converted
/// as it is used, inside chapter 6's `dot_bf16` kernel.
pub struct DenseBf16 {
    data: Bf16Data,
    rows: usize,
    cols: usize,
}

impl DenseBf16 {
    /// The weights, `rows × cols`, row by row.
    pub fn values(&self) -> &[Bf16] {
        match &self.data {
            Bf16Data::Mapped { file, start } => {
                let bytes = &file.bytes()[*start..*start + self.rows * self.cols * 2];
                reinterpret(bytes).expect("alignment was checked when loading")
            }
            Bf16Data::Aligned(values) => values,
        }
    }

    /// The address of the first weight, modulo 64 (0 for a cache-line
    /// aligned matrix).
    pub fn misalignment(&self) -> usize {
        self.values().as_ptr() as usize % 64
    }
}

impl Matrix for DenseBf16 {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn bytes(&self) -> usize {
        self.rows * self.cols * 2
    }

    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        bf16_to_f32_slice(&self.values()[r * self.cols..(r + 1) * self.cols], out);
    }

    fn matmul(
        &self,
        pool: &mut SpinPool,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        scratch: &mut Vec<f32>,
    ) {
        matmul_pooled(
            pool,
            x,
            self.values(),
            y,
            m,
            self.cols,
            self.rows,
            scratch,
            dot_bf16,
        );
    }
}

/// The open checkpoint: the mapped file and its parsed header.
struct Checkpoint<'a> {
    file: &'a Arc<MappedFile>,
    tensors: SafeTensors<'a>,
    used: BTreeSet<String>,
}

impl<'a> Checkpoint<'a> {
    /// The tensor's bytes, after checking its type and shape.
    fn bytes(&mut self, name: &str, shape: &[usize]) -> Result<&'a [u8], Error> {
        let t = self
            .tensors
            .tensor(name)
            .map_err(|_| Error::Tensor(format!("missing tensor {name}")))?;
        if t.dtype != Dtype::BF16 {
            return Err(Error::Unsupported(format!(
                "{name} is {:?}, not BF16",
                t.dtype
            )));
        }
        if t.shape != shape {
            return Err(Error::Tensor(format!(
                "{name} has shape {:?}, the config implies {shape:?}",
                t.shape
            )));
        }
        self.used.insert(name.to_owned());
        Ok(t.data)
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        let bytes = self.bytes(name, &[len])?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| Bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }

    fn bf16(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        placement: Placement,
    ) -> Result<DenseBf16, Error> {
        let bytes = self.bytes(name, &[rows, cols])?;
        let values: &[Bf16] = reinterpret(bytes)
            .ok_or_else(|| Error::Tensor(format!("{name} is not 2-byte aligned in the file")))?;
        let data = match placement {
            Placement::Mapped => Bf16Data::Mapped {
                file: Arc::clone(self.file),
                // Offset of the tensor inside the file.
                start: bytes.as_ptr() as usize - self.file.bytes().as_ptr() as usize,
            },
            Placement::Aligned => Bf16Data::Aligned(AlignedVec::from_slice(values)),
        };
        Ok(DenseBf16 { data, rows, cols })
    }

    fn f32(&mut self, name: &str, rows: usize, cols: usize) -> Result<DenseF32, Error> {
        let bytes = self.bytes(name, &[rows, cols])?;
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| Bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        Ok(DenseF32::new(&values, rows, cols))
    }
}

/// Loads SmolLM2 (or any Llama-architecture checkpoint in `bf16` this engine
/// supports) with `bf16` weights.
pub fn load_bf16(dir: &Path, placement: Placement) -> Result<(Model<DenseBf16>, ModelInfo), Error> {
    load(dir, |ck, name, rows, cols| {
        ck.bf16(name, rows, cols, placement)
    })
}

/// Loads the same checkpoint with every weight converted to `f32`: twice
/// the memory, the same numbers.
#[expect(
    clippy::redundant_closure_for_method_calls,
    reason = "the suggested `Checkpoint::f32` does not compile: it is not general over lifetimes"
)]
pub fn load_f32(dir: &Path) -> Result<(Model<DenseF32>, ModelInfo), Error> {
    // A closure, not the path `Checkpoint::f32`: the path names the method
    // for one particular lifetime of `Checkpoint<'_>`, while `load` needs a
    // function that works for any (see the lesson, section 6).
    load(dir, |ck, name, rows, cols| ck.f32(name, rows, cols))
}

fn load<W: Matrix>(
    dir: &Path,
    mut matrix: impl FnMut(&mut Checkpoint<'_>, &str, usize, usize) -> Result<W, Error>,
) -> Result<(Model<W>, ModelInfo), Error> {
    let info = read_config(&dir.join("config.json"))?;
    let path = dir.join("model.safetensors");
    let file = Arc::new(MappedFile::open(&path).map_err(|source| Error::Io { path, source })?);
    let mut ck = Checkpoint {
        file: &file,
        tensors: SafeTensors::parse(file.bytes())?,
        used: BTreeSet::new(),
    };
    let c = &info.config;
    let (h, q, kv, inter) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size);

    let mut layers = Vec::with_capacity(c.num_layers);
    for i in 0..c.num_layers {
        let p = format!("model.layers.{i}");
        layers.push(Layer {
            attn_norm: ck.vector(&format!("{p}.input_layernorm.weight"), h)?,
            wq: matrix(&mut ck, &format!("{p}.self_attn.q_proj.weight"), q, h)?,
            wk: matrix(&mut ck, &format!("{p}.self_attn.k_proj.weight"), kv, h)?,
            wv: matrix(&mut ck, &format!("{p}.self_attn.v_proj.weight"), kv, h)?,
            wo: matrix(&mut ck, &format!("{p}.self_attn.o_proj.weight"), h, q)?,
            mlp_norm: ck.vector(&format!("{p}.post_attention_layernorm.weight"), h)?,
            w_gate: matrix(&mut ck, &format!("{p}.mlp.gate_proj.weight"), inter, h)?,
            w_up: matrix(&mut ck, &format!("{p}.mlp.up_proj.weight"), inter, h)?,
            w_down: matrix(&mut ck, &format!("{p}.mlp.down_proj.weight"), h, inter)?,
        });
    }
    let embed = matrix(&mut ck, "model.embed_tokens.weight", c.vocab_size, h)?;
    let final_norm = ck.vector("model.norm.weight", h)?;
    let lm_head = if c.tie_embeddings {
        None
    } else {
        Some(matrix(&mut ck, "lm_head.weight", c.vocab_size, h)?)
    };

    // A tensor nobody asked for means the checkpoint has parts this engine
    // does not know about (biases, extra norms...): refuse rather than
    // silently compute something else.
    if let Some(extra) = ck.tensors.names().find(|n| !ck.used.contains(*n)) {
        return Err(Error::Unsupported(format!("unexpected tensor {extra}")));
    }
    let model = Model::new(c.clone(), embed, layers, final_norm, lm_head);
    Ok((model, info))
}
