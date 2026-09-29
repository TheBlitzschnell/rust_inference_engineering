//! Chapter 16: running a real model.
//!
//! Everything so far used random weights. This crate loads SmolLM2 from the
//! three files Hugging Face publishes for it and runs it on chapter 14's
//! engine:
//!
//! - `config.json` → the architecture ([`config`]),
//! - `model.safetensors` → `bf16` weights, zero-copy from a memory map or
//!   copied into aligned memory ([`weights`]),
//! - `tokenizer.json` → the byte-level BPE tokenizer ([`tokenizer`]),
//!
//! plus the chat template that turns a conversation into a prompt
//! ([`chat`]). Tests compare every step against the Hugging Face reference.

use std::fmt;
use std::path::PathBuf;

pub mod chat;
pub mod config;
pub mod tokenizer;
pub mod weights;

pub use chat::{DEFAULT_SYSTEM, Message, chat_prompt};
pub use config::{ModelInfo, parse_config, read_config};
pub use tokenizer::Tokenizer;
pub use weights::{DenseBf16, Placement, load_bf16, load_f32};

/// Where `./tools/download_model.sh` puts SmolLM2-135M-Instruct.
pub fn model_dir() -> PathBuf {
    ch09_safetensors::smollm2_dir("135m")
}

/// Everything that can go wrong while loading a model.
#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    SafeTensors(ch09_safetensors::Error),
    /// The checkpoint uses a feature this engine does not implement.
    Unsupported(String),
    /// A tensor is missing, has the wrong type or the wrong shape.
    Tensor(String),
    Tokenizer(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Json { path, source } => write!(f, "{}: invalid JSON: {source}", path.display()),
            Error::SafeTensors(e) => write!(f, "safetensors: {e}"),
            Error::Unsupported(what) => write!(f, "unsupported model: {what}"),
            Error::Tensor(what) => write!(f, "bad checkpoint: {what}"),
            Error::Tokenizer(what) => write!(f, "bad tokenizer: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Json { source, .. } => Some(source),
            Error::SafeTensors(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ch09_safetensors::Error> for Error {
    fn from(e: ch09_safetensors::Error) -> Self {
        Error::SafeTensors(e)
    }
}

/// Reads a whole file, attaching its path to any error.
fn read_to_string(path: &std::path::Path) -> Result<String, Error> {
    std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}
