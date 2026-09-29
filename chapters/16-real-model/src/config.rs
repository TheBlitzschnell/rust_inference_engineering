//! Reading `config.json`: the model's architecture as numbers.

use crate::Error;
use ch12_attention::RopeLayout;
use ch13_transformer::Config;
use serde::Deserialize;
use std::path::Path;

/// The fields of a Hugging Face Llama `config.json` this engine reads.
/// Unknown fields are ignored; fields that change the computation are
/// checked, so an unsupported model fails loudly instead of running wrong.
#[derive(Debug, Deserialize)]
struct HfConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    /// Absent in older configs, which means one KV head per query head.
    num_key_value_heads: Option<usize>,
    /// Absent in most configs, which means `hidden_size / num_attention_heads`.
    head_dim: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    hidden_act: String,
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    bos_token_id: Option<u32>,
    eos_token_id: Option<OneOrMany>,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

/// `eos_token_id` is a number in some configs and a list in others.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

/// The architecture plus the special tokens the config names.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub config: Config,
    pub bos_token: Option<u32>,
    /// Tokens that end generation.
    pub eos_tokens: Vec<u32>,
}

/// Parses the text of a `config.json`.
pub fn parse_config(json: &str) -> Result<ModelInfo, Error> {
    let hf: HfConfig = serde_json::from_str(json).map_err(|source| Error::Json {
        path: "config.json".into(),
        source,
    })?;
    let unsupported = |what: String| Err(Error::Unsupported(what));
    if !hf.architectures.iter().any(|a| a == "LlamaForCausalLM") {
        return unsupported(format!("architectures {:?}", hf.architectures));
    }
    if hf.hidden_act != "silu" {
        return unsupported(format!("activation {:?}", hf.hidden_act));
    }
    if hf.rope_scaling.as_ref().is_some_and(|v| !v.is_null()) {
        return unsupported("rope_scaling (e.g. Llama 3.1's long-context RoPE)".into());
    }
    if hf.attention_bias || hf.mlp_bias {
        return unsupported("bias terms in attention or MLP".into());
    }
    let num_kv_heads = hf.num_key_value_heads.unwrap_or(hf.num_attention_heads);
    if num_kv_heads == 0 || !hf.num_attention_heads.is_multiple_of(num_kv_heads) {
        return unsupported(format!(
            "{} query heads cannot share {num_kv_heads} KV heads evenly",
            hf.num_attention_heads
        ));
    }
    let config = Config {
        vocab_size: hf.vocab_size,
        hidden_size: hf.hidden_size,
        intermediate_size: hf.intermediate_size,
        num_layers: hf.num_hidden_layers,
        num_heads: hf.num_attention_heads,
        num_kv_heads,
        head_dim: hf
            .head_dim
            .unwrap_or(hf.hidden_size / hf.num_attention_heads),
        rope_theta: hf.rope_theta,
        // Hugging Face's Llama code always rotates the two halves of each
        // head (chapter 12).
        rope_layout: RopeLayout::HalfSplit,
        rms_norm_eps: hf.rms_norm_eps,
        max_positions: hf.max_position_embeddings,
        tie_embeddings: hf.tie_word_embeddings,
    };
    let eos_tokens = match hf.eos_token_id {
        None => Vec::new(),
        Some(OneOrMany::One(id)) => vec![id],
        Some(OneOrMany::Many(ids)) => ids,
    };
    Ok(ModelInfo {
        config,
        bos_token: hf.bos_token_id,
        eos_tokens,
    })
}

/// Reads and parses `config.json`.
pub fn read_config(path: &Path) -> Result<ModelInfo, Error> {
    parse_config(&crate::read_to_string(path)?).map_err(|e| match e {
        Error::Json { source, .. } => Error::Json {
            path: path.to_owned(),
            source,
        },
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOLLM2: &str = r#"{
        "architectures": ["LlamaForCausalLM"], "bos_token_id": 1, "eos_token_id": 2,
        "hidden_act": "silu", "hidden_size": 576, "intermediate_size": 1536,
        "max_position_embeddings": 8192, "num_attention_heads": 9,
        "num_hidden_layers": 30, "num_key_value_heads": 3, "rms_norm_eps": 1e-05,
        "rope_scaling": null, "rope_theta": 100000, "tie_word_embeddings": true,
        "vocab_size": 49152, "attention_bias": false, "mlp_bias": false,
        "torch_dtype": "bfloat16"
    }"#;

    #[test]
    fn smollm2_config_matches_chapter_13() {
        let info = parse_config(SMOLLM2).unwrap();
        assert_eq!(info.config, Config::smollm2_135m());
        assert_eq!((info.bos_token, info.eos_tokens), (Some(1), vec![2]));
    }

    #[test]
    fn unsupported_features_are_rejected() {
        let scaled = SMOLLM2.replace(
            r#""rope_scaling": null"#,
            r#""rope_scaling": {"rope_type": "llama3"}"#,
        );
        assert!(matches!(parse_config(&scaled), Err(Error::Unsupported(_))));
        let gelu = SMOLLM2.replace("silu", "gelu");
        assert!(matches!(parse_config(&gelu), Err(Error::Unsupported(_))));
    }

    #[test]
    fn eos_can_be_a_list() {
        let many = SMOLLM2.replace(r#""eos_token_id": 2"#, r#""eos_token_id": [2, 0]"#);
        assert_eq!(parse_config(&many).unwrap().eos_tokens, [2, 0]);
    }
}
