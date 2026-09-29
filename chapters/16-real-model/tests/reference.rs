//! Compares this engine with Hugging Face's, using the fixtures written by
//! `tools/make_fixtures.py`. Tests that need the downloaded model skip (and
//! say so) when it is absent; run `./tools/download_model.sh` first.

use ch07_threads::SpinPool;
use ch09_safetensors::SafeTensors;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch16_real_model::{Placement, Tokenizer, load_bf16, load_f32, model_dir};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

/// The model directory, or `None` (with a message) if it is not there.
fn model_or_skip(file: &str) -> Option<PathBuf> {
    let dir = model_dir();
    if dir.join(file).exists() {
        Some(dir)
    } else {
        eprintln!(
            "skipping: {} not found (run ./tools/download_model.sh)",
            dir.join(file).display()
        );
        None
    }
}

#[derive(Deserialize)]
struct Case {
    text: Option<String>,
    file: Option<String>,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Generation {
    question: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
}

fn generations() -> Vec<Generation> {
    serde_json::from_str(&std::fs::read_to_string(fixture("generation.json")).unwrap()).unwrap()
}

#[test]
fn tokenizer_matches_hugging_face_token_for_token() {
    let Some(dir) = model_or_skip("tokenizer.json") else {
        return;
    };
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let cases: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(fixture("tokenizer_cases.json")).unwrap())
            .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let missing = tok.missing_bytes();
    assert_eq!(missing.len(), 21, "SmolLM2 has no token for 21 bytes");
    for case in cases {
        let text = match (&case.text, &case.file) {
            (Some(t), _) => t.clone(),
            (None, Some(f)) => std::fs::read_to_string(root.join(f)).unwrap(),
            (None, None) => panic!("case without text"),
        };
        let ids = tok.encode(&text);
        assert_eq!(ids, case.ids, "encoding {:?}", &text[..text.len().min(60)]);
        // Decoding gives the text back, except for bytes the vocabulary
        // cannot represent, which were dropped.
        if !text.bytes().any(|b| missing.contains(&b)) {
            assert_eq!(tok.decode(&ids), text, "decoding back");
        }
    }
}

#[test]
fn the_chat_prompt_tokenizes_like_the_reference() {
    let Some(dir) = model_or_skip("tokenizer.json") else {
        return;
    };
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    for g in generations() {
        let prompt = ch16_real_model::chat_prompt(&[ch16_real_model::Message {
            role: "user",
            content: &g.question,
        }]);
        assert_eq!(tok.encode(&prompt), g.prompt_ids);
    }
}

fn prefill<W: Matrix>(model: &Model<W>, ids: &[u32]) -> Vec<f32> {
    let mut pool = SpinPool::new(4);
    let mut cache = KvCache::new(&model.config, 128);
    let mut scratch = Scratch::new(&model.config, 64, 128);
    model
        .forward_last(&mut pool, ids, &mut cache, &mut scratch)
        .to_vec()
}

#[test]
fn logits_match_pytorch() {
    let Some(dir) = model_or_skip("model.safetensors") else {
        return;
    };
    let bytes = std::fs::read(fixture("reference.safetensors")).unwrap();
    let reference = SafeTensors::parse(&bytes).unwrap();
    let want = reference
        .tensor("logits_last")
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let ids: Vec<u32> = serde_json::from_str(&reference.metadata()["prompt_ids"]).unwrap();

    let (model, _) = load_bf16(&dir, Placement::Mapped).unwrap();
    let got = prefill(&model, &ids);
    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let largest = want.iter().map(|x| x.abs()).fold(0.0, f32::max);
    eprintln!("largest |logit| {largest}, max difference {max_diff:e}");
    assert!(max_diff < 1e-3, "max difference {max_diff}");
    assert_eq!(ch14_kv_cache::argmax(&got), ch14_kv_cache::argmax(&want));
}

#[test]
fn mapped_aligned_and_f32_weights_agree() {
    let Some(dir) = model_or_skip("model.safetensors") else {
        return;
    };
    let ids = &generations()[0].prompt_ids;
    let mapped = prefill(&load_bf16(&dir, Placement::Mapped).unwrap().0, ids);
    let aligned = prefill(&load_bf16(&dir, Placement::Aligned).unwrap().0, ids);
    let f32 = prefill(&load_f32(&dir).unwrap().0, ids);
    // Same numbers, same kernel: bit for bit, whatever the alignment.
    assert_eq!(mapped, aligned);
    // Same numbers, a different kernel (summation order): nearly equal.
    let diff = mapped
        .iter()
        .zip(&f32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff < 1e-3, "bf16 and f32 paths differ by {diff}");
}

#[test]
fn greedy_generation_matches_pytorch() {
    let Some(dir) = model_or_skip("model.safetensors") else {
        return;
    };
    let (model, info) = load_bf16(&dir, Placement::Mapped).unwrap();
    let mut pool = SpinPool::new(4);
    let mut cache = KvCache::new(&model.config, 256);
    let mut scratch = Scratch::new(&model.config, 64, 256);
    for g in generations() {
        let mut sampler = ch15_sampling::Sampler::new(ch15_sampling::SamplingParams::greedy());
        let mut out = Vec::new();
        let reason = ch15_sampling::generate(
            &model,
            &mut pool,
            &g.prompt_ids,
            &mut sampler,
            &mut cache,
            &mut scratch,
            g.greedy_ids.len(),
            &info.eos_tokens,
            |t| {
                out.push(t);
                std::ops::ControlFlow::Continue(())
            },
        );
        // Hugging Face includes the stop token in its output; we report it
        // as the finish reason instead.
        let want: Vec<u32> = g
            .greedy_ids
            .iter()
            .copied()
            .take_while(|t| !info.eos_tokens.contains(t))
            .collect();
        assert_eq!(out, want, "{}", g.question);
        let stopped = g.greedy_ids.len() > want.len();
        assert_eq!(
            reason,
            if stopped {
                ch15_sampling::FinishReason::StopToken
            } else {
                ch15_sampling::FinishReason::Length
            }
        );
    }
}

#[test]
fn mapped_weights_are_where_the_file_puts_them() {
    let Some(dir) = model_or_skip("model.safetensors") else {
        return;
    };
    let misalignments = |placement| {
        let (model, _) = load_bf16(&dir, placement).unwrap();
        let mut all: Vec<usize> = model
            .layers
            .iter()
            .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down])
            .chain([&model.embed])
            .map(ch16_real_model::DenseBf16::misalignment)
            .collect();
        all.sort_unstable();
        all.dedup();
        all
    };
    // The file's data section starts 8 bytes past a multiple of 64 (chapter
    // 9), and every tensor size is a multiple of 64 bytes, so every mapped
    // matrix starts 8 bytes into a cache line. Copies start at 0.
    assert_eq!(misalignments(Placement::Mapped), [8]);
    assert_eq!(misalignments(Placement::Aligned), [0]);
}
