//! Quantized models against chapter 16's PyTorch fixtures. Skips when the
//! model is not downloaded.

use ch07_threads::SpinPool;
use ch09_safetensors::SafeTensors;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch16_real_model::{Placement, load_bf16, model_dir};
use ch17_profiling::map_matrices;
use ch18_int8::{Activations, Granularity, Q8Matrix};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Generation {
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
}

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../16-real-model/fixtures")
}

fn q8(granularity: Granularity, activations: Activations) -> Model<Q8Matrix> {
    let model = load_bf16(&model_dir(), Placement::Mapped).unwrap().0;
    map_matrices(model, |m| {
        let values: Vec<f32> = m.values().iter().map(|v| v.to_f32()).collect();
        Q8Matrix::new(&values, m.rows(), m.cols(), granularity, activations)
    })
}

fn greedy<W: Matrix>(model: &Model<W>, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut pool = SpinPool::new(4);
    let mut cache = KvCache::new(&model.config, 256);
    let mut scratch = Scratch::new(&model.config, 64, 256);
    ch14_kv_cache::generate_greedy(
        model,
        &mut pool,
        prompt,
        n,
        &mut cache,
        &mut scratch,
        |_, _| {},
    )[prompt.len()..]
        .to_vec()
}

#[test]
fn int8_models_stay_close_to_the_reference() {
    if !model_dir().join("model.safetensors").exists() {
        eprintln!("skipping: model not found (run ./tools/download_model.sh)");
        return;
    }
    let bytes = std::fs::read(fixtures().join("reference.safetensors")).unwrap();
    let reference = SafeTensors::parse(&bytes).unwrap();
    let want = reference
        .tensor("logits_last")
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let ids: Vec<u32> = serde_json::from_str(&reference.metadata()["prompt_ids"]).unwrap();
    let runs: Vec<Generation> =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("generation.json")).unwrap())
            .unwrap();

    for (granularity, activations) in [
        (Granularity::PerBlock, Activations::Float),
        (Granularity::PerBlock, Activations::Int8PerBlock),
    ] {
        let model = q8(granularity, activations);
        let mut pool = SpinPool::new(4);
        let mut cache = KvCache::new(&model.config, 128);
        let mut scratch = Scratch::new(&model.config, 64, 128);
        let got = model.forward_last(&mut pool, &ids, &mut cache, &mut scratch);
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("{activations:?}: largest logit difference {diff}");
        // Logits reach 31. Measured on the reference machine: the largest
        // single logit moves by 0.61 (W8A32) and 1.40 (W8A8); section 3.5
        // of the lesson measures quality properly, with perplexity and KL.
        assert!(diff < 2.0, "{activations:?}: {diff}");
        assert_eq!(ch14_kv_cache::argmax(got), ch14_kv_cache::argmax(&want));
        // The short answer ("The capital of France is Paris.") is unchanged.
        let first = &runs[0];
        let n = first.greedy_ids.len() - 1; // without the stop token
        assert_eq!(greedy(&model, &first.prompt_ids, n), first.greedy_ids[..n]);
    }
}
