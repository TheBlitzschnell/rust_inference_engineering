//! The faster matrix types must compute the same model: checked against
//! chapter 16's PyTorch fixture. Skips when the model is not downloaded.

use ch07_threads::SpinPool;
use ch09_safetensors::SafeTensors;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch16_real_model::{Placement, load_bf16, model_dir};
use ch17_profiling::{FusedBf16, Rows4Bf16, TiledBf16, map_matrices};
use std::path::Path;

fn logits<W: Matrix>(model: &Model<W>, ids: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut pool = SpinPool::new(4);
    let mut cache = KvCache::new(&model.config, 128);
    let mut scratch = Scratch::new(&model.config, 64, 128);
    let prefill = model
        .forward_last(&mut pool, ids, &mut cache, &mut scratch)
        .to_vec();
    let decode = model
        .forward_last(&mut pool, &[504], &mut cache, &mut scratch)
        .to_vec();
    (prefill, decode)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn faster_matrices_compute_the_same_model() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: model not found (run ./tools/download_model.sh)");
        return;
    }
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../16-real-model/fixtures/reference.safetensors");
    let bytes = std::fs::read(fixture).unwrap();
    let reference = SafeTensors::parse(&bytes).unwrap();
    let want = reference
        .tensor("logits_last")
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let ids: Vec<u32> = serde_json::from_str(&reference.metadata()["prompt_ids"]).unwrap();
    let load = || load_bf16(&dir, Placement::Mapped).unwrap().0;

    let (plain_prefill, plain_decode) = logits(&load(), &ids);
    let (tiled_prefill, tiled_decode) = logits(&map_matrices(load(), TiledBf16), &ids);
    let (fused_prefill, fused_decode) = logits(&map_matrices(load(), FusedBf16), &ids);
    let (rows4_prefill, _) = logits(&map_matrices(load(), Rows4Bf16), &ids);

    // Prefill: every version agrees with PyTorch as closely as chapter 16's.
    for (name, got) in [
        ("tiled", &tiled_prefill),
        ("fused", &fused_prefill),
        ("rows4", &rows4_prefill),
    ] {
        let d = max_diff(got, &want);
        assert!(d < 1e-3, "{name}: {d}");
    }
    // Decode after the prefill: the fused pass does exactly the plain
    // model's arithmetic, so it agrees bit for bit. The tiled model's decode
    // path is the plain one too, but its cache was filled by the tiled
    // prefill, which adds in a different order: close, not identical.
    assert!(fused_decode == plain_decode, "fused decode differs");
    let d = max_diff(&tiled_decode, &plain_decode);
    assert!(d < 1e-3, "tiled decode: {d}");
    assert!(max_diff(&plain_prefill, &want) < 1e-3);
}
