//! Quantizes SmolLM2-135M to 4 bits and measures size, quality and speed.
//!
//! Run with: cargo run --release -p ch19-4bit [weights|quality|speed|answers]
//! (needs the model: ./tools/download_model.sh)

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch16_real_model::{
    DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16, model_dir,
};
use ch17_profiling::{Comparison, TiledBf16, compare, map_matrices};
use ch18_int8::eval::{Quality, log_probs};
use ch18_int8::quant::relative_error;
use ch18_int8::{Activations, BLOCK, Granularity, Q8Matrix};
use ch19_4bit::quant::dequantize;
use ch19_4bit::{BlockQ4, Q4Matrix, Recorder, Scheme, quantize, quantize_weighted};
use std::path::Path;

fn main() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    let needs_importance = run("weights") || run("quality") || run("answers");
    let importance = if needs_importance {
        calibrate(&dir)
    } else {
        Vec::new()
    };
    if run("weights") {
        weights(&dir, &importance);
    }
    if run("quality") {
        quality(&dir, &importance);
    }
    if run("speed") {
        speed(&dir);
    }
    if run("answers") {
        answers(&dir, &importance);
    }
}

/// The 4-bit variants compared: name, values per scale (0 = one per row),
/// scheme.
const VARIANTS: [(&str, usize, Scheme); 6] = [
    ("symmetric, per row", 0, Scheme::Symmetric),
    ("symmetric, 192", 192, Scheme::Symmetric),
    ("symmetric, 64", 64, Scheme::Symmetric),
    ("symmetric searched, 64", 64, Scheme::SymmetricSearch),
    ("min-max, 64", 64, Scheme::MinMax),
    ("min-max searched, 64", 64, Scheme::MinMaxSearch),
];

/// The importance-weighted variants (64 values per scale).
const WEIGHTED: [(&str, Scheme); 2] = [
    ("symmetric searched, 64, imp.", Scheme::SymmetricSearch),
    ("min-max searched, 64, imp.", Scheme::MinMaxSearch),
];

fn plain(dir: &Path) -> Model<DenseBf16> {
    load_bf16(dir, Placement::Mapped).expect("model").0
}

fn to_f32(m: &DenseBf16) -> Vec<f32> {
    m.values().iter().map(|v| v.to_f32()).collect()
}

fn q4(dir: &Path, group: usize, scheme: Scheme, int8_activations: bool) -> Model<Q4Matrix> {
    map_matrices(plain(dir), |m| {
        let g = if group == 0 { m.cols() } else { group };
        Q4Matrix::new(&to_f32(&m), m.rows(), m.cols(), g, scheme, int8_activations)
    })
}

/// A 4-bit model quantized with importance weights (`imp` from `calibrate`).
fn q4_weighted(dir: &Path, scheme: Scheme, imp: &[Vec<f32>]) -> Model<Q4Matrix> {
    let mut next = imp.iter();
    map_matrices(plain(dir), |m| {
        let importance = next.next().expect("one importance per matrix");
        Q4Matrix::new_weighted(
            &to_f32(&m),
            m.rows(),
            m.cols(),
            64,
            scheme,
            importance,
            true,
        )
    })
}

fn q8(dir: &Path) -> Model<Q8Matrix> {
    map_matrices(plain(dir), |m| {
        Q8Matrix::new(
            &to_f32(&m),
            m.rows(),
            m.cols(),
            Granularity::PerBlock,
            Activations::Int8PerBlock,
        )
    })
}

/// The text: evaluation uses tokens 0..2048, calibration 4096..6144, so no
/// model is judged on the text its importance was measured on.
fn text_tokens(dir: &Path) -> Vec<u32> {
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    tok.encode(include_str!(
        "../../15-sampling/data/pride-and-prejudice-1-6.txt"
    ))
}

/// Runs the bf16 model on calibration text with every matrix wrapped in a
/// `Recorder`, and returns each matrix's column importance, in the order
/// `map_matrices` visits the matrices (layer by layer, then the embedding).
fn calibrate(dir: &Path) -> Vec<Vec<f32>> {
    let tokens = text_tokens(dir);
    let model = map_matrices(plain(dir), Recorder::new);
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&model.config, 512);
    let mut scratch = Scratch::new(&model.config, 512, 512);
    for window in tokens[4096..6144].chunks_exact(512) {
        cache.clear();
        model.forward_all(&mut pool, window, &mut cache, &mut scratch);
    }
    let mut out = Vec::new();
    for l in &model.layers {
        for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
            out.push(m.importance());
        }
    }
    out.push(model.embed.importance());
    out
}

/// Part 1: weight error and size.
fn weights(dir: &Path, imp: &[Vec<f32>]) {
    let model = plain(dir);
    // Same order as `calibrate`: layer by layer, then the embedding.
    let mut matrices: Vec<&DenseBf16> = Vec::new();
    for l in &model.layers {
        matrices.extend([&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]);
    }
    matrices.push(&model.embed);
    let weights: usize = matrices.iter().map(|m| m.rows() * m.cols()).sum();
    println!("== 1. 4-bit weights: error and size ({weights} weights)");
    let q4_bytes = weights / BLOCK * size_of::<BlockQ4>();
    println!(
        "   stored as {} bytes per 64 weights = {:.1} bits per weight, {:.0} MB (bf16: {:.0} MB, int8 blocks: {:.0} MB)",
        size_of::<BlockQ4>(),
        8.0 * size_of::<BlockQ4>() as f64 / 64.0,
        q4_bytes as f64 / 1e6,
        (weights * 2) as f64 / 1e6,
        (weights / BLOCK * size_of::<ch18_int8::BlockQ8>()) as f64 / 1e6
    );
    for (name, group, scheme) in VARIANTS {
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for m in &matrices {
            let values = to_f32(m);
            let g = if group == 0 { m.cols() } else { group };
            let back = dequantize(&quantize(&values, m.rows(), m.cols(), g, scheme));
            let e = relative_error(&values, &back);
            let n: f64 = values.iter().map(|&v| f64::from(v).powi(2)).sum();
            err += e * e * n;
            norm += n;
        }
        println!(
            "   {name:<30} relative error {:>6.2}%",
            100.0 * (err / norm).sqrt()
        );
    }
    // The weighted variants: plain error, and the error weighted by
    // importance (what their search minimized), each against the same
    // scheme without importance.
    for (name, scheme) in WEIGHTED {
        let (mut err, mut plain_err, mut norm) = (0.0f64, 0.0f64, 0.0f64);
        let (mut werr, mut plain_werr, mut wnorm) = (0.0f64, 0.0f64, 0.0f64);
        for (m, importance) in matrices.iter().zip(imp) {
            let values = to_f32(m);
            let (rows, cols) = (m.rows(), m.cols());
            let with = dequantize(&quantize_weighted(
                &values, rows, cols, 64, scheme, importance,
            ));
            let without = dequantize(&quantize(&values, rows, cols, 64, scheme));
            for (j, ((&v, &a), &b)) in values.iter().zip(&with).zip(&without).enumerate() {
                let w = f64::from(importance[j % cols]);
                let (ea, eb, v2) = (
                    f64::from(v - a).powi(2),
                    f64::from(v - b).powi(2),
                    f64::from(v).powi(2),
                );
                err += ea;
                plain_err += eb;
                norm += v2;
                werr += w * ea;
                plain_werr += w * eb;
                wnorm += w * v2;
            }
        }
        println!(
            "   {name:<30} relative error {:>6.2}% (without importance {:.2}%); weighted by importance {:.2}% (without {:.2}%)",
            100.0 * (err / norm).sqrt(),
            100.0 * (plain_err / norm).sqrt(),
            100.0 * (werr / wnorm).sqrt(),
            100.0 * (plain_werr / wnorm).sqrt()
        );
    }
    println!();
}

/// Part 2: quality on 2,048 tokens of text against the bf16 model.
fn quality(dir: &Path, imp: &[Vec<f32>]) {
    let all = text_tokens(dir);
    let (window, windows) = (512, 4);
    let reference = map_matrices(plain(dir), TiledBf16);
    let int8 = q8(dir);
    let mut variants: Vec<(String, Model<Q4Matrix>)> = VARIANTS
        .iter()
        .map(|&(name, group, scheme)| (format!("W4A8 {name}"), q4(dir, group, scheme, true)))
        .collect();
    for (name, scheme) in WEIGHTED {
        variants.push((format!("W4A8 {name}"), q4_weighted(dir, scheme, imp)));
    }
    variants.push((
        "W4A32 min-max searched, 64".into(),
        q4(dir, 64, Scheme::MinMaxSearch, false),
    ));

    let mut pool = SpinPool::with_all_cores();
    let c = &reference.config;
    let mut cache = KvCache::new(c, window);
    let mut scratch = Scratch::new(c, window, window);
    let mut int8_quality = Quality::default();
    let mut results = vec![Quality::default(); variants.len()];
    for w in 0..windows {
        let tokens = &all[w * window..(w + 1) * window];
        let r = log_probs(&reference, &mut pool, tokens, &mut cache, &mut scratch);
        let lp = log_probs(&int8, &mut pool, tokens, &mut cache, &mut scratch);
        int8_quality.add(&r, &lp, tokens, c.vocab_size);
        for ((_, model), q) in variants.iter().zip(&mut results) {
            let lp = log_probs(model, &mut pool, tokens, &mut cache, &mut scratch);
            q.add(&r, &lp, tokens, c.vocab_size);
        }
    }
    println!(
        "== 2. quality on {} tokens of Pride and Prejudice (bf16 perplexity {:.3})",
        int8_quality.tokens,
        int8_quality.reference_perplexity()
    );
    println!(
        "   {:<38} {:>10} {:>11} {:>12}",
        "weights", "perplexity", "KL (nats)", "same top-1"
    );
    let row = |name: &str, q: &Quality| {
        println!(
            "   {name:<38} {:>10.3} {:>11.4} {:>11.1}%",
            q.candidate_perplexity(),
            q.mean_kl(),
            100.0 * q.top1_agreement()
        );
    };
    row("W8A8 per block (chapter 18)", &int8_quality);
    for ((name, _), q) in variants.iter().zip(&results) {
        row(name, q);
    }
    println!();
}

/// Part 3: decode speed, interleaved against bf16 and against int8.
fn speed(dir: &Path) {
    let bf16 = plain(dir);
    let int8 = q8(dir);
    let four = q4(dir, 64, Scheme::MinMaxSearch, true);
    println!("== 3. decode speed, 4 threads");
    println!(
        "   weights read per token: bf16 {:.0} MB, int8 {:.0} MB, 4-bit {:.0} MB",
        bf16.weight_bytes_per_token() as f64 / 1e6,
        int8.weight_bytes_per_token() as f64 / 1e6,
        four.weight_bytes_per_token() as f64 / 1e6
    );
    let show = |what: &str, c: &Comparison| {
        println!(
            "   {what}: {:.2?} -> {:.2?} per 16 tokens, speedup {:.2}x (80% of pairs: {:.2}-{:.2}x)",
            c.a.median, c.b.median, c.ratio, c.ratio_p10, c.ratio_p90
        );
    };
    show("bf16 -> 4-bit", &decode_ab(&bf16, &four));
    show("int8 -> 4-bit", &decode_ab(&int8, &four));
    println!();
}

fn decode_ab<A: Matrix, B: Matrix>(a: &Model<A>, b: &Model<B>) -> Comparison {
    let mut pool = SpinPool::with_all_cores();
    let c = &a.config;
    let (mut ca, mut sa) = (KvCache::new(c, 512), Scratch::new(c, 64, 512));
    let (mut cb, mut sb) = (KvCache::new(c, 512), Scratch::new(c, 64, 512));
    let p: Vec<u32> = (0..40).map(|i| i * 131 + 7).collect();
    a.forward_last(&mut pool, &p, &mut ca, &mut sa);
    b.forward_last(&mut pool, &p, &mut cb, &mut sb);
    compare(
        &mut pool,
        30,
        |pool| {
            if ca.len() > 440 {
                ca.truncate(40);
            }
            for i in 0..16u32 {
                a.forward_last(pool, &[(i * 37 + 100) % 49_000], &mut ca, &mut sa);
            }
        },
        |pool| {
            if cb.len() > 440 {
                cb.truncate(40);
            }
            for i in 0..16u32 {
                b.forward_last(pool, &[(i * 37 + 100) % 49_000], &mut cb, &mut sb);
            }
        },
    )
}

/// Part 4: the same question, answered greedily by each model.
fn answers(dir: &Path, imp: &[Vec<f32>]) {
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let prompt = tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: "Explain in two sentences why the sky is blue.",
    }]));
    println!("== 4. greedy answers to \"Explain in two sentences why the sky is blue.\"");
    println!("   bf16:\n     {}", answer(&plain(dir), &prompt, &tok));
    println!(
        "   int8 (W8A8 per block):\n     {}",
        answer(&q8(dir), &prompt, &tok)
    );
    for (name, group, scheme) in [VARIANTS[0], VARIANTS[5]] {
        println!(
            "   4-bit {name}:\n     {}",
            answer(&q4(dir, group, scheme, true), &prompt, &tok)
        );
    }
    let (name, scheme) = WEIGHTED[1];
    println!(
        "   4-bit {name}:\n     {}",
        answer(&q4_weighted(dir, scheme, imp), &prompt, &tok)
    );
}

fn answer<W: Matrix>(model: &Model<W>, prompt: &[u32], tok: &Tokenizer) -> String {
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&model.config, 256);
    let mut scratch = Scratch::new(&model.config, 64, 256);
    let out = ch14_kv_cache::generate_greedy(
        model,
        &mut pool,
        prompt,
        60,
        &mut cache,
        &mut scratch,
        |_, _| {},
    );
    // Stop at <|im_end|> (id 2), as a chat front end would.
    let generated: Vec<u32> = out[prompt.len()..]
        .iter()
        .copied()
        .take_while(|&t| t != 2)
        .collect();
    tok.decode(&generated).replace('\n', " ")
}
