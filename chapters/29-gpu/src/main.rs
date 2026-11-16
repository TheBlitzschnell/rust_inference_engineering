//! GPU kernels for a forward pass, run in the SIMT simulator on SmolLM2's
//! real weights, and roofline estimates for real GPUs.
//!
//! Run with: cargo run --release -p ch29-gpu [traffic|roofline]
//! (the traffic part needs the model: ./tools/download_model.sh)

use ch14_kv_cache::Matrix;
use ch16_real_model::{Placement, load_bf16, model_dir};
use ch29_gpu::kernels::{
    gemv_thread_per_row, gemv_warp_per_row, matmul_naive, matmul_tiled, rms_norm,
};
use ch29_gpu::roofline::{DEVICES, ModelSize};
use ch29_gpu::{Global, Stats};

fn main() {
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("traffic") {
        traffic();
    }
    if run("roofline") {
        roofline();
    }
}

fn kb(bytes: usize) -> String {
    format!("{:.0} KB", bytes as f64 / 1024.0)
}

fn row(name: &str, s: &Stats, err: f32) {
    println!(
        "   {name:<28} {:>9} {:>11} {:>11} {:>6.0}% {:>9} {:>5} {:>4} {:>9.1e}",
        s.loads,
        kb(s.requested_bytes),
        kb(s.fetched_bytes()),
        100.0 * s.load_efficiency(),
        if s.shared_instructions == 0 {
            "-".to_string()
        } else {
            format!(
                "{:.2}",
                s.shared_wavefronts as f64 / s.shared_instructions as f64
            )
        },
        s.barriers,
        s.races,
        err
    );
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Part 1: the kernels in the simulator, on layer 0 of SmolLM2-135M.
fn traffic() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    let model = load_bf16(&dir, Placement::Mapped).expect("model").0;
    let to_f32 = |w: &dyn Fn(usize, &mut [f32]), rows: usize, cols: usize| {
        let mut v = vec![0.0; rows * cols];
        for (r, row) in v.chunks_exact_mut(cols).enumerate() {
            w(r, row);
        }
        v
    };
    let layer = &model.layers[0];
    let (rows, cols) = (layer.w_up.rows(), layer.w_up.cols());
    let w = to_f32(&|r, out| layer.w_up.row_to_f32(r, out), rows, cols);
    // One token's hidden state (an embedding), normalized below.
    let mut x = vec![0.0; cols];
    model.embed.row_to_f32(260, &mut x);
    let norm = layer.mlp_norm.clone();
    println!("== 1. kernels in the SIMT simulator, on SmolLM2-135M's layer 0 (f32 values)");
    println!(
        "   kernel                           loads   requested     fetched  useful  wavefronts sync race max error"
    );
    println!(
        "                                                                          per access"
    );

    // RMSNorm of one token.
    let mut normed = vec![0.0; cols];
    let s = rms_norm(
        &Global::new(&x, 0),
        &Global::new(&norm, 1 << 30),
        &mut normed,
        1,
        cols,
        1e-5,
    );
    let want = ch08_operators::rms_norm_alloc(&x, &norm, 1e-5);
    row("RMSNorm, 1 × 576", &s, max_err(&normed, &want));

    // The MLP's up projection for that token: decode's matrix-vector product.
    let want: Vec<f32> = (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * normed[c]).sum())
        .collect();
    let (gw, gx) = (Global::new(&w, 0), Global::new(&normed, 1 << 30));
    let mut y = vec![0.0; rows];
    let s = gemv_thread_per_row(&gw, &gx, &mut y, rows, cols);
    row("W·x, 1536 × 576, thread/row", &s, max_err(&y, &want));
    let mut y = vec![0.0; rows];
    let s = gemv_warp_per_row(&gw, &gx, &mut y, rows, cols);
    row("W·x, 1536 × 576, warp/row", &s, max_err(&y, &want));

    // Prefill: 64 tokens through 256 of the up projection's rows.
    let (m, n) = (64, 256);
    let a: Vec<f32> = (0..m)
        .flat_map(|t| {
            let mut e = vec![0.0; cols];
            model.embed.row_to_f32(1000 + t * 37, &mut e);
            e
        })
        .collect();
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            (0..cols)
                .map(|j| a[i / n * cols + j] * w[i % n * cols + j])
                .sum()
        })
        .collect();
    let (ga, gb) = (Global::new(&a, 0), Global::new(&w[..n * cols], 1 << 30));
    let mut c = vec![0.0; m * n];
    let s = matmul_naive(&ga, &gb, &mut c, m, n, cols);
    row("A·Wᵀ, 64 × 256 × 576, naive", &s, max_err(&c, &want));
    for (name, transposed) in [("tiled", false), ("tiled, B transposed", true)] {
        let mut c = vec![0.0; m * n];
        let s = matmul_tiled(&ga, &gb, &mut c, m, n, cols, transposed);
        row(&format!("A·Wᵀ, {name}"), &s, max_err(&c, &want));
    }
    println!();
}

/// Part 2: decoding and prefill limits for real GPUs, from their
/// specifications (estimates, not measurements).
fn roofline() {
    println!("== 2. roofline estimates from published specifications (bf16 weights; not measured)");
    let models = [
        ModelSize {
            name: "SmolLM2-135M",
            params: 0.135e9,
        },
        ModelSize {
            name: "SmolLM2-360M",
            params: 0.362e9,
        },
        ModelSize {
            name: "8B",
            params: 8.0e9,
        },
        ModelSize {
            name: "70B",
            params: 70.6e9,
        },
    ];
    println!(
        "                       ridge   decode at batch 1, least time per token          compute-bound  prefill 1,000"
    );
    println!(
        "   device         (FLOP/byte)  SmolLM2-135M  SmolLM2-360M        8B       70B   from batch   tokens, 8B"
    );
    for d in DEVICES {
        let cells: Vec<String> = models
            .iter()
            .map(|m| {
                if d.fits(m) {
                    format!("{:.2} ms", d.decode_floor(m) * 1e3)
                } else {
                    "too big".into()
                }
            })
            .collect();
        println!(
            "   {:<18} {:>7.0} {:>13} {:>13} {:>9} {:>9} {:>12.0} {:>12}",
            d.name,
            d.ridge(),
            cells[0],
            cells[1],
            cells[2],
            cells[3],
            d.compute_bound_batch(),
            format!("{:.0} ms", d.prefill_floor(&models[2], 1000.0) * 1e3)
        );
    }
    println!();
}
