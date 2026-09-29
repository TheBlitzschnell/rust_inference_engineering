//! Times every matmul kernel on the shapes that matter for inference.
//!
//! Run with: cargo run --release -p ch05-matmul

use std::hint::black_box;

use ch04_memory::{best_of, peak_gflops_one_core};
use ch05_matmul::{
    matmul_blocked, matmul_flops, matmul_ikj, matmul_naive, matmul_nt, matmul_nt_tiled,
    random_matrix, transpose,
};

/// An NN or NT kernel with the shared signature.
type Kernel = fn(&[f32], &[f32], &mut [f32], usize, usize, usize);

fn main() {
    let peak = peak_gflops_one_core();
    println!("single-core arithmetic peak measured by chapter 4's probe: {peak:.1} GFLOP/s\n");

    let shapes = [
        ("square", 512, 512, 512),
        ("square", 1024, 1024, 1024),
        ("decode: 1 token", 1, 4096, 4096),
        ("decode: batch of 16", 16, 4096, 4096),
        ("prefill: 128 tokens", 128, 2048, 2048),
    ];
    let kernels: [(&str, bool, Kernel); 7] = [
        ("naive (ijk)", false, matmul_naive),
        ("loop order ikj", false, matmul_ikj),
        ("blocked ikj", false, matmul_blocked),
        ("NT dot products", true, matmul_nt),
        ("NT tiled 1x4", true, matmul_nt_tiled::<1, 4>),
        ("NT tiled 2x4", true, matmul_nt_tiled::<2, 4>),
        ("NT tiled 4x4", true, matmul_nt_tiled::<4, 4>),
    ];

    for (label, m, k, n) in shapes {
        println!("== {label}: [{m} x {k}] x [{k} x {n}]");
        println!("   kernel            |      time | GFLOP/s | % of peak");
        let a = random_matrix(m * k, 1);
        let b = random_matrix(k * n, 2);
        let bt = transpose(&b, k, n); // weights stored as rows, like PyTorch
        let mut c = vec![0.0f32; m * n];
        let flops = matmul_flops(m, k, n);
        for (name, is_nt, kernel) in kernels {
            // The naive kernel takes minutes on the big shapes; skip it there.
            if name.starts_with("naive") && flops > 2e9 {
                continue;
            }
            let right = if is_nt { &bt } else { &b };
            kernel(&a, right, &mut c, m, k, n); // warm-up
            let t = best_of(3, || {
                kernel(black_box(&a), black_box(right), &mut c, m, k, n);
            });
            let gflops = flops / t.as_secs_f64() / 1e9;
            println!(
                "   {name:<17} | {:>9} | {gflops:>7.2} | {:>8.0}%",
                format!("{t:.2?}"),
                100.0 * gflops / peak
            );
        }
        println!();
    }
}
