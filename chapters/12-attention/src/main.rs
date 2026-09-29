//! Looks inside attention: the weight matrix, RoPE's distance property, the
//! quadratic cost, and how grouped-query attention shrinks the KV cache.
//!
//! Run with: cargo run --release -p ch12-attention

use std::hint::black_box;
use std::time::Instant;

use ch06_simd::dot;
use ch12_attention::{
    Heads, Rope, RopeLayout, attention, attention_weights, kv_bytes_per_token, random_vec,
};

fn main() {
    weights_grid();
    rope_distance();
    quadratic_cost();
    kv_cache_sizes();
}

/// Part 1: causal attention weights for six tokens.
fn weights_grid() {
    let heads = Heads {
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 16,
    };
    let n = 6;
    let q = random_vec(n * 16, 1);
    let k = random_vec(n * 16, 2);
    let w = attention_weights(&q, &k, heads, 0, 0, true);
    println!("== 1. causal attention weights, 6 tokens (row = query, column = key)");
    for t in 0..n {
        let row: Vec<String> = w[t * n..(t + 1) * n]
            .iter()
            .map(|&p| {
                if p == 0.0 {
                    "  .  ".to_string()
                } else {
                    format!("{p:.2} ")
                }
            })
            .collect();
        println!("   token {t}: {}", row.join(""));
    }
    println!();
}

/// Part 2: RoPE scores depend on distance, not absolute position.
fn rope_distance() {
    let rope = Rope::new(64, 8192, 10_000.0, RopeLayout::HalfSplit);
    let q = random_vec(64, 3);
    let k = random_vec(64, 4);
    let score = |m: usize, n: usize| {
        let (mut qm, mut kn) = (q.clone(), k.clone());
        rope.apply(&mut qm, m);
        rope.apply(&mut kn, n);
        dot(&qm, &kn)
    };
    println!("== 2. RoPE: the score between the same query and key at different positions");
    for (m, n) in [(3, 0), (103, 100), (5003, 5000), (100, 0), (5100, 5000)] {
        println!(
            "   query at {m:>4}, key at {n:>4} (distance {:>3}): score {:+.5}",
            m - n,
            score(m, n)
        );
    }
    println!("   the same vector as query and key, at growing distance:");
    let self_score = |dist: usize| {
        let (mut a, mut b) = (q.clone(), q.clone());
        rope.apply(&mut a, dist);
        rope.apply(&mut b, 0);
        dot(&a, &b)
    };
    let line: Vec<String> = [0, 1, 4, 16, 64, 256, 1024, 4096]
        .iter()
        .map(|&d| format!("d={d}: {:+.2}", self_score(d)))
        .collect();
    println!("   {}", line.join("  "));
    println!();
}

/// Part 3: prefill attention for one layer of SmolLM2 grows with n².
fn quadratic_cost() {
    let heads = Heads {
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
    };
    println!(
        "== 3. causal attention for one SmolLM2 layer (9 query heads, 3 KV heads, d = 64), one core"
    );
    println!("   tokens |      time | x previous | score matrix if stored");
    let mut previous = None;
    for n in [256usize, 512, 1024, 2048, 4096] {
        let q = random_vec(n * 9 * 64, 5);
        let k = random_vec(n * 3 * 64, 6);
        let v = random_vec(n * 3 * 64, 7);
        let mut out = vec![0.0f32; q.len()];
        let mut scores = vec![0.0f32; n];
        let start = Instant::now();
        attention(&q, &k, &v, &mut out, heads, 0, true, &mut scores);
        black_box(&out);
        let t = start.elapsed();
        let ratio = previous.map_or(String::from("-"), |p: std::time::Duration| {
            format!("{:.1}", t.as_secs_f64() / p.as_secs_f64())
        });
        previous = Some(t);
        println!(
            "   {n:>6} | {:>9} | {ratio:>10} | {:>8.1} MB",
            format!("{t:.1?}"),
            (n * n * 9 * 4) as f64 / 1e6
        );
    }
    println!();
}

/// Part 4: KV cache bytes per token for MHA, GQA and MQA.
fn kv_cache_sizes() {
    println!("== 4. KV cache size (keys + values, all layers)");
    println!("   model                              | heads (q/kv) | per token | 8,192 tokens");
    let rows = [
        ("SmolLM2-135M as trained (GQA), f32", 30, 9, 3, 64, 4),
        ("SmolLM2-135M if it were MHA, f32", 30, 9, 9, 64, 4),
        ("8B, 32 layers, MHA, bf16", 32, 32, 32, 128, 2),
        ("8B, 32 layers, GQA (8 kv), bf16", 32, 32, 8, 128, 2),
        ("8B, 32 layers, MQA (1 kv), bf16", 32, 32, 1, 128, 2),
    ];
    for (name, layers, n_heads, n_kv_heads, head_dim, bytes) in rows {
        let heads = Heads {
            n_heads,
            n_kv_heads,
            head_dim,
        };
        let per_token = kv_bytes_per_token(layers, heads, bytes);
        println!(
            "   {name:<34} | {n_heads:>5}/{n_kv_heads:<6} | {:>6.1} KB | {:>8.2} GB",
            per_token as f64 / 1024.0,
            (per_token * 8192) as f64 / 1e9
        );
    }
}
