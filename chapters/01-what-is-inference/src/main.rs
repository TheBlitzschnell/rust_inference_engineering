//! Measures the toy model three ways: latency of single requests, throughput
//! as the batch grows, and the cost of copying weights instead of borrowing.
//!
//! Run with: cargo run --release -p ch01-what-is-inference

use std::hint::black_box;
use std::time::Instant;

use ch01_what_is_inference::{LinearModel, predict_with_owned_model, summarize};

const IN_DIM: usize = 4096;
const OUT_DIM: usize = 4096;

fn main() {
    let model = LinearModel::new(IN_DIM, OUT_DIM);
    let mb = model.weight_bytes() as f64 / 1e6;
    println!("model: {OUT_DIM} x {IN_DIM} f32 weights = {mb:.1} MB\n");

    measure_latency(&model);
    measure_batching(&model);
    measure_borrow_vs_clone(&model);
}

/// Part 1: time 200 single requests and report the distribution.
fn measure_latency(model: &LinearModel) {
    let x = vec![0.5f32; model.in_dim()];
    let mut y = vec![0.0f32; model.out_dim()];

    // Warm-up: the first calls pay for page faults and cold caches.
    for _ in 0..10 {
        model.predict(black_box(&x), &mut y);
    }

    let mut samples = Vec::with_capacity(200);
    for _ in 0..200 {
        let start = Instant::now();
        model.predict(black_box(&x), &mut y);
        black_box(&y);
        samples.push(start.elapsed());
    }
    let s = summarize(&mut samples);
    println!("== 1. latency of one request ({} samples)", s.count);
    println!(
        "   mean {:?}  p50 {:?}  p90 {:?}  p99 {:?}  max {:?}",
        s.mean, s.p50, s.p90, s.p99, s.max
    );
    let secs = s.p50.as_secs_f64();
    println!(
        "   at p50: {:.1} GB/s of weights read, {:.2} GFLOP/s\n",
        model.weight_bytes() as f64 / secs / 1e9,
        model.flops_per_input() as f64 / secs / 1e9
    );
}

/// Part 2: throughput and latency as the batch size grows.
fn measure_batching(model: &LinearModel) {
    println!("== 2. batching");
    println!("   batch | time per batch | requests/s | GFLOP/s");
    for batch in [1usize, 2, 4, 8, 16, 32, 64] {
        let xs = vec![0.5f32; batch * model.in_dim()];
        let mut ys = vec![0.0f32; batch * model.out_dim()];
        model.predict_batch(black_box(&xs), batch, &mut ys); // warm-up

        let iters = 20;
        let start = Instant::now();
        for _ in 0..iters {
            model.predict_batch(black_box(&xs), batch, &mut ys);
            black_box(&ys);
        }
        let per_batch = start.elapsed() / iters;
        let secs = per_batch.as_secs_f64();
        println!(
            "   {batch:>5} | {:>14} | {:>10.0} | {:>7.2}",
            format!("{per_batch:.2?}"),
            batch as f64 / secs,
            (batch * model.flops_per_input()) as f64 / secs / 1e9
        );
    }
    println!();
}

/// Part 3: the same request, served by borrowing the model versus by
/// handing each request its own copy.
fn measure_borrow_vs_clone(model: &LinearModel) {
    let x = vec![0.5f32; model.in_dim()];
    let mut y = vec![0.0f32; model.out_dim()];
    let iters = 20;

    let start = Instant::now();
    for _ in 0..iters {
        model.predict(black_box(&x), &mut y);
    }
    let borrowed = start.elapsed() / iters;

    let start = Instant::now();
    for _ in 0..iters {
        predict_with_owned_model(model.clone(), black_box(&x), &mut y);
    }
    let cloned = start.elapsed() / iters;

    println!("== 3. borrowing vs cloning the weights");
    println!("   borrow (&model):       {borrowed:.2?} per request");
    println!("   clone  (model.clone()): {cloned:.2?} per request");
    println!(
        "   cloning costs {:.1}x the time, plus {:.1} MB of extra memory per in-flight request",
        cloned.as_secs_f64() / borrowed.as_secs_f64(),
        model.weight_bytes() as f64 / 1e6
    );
}
