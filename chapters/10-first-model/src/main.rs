//! Loads the trained classifier, checks it, then compares two ways of
//! serving a larger model.
//!
//! Run with:
//!   cargo run --release -p ch10-first-model --bin train   (once)
//!   cargo run --release -p ch10-first-model

use std::hint::black_box;
use std::time::{Duration, Instant};

use ch01_what_is_inference::summarize;
use ch07_threads::SpinPool;
use ch10_first_model::{Mlp, Workspace, accuracy, argmax, model_path, probabilities, spirals};

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    serve_the_trained_model();
    compare_serving_designs(cores);
}

/// Part 1: load, check accuracy, answer a few requests.
fn serve_the_trained_model() {
    let path = model_path();
    let start = Instant::now();
    let model = match Mlp::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("train the model first: cargo run --release -p ch10-first-model --bin train");
            std::process::exit(1);
        }
    };
    println!("== 1. the trained spiral classifier");
    println!("   loaded {} in {:.2?}", path.display(), start.elapsed());
    let widths: Vec<usize> = std::iter::once(model.input_dim())
        .chain(model.layers.iter().map(|l| l.out_dim))
        .collect();
    println!("   layers {widths:?}, {} parameters", model.params());

    let mut pool = SpinPool::new(1);
    let (test_x, test_y) = spirals(100, 3, 0.2, 99);
    println!(
        "   accuracy on 300 held-out points: {:.1}%",
        100.0 * accuracy(&model, &mut pool, &test_x, &test_y)
    );

    let mut ws = Workspace::new(&model, 1);
    for point in [[0.0f32, 0.0], [0.5, 0.1], [-0.3, 0.6], [0.2, -0.7]] {
        let mut logits = model.forward(&mut pool, &point, 1, &mut ws).to_vec();
        probabilities(&mut logits);
        println!(
            "   point {point:?} -> class {} with probabilities {:.3?}",
            argmax(&logits),
            logits
        );
    }
    println!();
}

/// Part 2: a model big enough for serving choices to matter.
fn compare_serving_designs(cores: usize) {
    let model = Mlp::random(&[1024, 4096, 4096, 1000], 1);
    let requests = 256;
    let inputs: Vec<f32> = (0..requests * model.input_dim())
        .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
        .collect();
    let dim = model.input_dim();
    println!(
        "== 2. serving a {} -> 4096 -> 4096 -> {} MLP ({:.1} M parameters, {:.0} MB) to {requests} requests",
        model.input_dim(),
        model.output_dim(),
        model.params() as f64 / 1e6,
        model.params() as f64 * 4.0 / 1e6
    );
    println!(
        "   design                               | total time | requests/s | per-request latency p50 / p99"
    );

    // A: one request at a time, one thread.
    let mut pool = SpinPool::new(1);
    let mut ws = Workspace::new(&model, 1);
    black_box(model.forward(&mut pool, &inputs[..dim], 1, &mut ws));
    let (total, mut lat) = one_at_a_time(&model, &mut pool, &mut ws, &inputs, dim);
    report(
        "one thread, one request at a time",
        total,
        requests,
        &mut lat,
    );

    // B: one request at a time, every layer split across all cores.
    let mut pool = SpinPool::new(cores);
    let (total, mut lat) = one_at_a_time(&model, &mut pool, &mut ws, &inputs, dim);
    report(
        &format!("{cores} threads split each request"),
        total,
        requests,
        &mut lat,
    );

    // C: several requests in flight, one per thread, each thread serial.
    // Scoped threads can borrow `model` directly: no Arc, no copy.
    let start = Instant::now();
    let per_thread = requests / cores;
    let mut lat: Vec<Duration> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..cores)
            .map(|t| {
                let model = &model;
                let mine = &inputs[t * per_thread * dim..(t + 1) * per_thread * dim];
                s.spawn(move || {
                    let mut pool = SpinPool::new(1);
                    let mut ws = Workspace::new(model, 1);
                    one_at_a_time(model, &mut pool, &mut ws, mine, dim).1
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker"))
            .collect()
    });
    report(
        &format!("{cores} threads, one request each"),
        start.elapsed(),
        requests,
        &mut lat,
    );

    // D: batches of requests, every layer split across all cores.
    for batch in [4, 16, 64, 256] {
        let mut ws = Workspace::new(&model, batch);
        black_box(model.forward(&mut pool, &inputs[..batch * dim], batch, &mut ws));
        let start = Instant::now();
        let mut lat = Vec::new();
        for chunk in inputs.chunks_exact(batch * dim) {
            let t = Instant::now();
            black_box(model.forward(&mut pool, chunk, batch, &mut ws));
            // Every request in the batch waits for the whole batch.
            lat.extend(std::iter::repeat_n(t.elapsed(), batch));
        }
        report(
            &format!("{cores} threads, batches of {batch}"),
            start.elapsed(),
            requests,
            &mut lat,
        );
    }
}

fn one_at_a_time(
    model: &Mlp,
    pool: &mut SpinPool,
    ws: &mut Workspace,
    inputs: &[f32],
    dim: usize,
) -> (Duration, Vec<Duration>) {
    let start = Instant::now();
    let mut lat = Vec::with_capacity(inputs.len() / dim);
    for x in inputs.chunks_exact(dim) {
        let t = Instant::now();
        black_box(model.forward(pool, x, 1, ws));
        lat.push(t.elapsed());
    }
    (start.elapsed(), lat)
}

fn report(label: &str, total: Duration, requests: usize, lat: &mut [Duration]) {
    let s = summarize(lat);
    println!(
        "   {label:<36} | {:>10} | {:>10.0} | {:>9} / {:>9}",
        format!("{total:.1?}"),
        requests as f64 / total.as_secs_f64(),
        format!("{:.2?}", s.p50),
        format!("{:.2?}", s.p99)
    );
}
