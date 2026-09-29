//! Tensor views in action, and what copying costs compared with not copying.
//!
//! Run with: cargo run --release -p ch03-tensors

use std::hint::black_box;
use std::time::Instant;

use ch03_tensors::{
    Tensor, TensorView, sum_column_order, sum_row_order, transpose_copy_naive, transpose_copy_tiled,
};

fn main() {
    show_views();
    measure_view_vs_copy();
    measure_access_order();
}

/// Part 1: the same six numbers seen through different views.
fn show_views() {
    let t = Tensor::arange(&[2, 3]);
    let v = t.view();
    println!("== 1. one buffer, many views");
    println!("   buffer:        {:?}", t.data());
    println!(
        "   as [2, 3]:     shape {:?} strides {:?} -> {v:?}",
        v.shape(),
        v.strides()
    );
    let tr = v.transpose(0, 1);
    println!(
        "   transposed:    shape {:?} strides {:?} -> {tr:?}",
        tr.shape(),
        tr.strides()
    );
    let col = v.select(1, 2);
    println!(
        "   column 2:      shape {:?} strides {:?} offset {} -> {col:?}",
        col.shape(),
        col.strides(),
        col.offset()
    );
    let bias = [100.0f32, 200.0, 300.0];
    let b = TensorView::new(&bias, &[3]).broadcast_to(&[2, 3]);
    println!(
        "   broadcast:     shape {:?} strides {:?} -> {b:?}",
        b.shape(),
        b.strides()
    );
    println!(
        "   reshape of the transpose possible? {}",
        tr.reshape(&[6]).is_some()
    );
    println!();
}

/// Part 2: a transposed *view* versus a transposed *copy* of a large matrix.
fn measure_view_vs_copy() {
    let n = 4096;
    let t = Tensor::arange(&[n, n]);
    let mut dst = vec![0.0f32; n * n];
    transpose_copy_naive(t.data(), n, n, &mut dst); // warm-up: fault in `dst`

    println!("== 2. transposing a {n} x {n} f32 matrix (64 MB)");
    let start = Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        let view = black_box(t.view()).transpose(0, 1);
        black_box(&view);
    }
    println!(
        "   view (strides swapped): {:>10.2?}",
        start.elapsed() / iters
    );

    let start = Instant::now();
    transpose_copy_naive(black_box(t.data()), n, n, &mut dst);
    black_box(&dst);
    println!("   naive copy:             {:>10.2?}", start.elapsed());

    for tile in [8, 32, 128] {
        let start = Instant::now();
        transpose_copy_tiled(black_box(t.data()), n, n, &mut dst, tile);
        black_box(&dst);
        println!(
            "   tiled copy (tile {tile:>3}):  {:>10.2?}",
            start.elapsed()
        );
    }
    println!();
}

/// Part 3: the same additions, in memory order versus jumping across rows.
fn measure_access_order() {
    println!("== 3. summing a matrix: row order vs column order");
    for n in [256, 1024, 4096] {
        let data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32).collect();
        black_box(sum_row_order(&data, n, n)); // warm-up

        let start = Instant::now();
        black_box(sum_row_order(black_box(&data), n, n));
        let rows = start.elapsed();
        let start = Instant::now();
        black_box(sum_column_order(black_box(&data), n, n));
        let cols = start.elapsed();
        println!(
            "   {n:>4} x {n:<4} ({:>5.1} MB): rows {rows:>9.2?}  columns {cols:>9.2?}  ({:.1}x slower)",
            (n * n * 4) as f64 / 1e6,
            cols.as_secs_f64() / rows.as_secs_f64()
        );
    }
}
