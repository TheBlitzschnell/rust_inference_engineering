//! Maps out this machine's memory hierarchy and draws its roofline.
//!
//! Run with: cargo run --release -p ch04-memory
//! (takes about a minute; the largest buffers are 1 GB)

use std::time::Duration;

use ch04_memory::{
    Roofline, first_touch, linear_layer_intensity, load_latency_ns, parallel_read_bandwidth,
    peak_gflops_one_core, read_bandwidth, strided_pass, write_bandwidth,
};

const KB: usize = 1 << 10;
const MB: usize = 1 << 20;

/// A function that times one strided pass over a buffer.
type Probe = fn(&[f32]) -> Duration;

fn human(bytes: usize) -> String {
    if bytes >= MB {
        format!("{} MB", bytes / MB)
    } else {
        format!("{} KB", bytes / KB)
    }
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);

    println!("== 1. bandwidth of one core vs working-set size");
    println!("   size     |  read GB/s | write GB/s");
    let sizes = [
        16 * KB,
        32 * KB,
        128 * KB,
        512 * KB,
        MB,
        2 * MB,
        4 * MB,
        16 * MB,
        64 * MB,
        256 * MB,
        1024 * MB,
    ];
    for bytes in sizes {
        println!(
            "   {:>8} | {:>10.1} | {:>10.1}",
            human(bytes),
            read_bandwidth(bytes),
            write_bandwidth(bytes)
        );
    }

    println!("\n== 2. latency of one dependent load vs working-set size");
    for bytes in [
        16 * KB,
        128 * KB,
        MB,
        4 * MB,
        16 * MB,
        64 * MB,
        256 * MB,
        1024 * MB,
    ] {
        println!(
            "   {:>8} | {:>6.1} ns",
            human(bytes),
            load_latency_ns(bytes)
        );
    }

    println!("\n== 3. reading one float out of every k, over a 256 MB buffer");
    let data = vec![1.0f32; 256 * MB / 4];
    println!("   k (stride) | time for the whole pass | floats read | ns per float read");
    let probes: [(usize, Probe); 10] = [
        (1, strided_pass::<1>),
        (2, strided_pass::<2>),
        (4, strided_pass::<4>),
        (8, strided_pass::<8>),
        (16, strided_pass::<16>),
        (32, strided_pass::<32>),
        (64, strided_pass::<64>),
        (128, strided_pass::<128>),
        (256, strided_pass::<256>),
        (1024, strided_pass::<1024>),
    ];
    for (stride, probe) in probes {
        let pass = probe(&data);
        let reads = data.len() / stride;
        println!(
            "   {stride:>10} | {:>20.1} ms | {reads:>11} | {:>17.2}",
            pass.as_secs_f64() * 1e3,
            pass.as_secs_f64() * 1e9 / reads as f64
        );
    }
    drop(data);

    println!("\n== 4. first touch of freshly allocated memory (256 MB)");
    let (first, second) = first_touch(256 * MB);
    println!("   first write pass: {first:.1?}   second write pass: {second:.1?}");

    println!("\n== 5. read bandwidth of the whole machine (1 GB buffer)");
    let mut machine_bandwidth: f64 = 0.0;
    for t in (1..=threads).filter(|t| t.is_power_of_two() || *t == threads) {
        let bw = parallel_read_bandwidth(1024 * MB, t);
        machine_bandwidth = machine_bandwidth.max(bw);
        println!("   {t} thread(s): {bw:>6.1} GB/s");
    }

    println!("\n== 6. arithmetic throughput");
    let one_core = peak_gflops_one_core();
    let machine_peak = one_core * threads as f64;
    println!(
        "   one core:  {one_core:.1} GFLOP/s   ({threads} cores: about {machine_peak:.0} GFLOP/s)"
    );

    let roof = Roofline {
        peak_gflops: machine_peak,
        bandwidth_gbs: machine_bandwidth,
    };
    println!(
        "\n== 7. roofline for this machine: {:.0} GFLOP/s peak, {:.1} GB/s, ridge at {:.1} FLOP/byte",
        roof.peak_gflops,
        roof.bandwidth_gbs,
        roof.ridge_point()
    );
    println!("   kernel                                | FLOP/byte | best GFLOP/s | limited by");
    let kernels = [
        ("vector add c = a + b (f32)", 1.0 / 12.0),
        ("dot product (f32)", 0.25),
        (
            "linear layer, batch 1, f32 weights",
            linear_layer_intensity(4.0, 1),
        ),
        (
            "linear layer, batch 1, bf16 weights",
            linear_layer_intensity(2.0, 1),
        ),
        (
            "linear layer, batch 1, int8 weights",
            linear_layer_intensity(1.0, 1),
        ),
        (
            "linear layer, batch 1, 4-bit weights",
            linear_layer_intensity(0.5, 1),
        ),
        (
            "linear layer, batch 16, bf16 weights",
            linear_layer_intensity(2.0, 16),
        ),
        (
            "linear layer, batch 256, bf16 weights",
            linear_layer_intensity(2.0, 256),
        ),
        ("matmul 4096 x 4096 x 4096 (f32)", 4096.0 / 6.0),
    ];
    for (name, ai) in kernels {
        println!(
            "   {name:<37} | {ai:>9.2} | {:>12.1} | {}",
            roof.attainable(ai),
            if roof.is_memory_bound(ai) {
                "memory"
            } else {
                "compute"
            }
        );
    }
}
