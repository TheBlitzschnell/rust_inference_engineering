//! The chapter's kernels for NVIDIA GPUs, written in Rust for cuda-oxide.
//!
//! STATUS: written against the cuda-oxide 0.1 documentation
//! (https://nvidia.github.io/cuda-rust/cuda-oxide/) and never compiled or
//! run for this course: the machine it was written on has no GPU, and
//! cuda-oxide needs its own pinned toolchain, LLVM 21 and CUDA 13. The
//! same kernels, line for line, run in this chapter's SIMT simulator
//! (`../src/kernels.rs`), where their results are checked against the CPU.
//! cuda-oxide is an alpha; expect to adapt names if its API has moved.
//!
//! To try it on an Ampere or newer GPU: install cargo-oxide (see the
//! chapter's README), run `cargo oxide new llm_kernels`, replace the new
//! project's `src/main.rs` with this file, and run `cargo oxide run`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};
use cuda_host::cuda_module;
use std::time::Instant;

/// Threads per block of the one-dimensional kernels, threads per warp, and
/// the matrix product's tile edge.
const BLOCK: usize = 256;
const WARP: usize = 32;
const TILE: usize = 16;

#[cuda_module]
mod kernels {
    use super::*;

    /// RMSNorm of each row of `x` (`n` values per row): one block per row.
    /// Each thread adds the squares of a strided slice of the row, the
    /// block adds the 256 partial sums in shared memory, and every thread
    /// scales its slice.
    #[kernel]
    pub fn rms_norm(x: &[f32], weight: &[f32], mut y: DisjointSlice<f32>, n: u32, eps: f32) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;
        let n = n as usize;
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let mut sum = 0.0f32;
        let mut c = tid;
        while c < n {
            let v = x[row * n + c];
            sum += v * v;
            c += BLOCK;
        }
        unsafe {
            PARTIAL[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = BLOCK / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PARTIAL[tid] += PARTIAL[tid + stride];
                }
            }
            // Outside the `if`: every thread of the block must reach it.
            thread::sync_threads();
            stride /= 2;
        }
        let total = unsafe { PARTIAL[0] };
        let scale = 1.0 / (total / n as f32 + eps).sqrt();
        let mut c = tid;
        while c < n {
            unsafe {
                *y.get_unchecked_mut(row * n + c) = x[row * n + c] * scale * weight[c];
            }
            c += BLOCK;
        }
    }

    /// `y = W·x` for `W` of `rows × cols`: one warp per row. Lane `l` adds
    /// up columns `l, l + 32, ...`, so each step of the warp reads 32
    /// consecutive floats; five shuffles add the 32 partial sums.
    #[kernel]
    pub fn gemv(w: &[f32], x: &[f32], mut y: DisjointSlice<f32>, rows: u32, cols: u32) {
        let (rows, cols) = (rows as usize, cols as usize);
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize * (BLOCK / WARP) + tid / WARP;
        let lane = tid % WARP;
        // `row` is the same for every lane of a warp, so the whole warp
        // takes this branch or none of it, as the shuffles require.
        if row < rows {
            let mut sum = 0.0f32;
            let mut c = lane;
            while c < cols {
                sum += w[row * cols + c] * x[c];
                c += WARP;
            }
            let mut offset = 16;
            while offset > 0 {
                sum += warp::shuffle_xor_f32(sum, offset);
                offset /= 2;
            }
            if lane == 0 {
                unsafe {
                    *y.get_unchecked_mut(row) = sum;
                }
            }
        }
    }

    /// `C = A·Bᵀ` (`A` of `m × k`, `B` of `n × k`) with 16 × 16 tiles in
    /// shared memory; blocks of 16 × 16 threads, one per output. `B`'s tile
    /// is stored transposed, so that the inner loop reads it without bank
    /// conflicts.
    #[kernel]
    pub fn matmul(a: &[f32], bt: &[f32], mut c: DisjointSlice<f32>, m: u32, n: u32, k: u32) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let (tx, ty) = (thread::threadIdx_x() as usize, thread::threadIdx_y() as usize);
        let (bx, by) = (thread::blockIdx_x() as usize, thread::blockIdx_y() as usize);
        let (row, col) = (by * TILE + ty, bx * TILE + tx);
        let mut acc = 0.0f32;
        let mut k0 = 0;
        while k0 < k {
            let i = k0 + tx;
            let va = if row < m && i < k { a[row * k + i] } else { 0.0 };
            let b_row = bx * TILE + ty;
            let vb = if b_row < n && i < k { bt[b_row * k + i] } else { 0.0 };
            unsafe {
                TILE_A[ty * TILE + tx] = va;
                TILE_B[tx * TILE + ty] = vb;
            }
            thread::sync_threads();
            for j in 0..TILE {
                acc += unsafe { TILE_A[ty * TILE + j] * TILE_B[j * TILE + tx] };
            }
            thread::sync_threads();
            k0 += TILE;
        }
        if row < m && col < n {
            unsafe {
                *c.get_unchecked_mut(row * n + col) = acc;
            }
        }
    }
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32 / 500.0) - 1.0)
        .collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this program owns the device code embedded for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };

    // Decode's matrix-vector product, at SmolLM2-135M's MLP shape.
    let (rows, cols) = (1536usize, 576usize);
    let (w, x) = (values(rows * cols, 1), values(cols, 2));
    let want: Vec<f32> = (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect();
    let w_dev = DeviceBuffer::from_host(&stream, &w)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(BLOCK / WARP) as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let start = Instant::now();
    for _ in 0..100 {
        // SAFETY: the grid covers every row, and `y_dev` has `rows` values.
        unsafe { module.gemv(&stream, config, &w_dev, &x_dev, &mut y_dev, rows as u32, cols as u32) }?;
    }
    let y = y_dev.to_host_vec(&stream)?; // waits for the launches to finish
    println!(
        "gemv {rows} x {cols}: {:.1} us per launch (with launch overhead), max error {:.1e}",
        start.elapsed().as_secs_f64() * 1e6 / 100.0,
        max_err(&y, &want)
    );

    // RMSNorm of 64 rows.
    let n = 576usize;
    let (x, weight) = (values(64 * n, 3), values(n, 4));
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let weight_dev = DeviceBuffer::from_host(&stream, &weight)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, 64 * n)?;
    let config = LaunchConfig {
        grid_dim: (64, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one block per row; every index written is below 64 · n.
    unsafe { module.rms_norm(&stream, config, &x_dev, &weight_dev, &mut y_dev, n as u32, 1e-5) }?;
    let y = y_dev.to_host_vec(&stream)?;
    let mut err = 0.0f32;
    for (row, out) in x.chunks_exact(n).zip(y.chunks_exact(n)) {
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-5).sqrt();
        let want: Vec<f32> = row.iter().zip(&weight).map(|(v, g)| v / rms * g).collect();
        err = err.max(max_err(out, &want));
    }
    println!("rms_norm 64 x {n}: max error {err:.1e}");

    // Prefill's matrix product: 64 tokens through the same projection.
    let m = 64usize;
    let a = values(m * cols, 5);
    let want: Vec<f32> = (0..m * rows)
        .map(|i| (0..cols).map(|j| a[i / rows * cols + j] * w[i % rows * cols + j]).sum())
        .collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, m * rows)?;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(TILE) as u32, m.div_ceil(TILE) as u32, 1),
        block_dim: (TILE as u32, TILE as u32, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the grid covers C, and only in-range outputs are written.
    unsafe { module.matmul(&stream, config, &a_dev, &w_dev, &mut c_dev, m as u32, rows as u32, cols as u32) }?;
    let c = c_dev.to_host_vec(&stream)?;
    println!("matmul {m} x {rows} x {cols}: max error {:.1e}", max_err(&c, &want));
    Ok(())
}
