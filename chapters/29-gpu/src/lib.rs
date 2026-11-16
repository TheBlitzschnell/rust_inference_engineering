//! Chapter 29: GPUs.
//!
//! This machine has no GPU, so the chapter's kernels run in two forms:
//!
//! - [`sim`] and [`kernels`]: a small SIMT simulator (grids, blocks, warps,
//!   shared memory, barriers, shuffles) and the kernels of a forward pass
//!   written for it, checked against CPU results, with the memory traffic a
//!   GPU would see: sectors per warp load, bank conflicts, races.
//! - `oxide/main.rs`: the same kernels for NVIDIA's cuda-oxide, which
//!   compiles Rust to PTX. Not compiled or run here (see the README).
//!
//! [`roofline`] estimates what real GPUs could do, from their published
//! specifications.

pub mod kernels;
pub mod roofline;
pub mod sim;

pub use sim::{Global, Stats, launch};
