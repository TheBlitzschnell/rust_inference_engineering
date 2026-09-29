//! Chapter 5: matrix multiplication, the operation inference spends its time on.
//!
//! Two layouts appear in this crate:
//!
//! - **NN** ("normal-normal"): `C[m×n] = A[m×k] · B[k×n]`, both stored
//!   row-major. This is the textbook form.
//! - **NT** ("normal-transposed"): `Y[m×n] = X[m×k] · Wᵀ`, where `W` is
//!   stored as `n` rows of `k` numbers. This is how a linear layer is
//!   computed in inference: PyTorch stores a layer's weight as
//!   `[out_features, in_features]`, so each output's weights are one
//!   contiguous row, and every output is a dot product of two contiguous
//!   rows.
//!
//! All functions are single-threaded. Chapter 6 adds explicit SIMD and
//! chapter 7 adds threads.

/// Checks that three buffers have the sizes an `m×k` by `k×n` product needs.
fn check_shapes(a: &[f32], b: &[f32], c: &[f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "left operand must be m×k");
    assert_eq!(b.len(), k * n, "right operand must hold k×n numbers");
    assert_eq!(c.len(), m * n, "output must be m×n");
}

/// Dot product with eight running sums (see chapter 1 and chapter 6).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    let mut sums = [0.0f32; 8];
    for (x, y) in a8.iter().zip(b8) {
        for lane in 0..8 {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let mut total: f32 = sums.iter().sum();
    for (x, y) in a_rest.iter().zip(b_rest) {
        total += x * y;
    }
    total
}

/// NN, the textbook triple loop: `c[i][j] = Σ_p a[i][p] · b[p][j]`.
///
/// The innermost loop walks down a *column* of `b`, jumping `n` floats each
/// step: one useful float per cache line.
pub fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(a, b, c, m, k, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// NN with the two inner loops swapped ("i-k-j" order).
///
/// For each `a[i][p]`, add `a[i][p] × (row p of b)` to row `i` of `c`. Both
/// rows are contiguous, and the innermost loop has no dependency between
/// iterations, so the compiler turns it into vector instructions.
pub fn matmul_ikj(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(a, b, c, m, k, n);
    c.fill(0.0);
    for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (&a_ip, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_ip * b_pj;
            }
        }
    }
}

/// Block sizes for [`matmul_blocked`]. A `KC × NC` block of `b` is
/// 256 × 512 × 4 bytes = 512 KB: it stays in the 2 MB L2 cache while all
/// `m` rows of `a` stream past it.
pub const KC: usize = 256;
pub const NC: usize = 512;

/// NN, i-k-j order, with the `k` and `n` loops cut into blocks so the part
/// of `b` being used stays in cache.
///
/// Without blocking, each row of `a` walks through *all* of `b` (k×n
/// floats) before the next row starts. Once `b` is larger than the cache,
/// every row of `a` re-reads `b` from main memory. With blocking, we finish
/// all the work that touches one cache-sized block of `b` before moving on.
pub fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(a, b, c, m, k, n);
    c.fill(0.0);
    for p0 in (0..k).step_by(KC) {
        let p1 = (p0 + KC).min(k);
        for j0 in (0..n).step_by(NC) {
            let j1 = (j0 + NC).min(n);
            // The block b[p0..p1][j0..j1] is now reused by every row of a.
            for i in 0..m {
                let c_part = &mut c[i * n + j0..i * n + j1];
                for p in p0..p1 {
                    let a_ip = a[i * k + p];
                    let b_part = &b[p * n + j0..p * n + j1];
                    for (c_ij, &b_pj) in c_part.iter_mut().zip(b_part) {
                        *c_ij += a_ip * b_pj;
                    }
                }
            }
        }
    }
}

/// NT, the linear-layer form: `y[i][j] = dot(row i of x, row j of w)`.
///
/// The outer loop walks the weights, the inner loop the inputs, so each
/// weight row is fetched from memory once and reused for all `m` inputs
/// while it is in cache (the batching idea from chapter 1).
pub fn matmul_nt(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    check_shapes(x, w, y, m, k, n);
    for (j, w_row) in w.chunks_exact(k).enumerate() {
        for (i, x_row) in x.chunks_exact(k).enumerate() {
            y[i * n + j] = dot(x_row, w_row);
        }
    }
}

/// Matrix-vector product `y = W x`: the NT form with a single input row.
/// This is the shape of every linear layer during decode.
pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32]) {
    matmul_nt(x, w, y, 1, x.len(), y.len());
}

/// NT with register tiling: computes an `MR × NR` block of outputs at once.
///
/// A plain dot product loads two numbers for every multiply-add. This
/// micro-kernel loads `MR` numbers of `x` and `NR` numbers of `w` and does
/// `MR × NR` multiply-adds with them, all on values held in registers. With
/// `MR = 2, NR = 4`, that is 6 loads for 8 multiply-adds instead of 16 loads.
pub fn matmul_nt_tiled<const MR: usize, const NR: usize>(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    check_shapes(x, w, y, m, k, n);
    let m_full = m - m % MR;
    let n_full = n - n % NR;
    for j0 in (0..n_full).step_by(NR) {
        for i0 in (0..m_full).step_by(MR) {
            micro_kernel::<MR, NR>(x, w, y, i0, j0, k, n);
        }
    }
    // The edges that do not fill a whole tile: plain dot products.
    for i in 0..m {
        let cols = if i < m_full { n_full..n } else { 0..n };
        for j in cols {
            y[i * n + j] = dot(row(x, i, k), row(w, j, k));
        }
    }
}

/// Row `r` of a row-major matrix with `k` columns.
fn row(buf: &[f32], r: usize, k: usize) -> &[f32] {
    &buf[r * k..(r + 1) * k]
}

/// Computes `y[i0..i0+MR][j0..j0+NR]`, keeping `MR × NR × 8` partial sums.
#[expect(
    clippy::needless_range_loop,
    reason = "r and c index acc, x, w and y together; iterators would hide the tile structure"
)]
fn micro_kernel<const MR: usize, const NR: usize>(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    i0: usize,
    j0: usize,
    k: usize,
    n: usize,
) {
    let mut acc = [[[0.0f32; 8]; NR]; MR];
    let k8 = k - k % 8;
    for p in (0..k8).step_by(8) {
        let xs: [&[f32; 8]; MR] = std::array::from_fn(|r| {
            let start = (i0 + r) * k + p;
            x[start..start + 8].try_into().expect("8 floats")
        });
        for c in 0..NR {
            let start = (j0 + c) * k + p;
            let wv: &[f32; 8] = w[start..start + 8].try_into().expect("8 floats");
            for r in 0..MR {
                for lane in 0..8 {
                    acc[r][c][lane] += xs[r][lane] * wv[lane];
                }
            }
        }
    }
    for r in 0..MR {
        for c in 0..NR {
            let mut sum: f32 = acc[r][c].iter().sum();
            for p in k8..k {
                sum += x[(i0 + r) * k + p] * w[(j0 + c) * k + p];
            }
            y[(i0 + r) * n + j0 + c] = sum;
        }
    }
}

/// Returns `b` (k×n, row-major) transposed into n×k, so an NN problem can be
/// run with an NT kernel.
pub fn transpose(b: &[f32], k: usize, n: usize) -> Vec<f32> {
    assert_eq!(b.len(), k * n);
    let mut t = vec![0.0; k * n];
    for p in 0..k {
        for j in 0..n {
            t[j * k + p] = b[p * n + j];
        }
    }
    t
}

/// FLOPs in an m×k by k×n product: one multiply and one add per term.
pub fn matmul_flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * (m * k * n) as f64
}

/// A reference product computed in f64, for tests.
pub fn matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k)
                .map(|p| f64::from(a[i * k + p]) * f64::from(b[p * n + j]))
                .sum();
        }
    }
    c
}

/// Deterministic pseudo-random numbers in [-1, 1), for tests and demos.
pub fn random_matrix(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel must match the f64 reference within a tolerance that
    /// grows with the length of the sums (chapter 2, section 3.9).
    fn assert_close(label: &str, got: &[f32], want: &[f64], k: usize) {
        let tol = 1e-5 * (k as f64).sqrt() * 4.0 + 1e-6;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (f64::from(g) - w).abs() <= tol * w.abs().max(1.0),
                "{label}, element {i}: got {g}, want {w}"
            );
        }
    }

    type Kernel = fn(&[f32], &[f32], &mut [f32], usize, usize, usize);

    #[test]
    fn all_kernels_agree_with_the_reference() {
        let nn: [(&str, Kernel); 3] = [
            ("naive", matmul_naive),
            ("ikj", matmul_ikj),
            ("blocked", matmul_blocked),
        ];
        let nt: [(&str, Kernel); 4] = [
            ("nt", matmul_nt),
            ("nt 1x4", matmul_nt_tiled::<1, 4>),
            ("nt 2x4", matmul_nt_tiled::<2, 4>),
            ("nt 4x4", matmul_nt_tiled::<4, 4>),
        ];
        // Odd sizes on purpose: they exercise every edge path.
        for (m, k, n) in [
            (1, 1, 1),
            (3, 5, 7),
            (7, 33, 9),
            (8, 64, 8),
            (13, 300, 17),
            (2, 600, 530),
        ] {
            let a = random_matrix(m * k, 1);
            let b = random_matrix(k * n, 2);
            let bt = transpose(&b, k, n);
            let want = matmul_reference(&a, &b, m, k, n);
            let mut c = vec![0.0; m * n];
            for (name, f) in nn {
                c.fill(f32::NAN); // any element a kernel forgets stays NaN
                f(&a, &b, &mut c, m, k, n);
                assert_close(name, &c, &want, k);
            }
            for (name, f) in nt {
                c.fill(f32::NAN);
                f(&a, &bt, &mut c, m, k, n);
                assert_close(name, &c, &want, k);
            }
        }
    }

    #[test]
    fn matvec_is_nt_with_one_row() {
        let (k, n) = (37, 11);
        let w = random_matrix(n * k, 3);
        let x = random_matrix(k, 4);
        let mut y = vec![0.0; n];
        matvec(&w, &x, &mut y);
        let want = matmul_reference(&x, &transpose(&w, n, k), 1, k, n);
        assert_close("matvec", &y, &want, k);
    }

    #[test]
    #[should_panic(expected = "output must be m×n")]
    fn wrong_output_size_panics() {
        let mut c = vec![0.0; 5];
        matmul_ikj(&[0.0; 6], &[0.0; 6], &mut c, 2, 3, 2);
    }
}
