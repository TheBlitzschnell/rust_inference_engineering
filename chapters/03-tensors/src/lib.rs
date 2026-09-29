//! Chapter 3: tensors as a flat buffer plus a shape and strides.
//!
//! A tensor is a block of numbers with a shape, like a 2 × 3 matrix or a
//! 32 × 128 × 64 cube of attention values. In memory it is always one flat,
//! contiguous run of numbers. The *strides* say how far to jump in that run
//! to move one step along each dimension. Changing the strides (and not the
//! data) is how transpose, slicing and broadcasting are done without copying.
//!
//! Two types:
//! - [`Tensor`] owns its buffer (`Vec<f32>`).
//! - [`TensorView`] borrows a buffer (`&'a [f32]`) and never copies it.
//!
//! # What the borrow checker prevents
//!
//! A view cannot outlive the tensor it points into. This does not compile
//! (error E0597, "`t` does not live long enough"):
//!
//! ```compile_fail,E0597
//! use ch03_tensors::Tensor;
//! let view;
//! {
//!     let t = Tensor::arange(&[2, 2]);
//!     view = t.view();
//! } // `t` is freed here, so `view` would point at freed memory
//! println!("{:?}", view.shape());
//! ```
//!
//! And the data cannot be changed while a view is reading it (error E0502,
//! "cannot borrow as mutable because it is also borrowed as immutable"):
//!
//! ```compile_fail,E0502
//! use ch03_tensors::Tensor;
//! let mut t = Tensor::arange(&[2, 2]);
//! let view = t.view();
//! t.data_mut()[0] = 42.0; // would change what `view` sees underneath it
//! println!("{}", view.get(&[0, 0]));
//! ```
//!
//! Moving the tensor is also blocked while a view exists (error E0505),
//! because moving could mean reallocating or freeing the buffer:
//!
//! ```compile_fail,E0505
//! use ch03_tensors::Tensor;
//! let t = Tensor::arange(&[2, 2]);
//! let view = t.view();
//! let buffer = t.into_vec(); // gives the buffer away while `view` uses it
//! println!("{} {}", view.get(&[0, 0]), buffer.len());
//! ```

use std::fmt;
use std::ops::Range;

/// Row-major strides for a shape: the last dimension is contiguous.
///
/// For shape `[2, 3, 4]` the strides are `[12, 4, 1]`: moving one step in
/// dimension 0 skips a whole 3 × 4 block (12 numbers), one step in dimension
/// 1 skips a row of 4, one step in dimension 2 moves to the next number.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut step = 1;
    for (stride, &dim) in strides.iter_mut().zip(shape).rev() {
        *stride = step;
        step *= dim;
    }
    strides
}

/// A tensor that owns its data.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Wraps an existing buffer. Panics if the length does not match the shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "buffer of {} numbers cannot have shape {shape:?}",
            data.len()
        );
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::from_vec(vec![0.0; shape.iter().product()], shape)
    }

    /// A tensor whose values are 0, 1, 2, ... in memory order. Handy for
    /// seeing where every element ends up after a view operation.
    pub fn arange(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_vec((0..n).map(|i| i as f32).collect(), shape)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Gives the buffer back, consuming the tensor. No copy.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Borrows the whole tensor as a view.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Changes the shape in place. Only the metadata changes; the numbers
    /// stay where they are. The element count must not change.
    pub fn reshape(&mut self, shape: &[usize]) {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.data.len(), "reshape must keep the element count");
        self.shape = shape.to_vec();
    }

    /// Splits a 2-D tensor into disjoint mutable row blocks, one per chunk of
    /// `rows_per_chunk` rows. Each block can be handed to a different thread
    /// (chapter 7) because the borrow checker can see they do not overlap.
    pub fn row_chunks_mut(&mut self, rows_per_chunk: usize) -> std::slice::ChunksMut<'_, f32> {
        assert_eq!(self.shape.len(), 2, "row_chunks_mut needs a 2-D tensor");
        let cols = self.shape[1];
        self.data.chunks_mut(rows_per_chunk * cols)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.view().fmt(f)
    }
}

/// A borrowed, possibly non-contiguous window onto someone else's buffer.
///
/// The lifetime `'a` ties the view to the buffer it points into: the
/// compiler will not let the buffer be freed or modified while a view of it
/// exists.
#[derive(Clone, PartialEq)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<'a> TensorView<'a> {
    /// Views a borrowed slice as a contiguous tensor of the given shape.
    pub fn new(data: &'a [f32], shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "slice length does not match shape");
        Self {
            data,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Where element `index` lives in the flat buffer:
    /// `offset + index[0]*strides[0] + index[1]*strides[1] + ...`
    pub fn position(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.rank(), "index has the wrong rank");
        let mut pos = self.offset;
        for ((&i, &dim), &stride) in index.iter().zip(&self.shape).zip(&self.strides) {
            assert!(
                i < dim,
                "index {index:?} out of bounds for shape {:?}",
                self.shape
            );
            pos += i * stride;
        }
        pos
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        self.data[self.position(index)]
    }

    /// Swaps two dimensions by swapping their sizes and strides. No data
    /// moves. The result is usually not contiguous.
    #[must_use]
    pub fn transpose(&self, a: usize, b: usize) -> TensorView<'a> {
        let mut view = self.clone();
        view.shape.swap(a, b);
        view.strides.swap(a, b);
        view
    }

    /// Keeps only `range` along dimension `dim`. No data moves: the offset
    /// moves to the first kept element and the size shrinks.
    #[must_use]
    pub fn slice(&self, dim: usize, range: Range<usize>) -> TensorView<'a> {
        assert!(range.start <= range.end && range.end <= self.shape[dim]);
        let mut view = self.clone();
        view.offset += range.start * self.strides[dim];
        view.shape[dim] = range.end - range.start;
        view
    }

    /// Picks index `i` along dimension `dim` and removes that dimension.
    /// `select(0, 3)` on a matrix gives row 3 as a 1-D view.
    #[must_use]
    pub fn select(&self, dim: usize, i: usize) -> TensorView<'a> {
        assert!(i < self.shape[dim]);
        let mut view = self.clone();
        view.offset += i * self.strides[dim];
        view.shape.remove(dim);
        view.strides.remove(dim);
        view
    }

    /// Repeats the tensor along dimensions of size 1 without copying, by
    /// giving those dimensions a stride of 0. Every step along a
    /// stride-0 dimension lands on the same numbers.
    #[must_use]
    pub fn broadcast_to(&self, shape: &[usize]) -> TensorView<'a> {
        assert!(
            shape.len() >= self.rank(),
            "cannot broadcast to fewer dimensions"
        );
        let extra = shape.len() - self.rank();
        let mut strides = vec![0; shape.len()];
        for (d, &target) in shape.iter().enumerate().skip(extra) {
            let (have, stride) = (self.shape[d - extra], self.strides[d - extra]);
            strides[d] = if have == target {
                stride
            } else {
                assert_eq!(have, 1, "cannot broadcast {:?} to {shape:?}", self.shape);
                0
            };
        }
        TensorView {
            data: self.data,
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        }
    }

    /// True if the elements, in row-major order, sit next to each other in
    /// memory with no gaps. Only contiguous views can be handed to fast
    /// kernels as a plain slice.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            // A dimension of size 1 is never stepped along, so its stride
            // does not matter.
            if dim != 1 && stride != expected {
                return false;
            }
            expected *= dim;
        }
        true
    }

    /// The underlying slice, if the view is contiguous. This is the
    /// zero-cost path to a kernel.
    pub fn as_slice(&self) -> Option<&'a [f32]> {
        self.is_contiguous()
            .then(|| &self.data[self.offset..self.offset + self.len()])
    }

    /// Reinterprets the shape without copying. Only possible when contiguous:
    /// a transposed matrix cannot be flattened without reordering its data.
    pub fn reshape(&self, shape: &[usize]) -> Option<TensorView<'a>> {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.len(), "reshape must keep the element count");
        self.as_slice().map(|s| TensorView::new(s, shape))
    }

    /// Visits every element in row-major order of the *view* (not of memory).
    pub fn for_each(&self, mut f: impl FnMut(f32)) {
        if let Some(slice) = self.as_slice() {
            slice.iter().copied().for_each(f);
            return;
        }
        if self.is_empty() {
            return;
        }
        let mut index = vec![0; self.rank()];
        loop {
            f(self.get(&index));
            // Advance the index like an odometer: bump the last digit, carry left.
            let mut d = self.rank();
            loop {
                if d == 0 {
                    return;
                }
                d -= 1;
                index[d] += 1;
                if index[d] < self.shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
    }

    /// Copies the view into a new contiguous tensor. This is the one place
    /// where a view operation costs memory and time, so it has a name that
    /// says so.
    pub fn to_contiguous(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.len());
        self.for_each(|x| data.push(x));
        Tensor::from_vec(data, &self.shape)
    }
}

impl fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut values = Vec::new();
        self.for_each(|x| values.push(x));
        write!(f, "Tensor(shape={:?}, values={values:?})", self.shape)
    }
}

/// Transposes a row-major matrix into a new buffer with a simple loop.
/// Reads are sequential, writes jump by `rows` each time.
pub fn transpose_copy_naive(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Same result, done in `tile × tile` blocks so that both the rows being
/// read and the rows being written stay in cache while a block is processed.
pub fn transpose_copy_tiled(src: &[f32], rows: usize, cols: usize, dst: &mut [f32], tile: usize) {
    assert_eq!(src.len(), rows * cols);
    assert_eq!(dst.len(), rows * cols);
    for r0 in (0..rows).step_by(tile) {
        for c0 in (0..cols).step_by(tile) {
            for r in r0..(r0 + tile).min(rows) {
                for c in c0..(c0 + tile).min(cols) {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Sums a row-major matrix in memory order: row by row.
pub fn sum_row_order(data: &[f32], rows: usize, cols: usize) -> f32 {
    assert_eq!(data.len(), rows * cols);
    let mut total = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            total += data[r * cols + c];
        }
    }
    total
}

/// Sums the same matrix column by column: every step jumps `cols` numbers
/// ahead in memory. Same additions (in a different order), very different
/// memory traffic.
pub fn sum_column_order(data: &[f32], rows: usize, cols: usize) -> f32 {
    assert_eq!(data.len(), rows * cols);
    let mut total = 0.0;
    for c in 0..cols {
        for r in 0..rows {
            total += data[r * cols + c];
        }
    }
    total
}

/// Sums a view by walking it through `get`, which works for any strides.
pub fn sum_strided(view: &TensorView<'_>) -> f32 {
    let mut total = 0.0;
    view.for_each(|x| total += x);
    total
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "views only move integers around, so values must match exactly"
)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert!(contiguous_strides(&[]).is_empty());
    }

    #[test]
    fn get_follows_the_stride_formula() {
        let t = Tensor::arange(&[2, 3, 4]);
        let v = t.view();
        assert_eq!(v.get(&[0, 0, 0]), 0.0);
        assert_eq!(v.get(&[0, 1, 2]), 6.0);
        assert_eq!(v.get(&[1, 2, 3]), 23.0);
    }

    #[test]
    fn transpose_swaps_indices_without_copying() {
        let t = Tensor::arange(&[2, 3]);
        let v = t.view();
        let tr = v.transpose(0, 1);
        assert_eq!(tr.shape(), &[3, 2]);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(tr.get(&[c, r]), v.get(&[r, c]));
            }
        }
        assert!(!tr.is_contiguous());
        assert!(tr.as_slice().is_none());
        // Same buffer: the view points into the tensor's memory.
        assert!(std::ptr::eq(tr.data.as_ptr(), t.data().as_ptr()));
    }

    #[test]
    fn slicing_moves_the_offset() {
        let t = Tensor::arange(&[4, 5]);
        let rows = t.view().slice(0, 1..3);
        assert_eq!(rows.shape(), &[2, 5]);
        assert_eq!(rows.offset(), 5);
        assert_eq!(rows.as_slice().unwrap(), &t.data()[5..15]);

        let cols = t.view().slice(1, 2..4);
        assert_eq!(cols.shape(), &[4, 2]);
        assert!(!cols.is_contiguous());
        assert_eq!(
            cols.to_contiguous().data(),
            &[2., 3., 7., 8., 12., 13., 17., 18.]
        );
    }

    #[test]
    fn select_picks_a_row_or_a_column() {
        let t = Tensor::arange(&[3, 4]);
        let row = t.view().select(0, 1);
        assert_eq!(row.as_slice().unwrap(), &[4., 5., 6., 7.]);
        let col = t.view().select(1, 2);
        assert_eq!(col.to_contiguous().data(), &[2., 6., 10.]);
    }

    #[test]
    fn broadcast_uses_zero_strides() {
        let bias = Tensor::from_vec(vec![10., 20., 30.], &[3]);
        let b = bias.view().broadcast_to(&[2, 3]);
        assert_eq!(b.strides(), &[0, 1]);
        assert_eq!(b.to_contiguous().data(), &[10., 20., 30., 10., 20., 30.]);

        // Adding a leading dimension of size 1 keeps the data contiguous.
        assert!(bias.view().broadcast_to(&[1, 3]).is_contiguous());

        let col = Tensor::from_vec(vec![1., 2.], &[2, 1]);
        let c = col.view().broadcast_to(&[2, 3]);
        assert_eq!(c.to_contiguous().data(), &[1., 1., 1., 2., 2., 2.]);
    }

    #[test]
    fn reshape_needs_contiguity() {
        let t = Tensor::arange(&[2, 6]);
        let r = t.view().reshape(&[3, 4]).unwrap();
        assert_eq!(r.get(&[2, 3]), 11.0);
        assert!(t.view().transpose(0, 1).reshape(&[12]).is_none());
    }

    #[test]
    fn transposes_agree() {
        let (rows, cols) = (37, 53);
        let src: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let mut a = vec![0.0; rows * cols];
        let mut b = vec![0.0; rows * cols];
        transpose_copy_naive(&src, rows, cols, &mut a);
        transpose_copy_tiled(&src, rows, cols, &mut b, 8);
        assert_eq!(a, b);
        let view = TensorView::new(&src, &[rows, cols]).transpose(0, 1);
        assert_eq!(view.to_contiguous().data(), &a[..]);
    }

    #[test]
    fn row_chunks_are_disjoint_and_cover_everything() {
        let mut t = Tensor::zeros(&[5, 3]);
        for (i, chunk) in t.row_chunks_mut(2).enumerate() {
            chunk.fill(i as f32);
        }
        assert_eq!(
            t.data(),
            &[0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 2., 2., 2.]
        );
    }
}
