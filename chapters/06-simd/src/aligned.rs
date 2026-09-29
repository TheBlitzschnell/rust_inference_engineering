//! A fixed-length buffer whose first element sits on a 64-byte boundary.
//!
//! `Vec<f32>` only promises the alignment of `f32` (4 bytes); in practice
//! the allocator returns 16-byte-aligned memory. A 64-byte AVX-512 load
//! from an address that is not a multiple of 64 spans two cache lines and
//! costs about twice as much. Kernels that stream weights therefore want
//! their buffers aligned to the cache line.
//!
//! This is also a compact example of the unsafe code behind every
//! container: allocate raw memory, initialize it, hand out slices, and free
//! it exactly once.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// The alignment every `AlignedVec` guarantees: one cache line.
pub const ALIGN: usize = 64;

/// An owned, heap-allocated `[T]` aligned to [`ALIGN`] bytes.
///
/// Restricted to `T: Copy` so that elements never need dropping: freeing the
/// memory is all the cleanup there is.
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T: Copy> AlignedVec<T> {
    /// The allocation layout for `len` elements. We always allocate at least
    /// one element because allocating zero bytes is not allowed.
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len.max(1) * size_of::<T>(), ALIGN.max(align_of::<T>()))
            .expect("allocation size overflows isize")
    }

    /// Builds a buffer of `len` elements, element `i` being `f(i)`.
    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        assert!(size_of::<T>() > 0, "zero-sized types are not supported");
        let layout = Self::layout(len);
        // SAFETY: `layout` has a non-zero size (at least one element of a
        // non-zero-sized type), which is what `alloc` requires.
        let raw = unsafe { alloc(layout) }.cast::<T>();
        let Some(ptr) = NonNull::new(raw) else {
            handle_alloc_error(layout)
        };
        for i in 0..len {
            // SAFETY: `i < len`, so the write stays inside the allocation.
            // `write` does not read or drop the uninitialized old contents.
            // If `f` panics, the buffer leaks, which is safe (T: Copy has
            // nothing to drop).
            unsafe { ptr.as_ptr().add(i).write(f(i)) };
        }
        Self { ptr, len }
    }

    /// Copies a slice into a new aligned buffer.
    pub fn from_slice(src: &[T]) -> Self {
        Self::from_fn(src.len(), |i| src[i])
    }

    /// A buffer of `len` copies of `value`.
    pub fn filled(len: usize, value: T) -> Self {
        Self::from_fn(len, |_| value)
    }
}

impl<T: Copy> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized elements that live as
        // long as `self`, and the returned shared slice borrows `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`; `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated in `from_fn` with exactly this layout,
        // and `drop` runs at most once.
        unsafe { dealloc(self.ptr.as_ptr().cast(), Self::layout(self.len)) };
    }
}

impl<T: Copy> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// SAFETY: `AlignedVec<T>` owns its buffer exclusively, exactly like
// `Vec<T>`, so it can move to another thread when `T` can, and be shared
// between threads when `T` can. `NonNull` alone does not implement these
// traits, which is why we must state them.
unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
// SAFETY: see above; `&AlignedVec<T>` only hands out `&[T]`.
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_aligned_and_holds_its_values() {
        for len in [0, 1, 3, 16, 17, 1000] {
            let v = AlignedVec::from_fn(len, |i| i as u32 * 3);
            assert_eq!(v.as_ptr() as usize % ALIGN, 0);
            assert_eq!(v.len(), len);
            assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32 * 3));
        }
    }

    #[test]
    fn can_be_written_cloned_and_sent() {
        let mut v = AlignedVec::filled(10, 1.5f32);
        v[3] = 7.0;
        let c = v.clone();
        let handle = std::thread::spawn(move || c.iter().sum::<f32>());
        assert!((handle.join().unwrap() - (9.0 * 1.5 + 7.0)).abs() < 1e-6);
        assert!((v[3] - 7.0).abs() < f32::EPSILON);
        assert_eq!(format!("{:?}", AlignedVec::from_slice(&[1u8, 2])), "[1, 2]");
    }
}
