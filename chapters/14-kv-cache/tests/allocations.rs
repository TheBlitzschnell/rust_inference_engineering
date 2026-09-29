//! Counts heap allocations during decode steps with a counting global
//! allocator. This file is its own test binary, so no other test allocates
//! at the same time.

use ch07_threads::SpinPool;
use ch13_transformer::Weights;
use ch14_kv_cache::{Config, KvCache, Model, Scratch, argmax};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Wraps the system allocator and counts every allocation.
struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator; the
// counters do not affect the memory handed out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller upholds `alloc`'s contract, which we pass on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn decode_steps_allocate_a_little_and_the_same_amount_every_step() {
    let config = Config::tiny();
    let model = Model::from_reference(&Weights::random(&config, 1));
    let mut pool = SpinPool::new(3);
    let mut cache = KvCache::new(&config, 128);
    let mut scratch = Scratch::new(&config, 16, 128);
    let prompt: Vec<u32> = (1..=20).collect();
    let mut next = argmax(model.forward_last(&mut pool, &prompt, &mut cache, &mut scratch));

    let mut per_step = Vec::new();
    for _ in 0..50 {
        let (count, bytes) = (
            ALLOCATIONS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        next = argmax(model.forward_last(&mut pool, &[next], &mut cache, &mut scratch));
        per_step.push((
            ALLOCATIONS.load(Ordering::Relaxed) - count,
            BYTES.load(Ordering::Relaxed) - bytes,
        ));
    }
    println!(
        "allocations per decode step (count, bytes): {:?}",
        per_step[0]
    );
    // The count does not grow with the context, and it is small: it is the
    // thread pool's list of chunks, one per parallel call.
    assert!(per_step.iter().all(|&step| step == per_step[0]));
    let parallel_calls = config.num_layers * 8 + 1;
    assert_eq!(per_step[0].0, parallel_calls);
}
