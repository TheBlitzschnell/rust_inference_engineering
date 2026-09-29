//! A pool of worker threads that *spin* while waiting for work.
//!
//! General-purpose pools (rayon, `std::thread::spawn`) put idle threads to
//! sleep and wake them through the operating system when work arrives. That
//! wake-up takes tens of microseconds on the reference machine, while one
//! matrix-vector product for a small model takes about the same. Inference
//! engines that run on CPUs (llama.cpp, for example) therefore keep their
//! workers spinning on a shared counter between operations: publishing new
//! work is then one atomic store, and every worker sees it within a fraction
//! of a microsecond. The price is CPU time burned while idle.
//!
//! # How it works
//!
//! - The pool owns `threads - 1` worker threads; the calling thread acts as
//!   worker 0, so all `threads` cores do useful work.
//! - [`SpinPool::run`] stores a pointer to the job, then increments
//!   `epoch`. Workers spin until they see a new epoch, run the job with their
//!   index, and decrement `remaining`. `run` spins until `remaining` is 0.
//!
//! # Why the `unsafe` is sound
//!
//! Workers are long-lived (`'static`) threads, but the job is a borrowed
//! closure that may reference local data (weights, activations). We erase
//! the closure's lifetime to hand it to the workers, exactly as
//! `std::thread::scope` does internally. This is sound because `run` does
//! not return, and does not unwind, until every worker has finished calling
//! the job: after that, no worker touches the pointer again. `run` takes
//! `&mut self`, so two jobs can never be in flight at once.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;

/// A job: called once per worker with the worker's index.
type Job<'a> = dyn Fn(usize) + Sync + 'a;

/// How many spins a waiting worker does before it starts yielding its time
/// slice to other threads. Spinning costs CPU; yielding costs latency.
#[cfg(not(miri))]
const SPINS_BEFORE_YIELD: u32 = 1 << 16;
/// Under Miri (which interprets every instruction, thousands of times more
/// slowly) spinning only wastes time; yielding at once checks the same
/// synchronization.
#[cfg(miri)]
const SPINS_BEFORE_YIELD: u32 = 0;

/// State shared between the pool handle and its workers.
struct Shared {
    /// Incremented once per job; workers wait for it to change.
    epoch: AtomicUsize,
    /// Workers (not counting the caller) still running the current job.
    remaining: AtomicUsize,
    /// Set when any worker's job panicked.
    panicked: AtomicBool,
    /// Tells workers to exit.
    stop: AtomicBool,
    /// The current job, with its lifetime erased. Written only by `run`
    /// while no worker is running; read by workers only after they observe
    /// the new epoch.
    job: UnsafeCell<Option<*const Job<'static>>>,
}

// SAFETY: `job` is only written by `run` while every worker is idle (between
// jobs), and only read by workers after an `Acquire` load of `epoch`
// synchronizes with the `Release` store that published it. All other fields
// are atomics.
unsafe impl Sync for Shared {}
// SAFETY: the raw pointer inside `job` refers to a `Sync` closure, which may
// be called from any thread.
unsafe impl Send for Shared {}

/// A fixed set of spinning worker threads. See the module documentation.
pub struct SpinPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl SpinPool {
    /// Creates a pool that runs jobs on `threads` threads in total: the
    /// caller plus `threads - 1` workers.
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            job: UnsafeCell::new(None),
        });
        let workers = (1..threads)
            .map(|index| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("spin-worker-{index}"))
                    .spawn(move || worker_loop(&shared, index))
                    .expect("failed to spawn worker thread")
            })
            .collect();
        Self { shared, workers }
    }

    /// A pool with one thread per available core.
    pub fn with_all_cores() -> Self {
        Self::new(std::thread::available_parallelism().map_or(1, usize::from))
    }

    /// Total number of threads that run each job, including the caller.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `job(i)` for every `i` in `0..threads()`, in parallel, and returns
    /// when all of them have finished. If any call panics, `run` panics too,
    /// after every worker has finished.
    #[expect(
        clippy::transmute_ptr_to_ptr,
        reason = "an `as` cast may not extend a trait object's lifetime; transmute states the intent"
    )]
    pub fn run(&mut self, job: &Job<'_>) {
        if self.workers.is_empty() {
            job(0);
            return;
        }
        // SAFETY: we only erase the lifetime here. The pointer is used by
        // workers strictly before `run` returns (see the module docs), so the
        // borrow it came from outlives every use.
        let erased: *const Job<'static> = unsafe {
            std::mem::transmute::<*const Job<'_>, *const Job<'static>>(std::ptr::from_ref(job))
        };
        // SAFETY: every worker is idle (the previous `run` waited for all of
        // them), so nobody reads `job` while we write it.
        unsafe { *self.shared.job.get() = Some(erased) };
        self.shared
            .remaining
            .store(self.workers.len(), Ordering::Relaxed);
        // Release: publishes the job pointer (and everything the caller wrote
        // before calling `run`) to workers that Acquire-load the new epoch.
        self.shared.epoch.fetch_add(1, Ordering::Release);

        // The caller is worker 0. Catch a panic so that we still wait for the
        // other workers before unwinding: returning early would leave them
        // holding a pointer into our stack frame.
        let own = catch_unwind(AssertUnwindSafe(|| job(0)));
        let mut spins = 0u32;
        // Acquire: makes every write the workers did during the job visible
        // to the caller once `remaining` reaches 0.
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            backoff(&mut spins);
        }
        // SAFETY: all workers have finished with the job; clear it so a stale
        // pointer is never left behind.
        unsafe { *self.shared.job.get() = None };
        if let Err(payload) = own {
            resume_unwind(payload);
        }
        assert!(
            !self.shared.panicked.swap(false, Ordering::Relaxed),
            "a SpinPool worker panicked"
        );
    }

    /// Splits `data` into one contiguous chunk per thread (each a multiple
    /// of `granule` elements, except possibly the last) and calls
    /// `f(start, chunk)` on each chunk in parallel, where `start` is the
    /// index of the chunk's first element in `data`.
    ///
    /// This is the safe way to let threads write disjoint parts of one
    /// output buffer: each `&mut` chunk is handed to exactly one thread.
    pub fn for_each_chunk_mut<T, F>(&mut self, data: &mut [T], granule: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        let granule = granule.max(1);
        let units = data.len().div_ceil(granule);
        let chunk = units.div_ceil(self.threads()).max(1) * granule;
        // One uncontended mutex per chunk: each worker locks only its own, so
        // this costs a few nanoseconds, and it lets safe code move a `&mut`
        // chunk into the thread that owns it.
        let chunks: Vec<Mutex<&mut [T]>> = data.chunks_mut(chunk).map(Mutex::new).collect();
        self.run(&|worker| {
            if let Some(slot) = chunks.get(worker) {
                let mut guard = slot.lock().expect("chunk lock poisoned");
                f(worker * chunk, &mut guard);
            }
        });
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for handle in self.workers.drain(..) {
            // A worker that panicked outside a job has nothing left to clean up.
            let _ = handle.join();
        }
    }
}

/// Spin briefly, then start yielding the CPU if the wait goes on.
fn backoff(spins: &mut u32) {
    if *spins < SPINS_BEFORE_YIELD {
        *spins += 1;
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

fn worker_loop(shared: &Shared, index: usize) {
    // Start from the epoch the pool was created with (0), *not* from
    // whatever `epoch` holds when this thread first runs: if `run` published
    // a job before this thread got scheduled, loading `epoch` here would
    // treat that job as already seen, and `run` would wait forever.
    let mut seen = 0;
    loop {
        let mut spins = 0u32;
        let epoch = loop {
            let now = shared.epoch.load(Ordering::Acquire);
            if now != seen {
                break now;
            }
            backoff(&mut spins);
        };
        seen = epoch;
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: the Acquire load of the new epoch synchronizes with the
        // Release increment in `run`, which happened after `job` was written.
        // `run` keeps the closure alive until we decrement `remaining`.
        let job = unsafe { (*shared.job.get()).expect("a job is published with every epoch") };
        // SAFETY: see above; the pointer is valid for the duration of the job.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(index) }));
        if result.is_err() {
            shared.panicked.store(true, Ordering::Relaxed);
        }
        shared.remaining.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_worker_runs_every_job() {
        let mut pool = SpinPool::new(4);
        let hits: Vec<AtomicU64> = (0..4).map(|_| AtomicU64::new(0)).collect();
        for _ in 0..50 {
            pool.run(&|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 50));
    }

    #[test]
    fn chunks_cover_the_buffer_exactly_once() {
        let mut pool = SpinPool::new(3);
        for len in [0, 1, 5, 64, 1001] {
            let mut data = vec![0u32; len];
            pool.for_each_chunk_mut(&mut data, 4, |start, chunk| {
                assert_eq!(start % 4, 0);
                for (k, v) in chunk.iter_mut().enumerate() {
                    *v += (start + k) as u32 + 1;
                }
            });
            assert!(data.iter().enumerate().all(|(i, &v)| v == i as u32 + 1));
        }
    }

    #[test]
    fn jobs_can_borrow_local_data() {
        let mut pool = SpinPool::new(2);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        pool.for_each_chunk_mut(&mut out, 1, |start, chunk| {
            for (k, o) in chunk.iter_mut().enumerate() {
                *o = input[start + k] * 10.0;
            }
        });
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn a_panicking_job_is_reported_and_the_pool_still_works() {
        let mut pool = SpinPool::new(3);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(&|i| assert!(i != 2, "worker 2 fails"));
        }));
        assert!(result.is_err());
        let count = AtomicU64::new(0);
        pool.run(&|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn single_thread_pool_runs_on_the_caller() {
        let mut pool = SpinPool::new(1);
        let caller = std::thread::current().id();
        pool.run(&|i| {
            assert_eq!(i, 0);
            assert_eq!(std::thread::current().id(), caller);
        });
    }
}
