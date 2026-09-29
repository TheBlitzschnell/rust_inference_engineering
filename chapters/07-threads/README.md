# Chapter 7: Threads

> **In one sentence:** splitting a matrix-vector product across cores multiplies the memory bandwidth you can use, but only if the cost of coordinating the threads is small compared with the work, which for small models means keeping worker threads awake and spinning.

**Where this fits:** chapter 4 showed that one core pulls about 11 GB/s from memory while four cores pull 30-45 GB/s. Chapter 6 made each core's kernel fast. This chapter puts all the cores to work. The spin pool built here is the thread pool the engine uses from chapter 14 on.

**You need:** chapters 4 and 6. Basic familiarity with Rust closures.

**You will build:** three ways to run a parallel matvec (spawning threads per call, rayon's pool, and a custom spinning pool); a parallel `Y = X·Wᵀ` for many input rows; a false-sharing experiment; and measurements of speed-up, overhead and bandwidth at sizes from SmolLM2's layers up to 256 MB.

---

## 1. The intuition

A restaurant kitchen gets a big order: 400 plates. One cook would take all evening, so the head chef splits the order among four cooks. Two things decide whether that helps.

First, **the pantry door**. All four cooks fetch ingredients through the same door. If one cook alone already keeps the door busy, four cooks just queue at it. Here the door is wide (memory bandwidth grows with cores up to a point), so four cooks do get more through.

Second, **calling the cooks over**. If the cooks are on a break in the staff room, the head chef has to walk over and call them for every order. For a 400-plate order that walk is nothing. For a 3-plate order, the walk takes longer than cooking the plates alone. The fix is to have the cooks stand at their stations, watching the ticket rail: a new ticket appears and they start immediately. They are "spinning": burning energy standing there, but ready instantly.

**Where the analogy breaks:** cooks cannot accidentally grab the same plate. Threads can accidentally write the same memory, which is a **data race**: the result depends on timing and is undefined in Rust's model. Rust's type system makes it impossible to write a data race in safe code, and most of this chapter's Rust is about how we give each thread its own part of the output so the compiler can see that no two threads overlap.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Thread** | An independent sequence of execution, scheduled by the OS onto a core. |
| **Thread pool** | A set of long-lived threads that run submitted tasks, so threads are not created per task. |
| **Work stealing** | A scheduling strategy where idle threads take tasks from busy threads' queues (rayon). |
| **Spinning / busy-waiting** | A waiting thread repeatedly checks a condition instead of sleeping. |
| **Data race** | Two threads access the same memory, at least one writes, with no synchronization. |
| **`Send`** | A type that may be moved to another thread. |
| **`Sync`** | A type that may be shared (by `&` reference) between threads. |
| **Scoped thread** | A thread guaranteed to finish before a scope ends, so it can borrow local data. |
| **Atomic** | A value whose operations cannot be interrupted halfway; the basis of lock-free coordination. |
| **Memory ordering** | Rules (`Relaxed`, `Acquire`, `Release`...) for when one thread's writes become visible to another. |
| **False sharing** | Threads writing *different* variables that share one cache line, forcing the line to bounce between cores. |
| **Speed-up** | Time with one thread ÷ time with N threads. |
| **Amdahl's law** | If a fraction s of the work is serial, speed-up can never exceed 1/s. |

## 3. The concepts in depth

### 3.1 Why threads help a memory-bound kernel

A matvec does very little arithmetic per byte, so adding cores does not help by adding arithmetic. It helps because each core can only have a limited number of memory requests in flight (chapter 4, section 3.3), and each additional core brings its own. Part 3 of the demo, on a 256 MB `f32` matrix:

```text
   threads | matvec f32 256 MB
         1 |  29.3ms   9.2 GB/s
         2 |  12.8ms  21.0 GB/s
         3 |   9.8ms  27.4 GB/s
         4 |   6.9ms  39.0 GB/s
```

Bandwidth scales roughly linearly with cores here, up to about 39 GB/s: right at the machine bandwidth chapter 4 measured. One thread uses a quarter of it.

For decode, this translates directly into tokens per second: SmolLM2 in `bf16` is 270 MB of weights, so the ceiling moves from about 40 tokens/s (one core) to about 140 (four cores).

### 3.2 Splitting the work without data races

In `y = W x`, output `y[j]` depends only on row `j` of `W` (and on `x`, which every thread only reads). So the natural split is by rows: thread 0 computes the first quarter of `y` from the first quarter of `W`'s rows, thread 1 the second quarter, and so on.

In C, you would pass every thread the same `y` pointer and trust yourself to only write your own range. In Rust, `y.chunks_mut(n)` produces separate `&mut [f32]` slices that are known not to overlap, and each thread receives its own slice. The compiler can then check that no thread writes anything outside its piece, and a data race cannot be written in safe code.

For a batch of inputs (`m > 1`, chapter 5's NT layout), splitting by weight rows means each thread produces a *column block* of `Y`, which is not contiguous in memory (the outputs for one weight row are `m` elements a whole row apart). Rather than reach for `unsafe`, we let each thread write into its own contiguous rows of a **transposed** temporary buffer `Yᵀ`, and transpose once at the end. The transpose is O(m·n), tiny next to the O(m·n·k) of the product. Choosing a layout for intermediate results so that parallel writes are contiguous, disjoint slices is a standard trick in parallel code.

### 3.3 Three ways to run a parallel matvec

**1. Spawn threads per call** (`std::thread::scope`). Simple and safe: the scope lets threads borrow `W`, `x` and `y` directly because it guarantees they finish before the borrows end. But creating and joining an OS thread involves the kernel. Measured here: **154 µs** to spawn and join 4 threads that do nothing.

**2. A pool with sleeping workers** (rayon). The threads already exist; tasks are pushed to queues, and idle workers *steal* work from busy ones, which balances uneven work automatically. When there is nothing to do, rayon's workers go to sleep, and a new parallel call must wake them through the OS. Measured here: **49 µs** per parallel call with no work. On a desktop machine the wake-up is typically faster (single-digit µs); in this cloud VM, where waking a sleeping virtual CPU goes through the hypervisor, it is slow. Either way it is not free.

**3. A pool with spinning workers** (`SpinPool`, this chapter). Workers never sleep between jobs: they loop, checking a shared counter. Publishing a job is one atomic increment, and every worker notices within a fraction of a microsecond. Measured: **1.3 µs** per parallel call. The price is that idle workers keep their cores busy, burning CPU and power while doing nothing.

Why this matters so much for inference: part 2 of the demo times one matvec at SmolLM2's sizes.

```text
   weights              |   serial | spawned threads |    rayon | spin pool
    576x576  (  1.3 MB) |   20.9µs |         230.7µs |   44.7µs |     9.6µs
   1536x576  (  3.5 MB) |  173.0µs |         279.8µs |  103.9µs |    21.2µs
```

A 576 × 576 matvec takes 21 µs on one core. Spawning threads makes it 11x *slower*. Rayon makes it 2x slower. Only the spin pool makes it faster (2.2x). SmolLM2 runs about 211 of these small matvecs per token (7 per layer × 30 layers, plus the output layer), so the per-call overhead decides whether multithreading helps at all. This is why CPU inference engines such as llama.cpp keep their worker threads spinning during generation.

### 3.4 Super-linear speed-up: four L2 caches

The 1536 × 576 row above went from 173 µs to 21 µs with 4 threads: 8x faster on 4 cores. That is not a measurement error. The matrix is 3.5 MB, more than one core's 2 MB L2 cache, so one core streams it from L3 at L3 speed. Split four ways, each core's 0.9 MB share fits in *its own* L2, and the demo calls the same matvec repeatedly, so each core finds its share already in L2. Four cores bring four L2 caches: 8 MB of fast storage instead of 2.

This happens in real inference too: a small model's layer weights, split across cores, can live in the cores' private caches between tokens. It is also a warning about benchmarking: repeated calls on the same small matrix can look far better than a real forward pass, where 30 layers' weights compete for those caches.

### 3.5 Compute-bound work scales too

The prefill-shaped matmul (128 tokens × 2048 × 2048) is compute-bound, and it scales close to linearly:

```text
   threads | matmul f32 128x2048x2048
         1 |    30.9ms   34.8 GFLOP/s
         2 |    19.6ms   54.8 GFLOP/s
         3 |    14.4ms   74.8 GFLOP/s
         4 |    10.3ms  104.4 GFLOP/s
```

3x on 4 cores. Compute-bound work has no shared bottleneck like the memory bus; the gap to a perfect 4x comes from the serial transpose at the end, synchronization, and the VM's noise.

### 3.6 An honest gap: `bf16` is not twice as fast

Part 3 also runs the same matvec with `bf16` weights, which are half the bytes:

```text
   threads | matvec f32 256 MB  | matvec bf16 128 MB
         1 |  29.3ms   9.2 GB/s |  22.6ms   5.9 GB/s
         4 |   6.9ms  39.0 GB/s |   5.5ms  24.5 GB/s
```

If the kernel were purely memory-bound, halving the bytes would halve the time. It only saves 20-25%: the `bf16` kernel is not reaching the memory roofline. Something other than DRAM bandwidth is limiting it. Chapter 17 uses profiling to find out what, and fixes it; for now, note that "memory-bound" is a hypothesis to verify, not something to assume.

### 3.7 False sharing

Part 4 gives each of 4 threads its own counter to increment 20 million times. No two threads touch the same counter, so there is no data race. But when the counters sit side by side in one array, they share one 64-byte cache line:

```text
   counters side by side (one cache line):  854.1ms
   each counter on its own cache line:      131.1ms
   the shared cache line costs 6.5x
```

Caches keep lines coherent across cores: before a core can write to a line, it must take exclusive ownership of it, which invalidates every other core's copy. Four cores writing to one line take turns owning it, and the line bounces between their caches on every write. This is **false sharing**: the data is not shared, but the cache line is. The fix is padding: `#[repr(align(64))]` on the counter type forces each onto its own line.

Where it bites in inference: per-thread statistics counters, per-thread partial sums written into one array, and output slices whose boundaries fall in the middle of a cache line (two threads writing the last element of one chunk and the first of the next). The spin pool's chunks are large, so boundary false sharing costs little there, but it is worth knowing when you design any per-thread array.

### 3.8 Amdahl's law

If a fraction `s` of a program's time is serial, then with N threads the best possible speed-up is

```text
speed-up ≤ 1 / (s + (1 − s) / N)
```

Per-call overhead acts like serial time. For a 21 µs matvec with 1.3 µs of spin-pool overhead, the overhead alone limits 4 threads to about 3x; with rayon's 49 µs overhead, parallelism loses outright. In a whole transformer, the non-parallelized parts (sampling, small operators between matmuls, the Python-free but still serial control flow) set the ceiling for the whole forward pass. Chapter 17 measures that fraction on the real model.

## 4. The code

The parallel kernels are in [`src/lib.rs`](src/lib.rs), the spin pool in [`src/pool.rs`](src/pool.rs), and the demo in [`src/main.rs`](src/main.rs).

### 4.1 Scoped threads: borrowing across threads

<!-- file: src/lib.rs -->
```rust
pub fn matvec_scoped(w: &[f32], x: &[f32], y: &mut [f32], threads: usize) {
    assert_eq!(w.len(), x.len() * y.len());
    let k = x.len();
    let rows_each = y.len().div_ceil(threads.max(1));
    std::thread::scope(|s| {
        for (y_part, w_part) in y.chunks_mut(rows_each).zip(w.chunks(rows_each * k)) {
            s.spawn(move || matvec_serial(w_part, x, y_part));
        }
    });
}
```

- `std::thread::scope(|s| ...)` creates a scope in which threads can be spawned with `s.spawn`. The scope does not return until every thread spawned inside it has finished.
- Because of that guarantee, the threads may borrow data from outside the scope: `w_part` and `x` are shared `&[f32]` borrows, `y_part` is a `&mut [f32]` borrow. With plain `std::thread::spawn` this would not compile, because a spawned thread could outlive the function and the borrows would dangle; you would have to use `Arc` and copy data.
- `y.chunks_mut(rows_each)` hands out non-overlapping mutable pieces of `y`, and `w.chunks(rows_each * k)` the matching rows of `W`. `zip` pairs them.
- `move ||` moves the three slice references into the closure (the references, not the data).

What makes this compile is the pair of marker traits `Send` and `Sync`. `s.spawn` requires the closure to be `Send`. A closure holding `&[f32]` is `Send` because `f32` is `Sync` (reading it from several threads is fine). A closure holding `&mut [f32]` is `Send` because `f32` is `Send`, and the `&mut` guarantees nobody else holds that slice. If you tried to give two threads the same `&mut` slice, the borrow checker would reject the second borrow before thread safety even came up.

### 4.2 Rayon

<!-- file: src/lib.rs -->
```rust
        y.par_chunks_mut(rows)
            .zip(w.par_chunks(rows * k))
            .for_each(|(y_part, w_part)| {
                for (out, w_row) in y_part.iter_mut().zip(w_part.chunks_exact(k)) {
                    *out = dot(w_row, x);
                }
            });
```

Rayon's parallel iterators mirror the standard ones: `par_chunks_mut` instead of `chunks_mut`, and the rest reads the same. Each chunk becomes a task of about `TASK_BYTES` (64 KB) of weights, which rayon distributes over its pool with work stealing. The same `Send`/`Sync` rules make it safe.

Rayon is the right default for most parallel Rust code: it balances uneven work, composes well, and needs no `unsafe`. Its weakness, for this workload on this machine, is only the wake-up latency when calls are tiny and frequent.

### 4.3 The spin pool

The whole design, in the order a job flows through it:

<!-- file: src/pool.rs -->
```rust
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
```

`Job<'a>` is `dyn Fn(usize) + Sync + 'a`: a closure that takes a worker index and may be called from several threads at once. The workers are ordinary `'static` threads created once in `SpinPool::new`, but the jobs are closures that borrow local data (weights, activations). So the job pointer is stored with its lifetime erased to `'static`, which is the one genuinely `unsafe` idea in this file.

<!-- file: src/pool.rs -->
```rust
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
```

`run` publishes a job in three steps: store the pointer, set the count of workers that must finish, and bump `epoch`. The order and the memory orderings are what make it correct:

- **`Release` on the epoch increment** means: every write this thread made before the increment (the job pointer, the `remaining` count, and whatever data the caller prepared for the job, such as the input vector) becomes visible to any thread that later reads the new epoch value with **`Acquire`**. Without this pairing, a worker on another core could see the new epoch but an old job pointer.
- The lifetime erasure is sound because of what happens next: `run` does not return until every worker has finished with the job.

<!-- file: src/pool.rs -->
```rust
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
```

The calling thread does a share of the work itself (worker 0), so `threads` threads compute and none sits idle waiting. Then it waits for `remaining` to reach 0. Each worker decrements `remaining` with `Release` after finishing its share, and this `Acquire` load makes the workers' output writes visible to the caller.

The `catch_unwind` is not optional. If the caller's own share panicked and we let the panic propagate immediately, `run` would return (by unwinding) while other workers were still using the job pointer, which points into the caller's stack frame that is about to be destroyed. **Unsafe code must be correct on the panic path too.** So we catch the panic, wait for everyone, and only then re-raise it.

`run` takes `&mut self`. That single choice makes it impossible, at compile time, to call `run` from two threads at once or to start a second job while one is running, both of which would break the protocol.

<!-- file: src/pool.rs -->
```rust
fn worker_loop(shared: &Shared, index: usize) {
    // Start from the epoch the pool was created with (0), *not* from
    // whatever `epoch` holds when this thread first runs: if `run` published
    // a job before this thread got scheduled, loading `epoch` here would
    // treat that job as already seen, and `run` would wait forever.
    let mut seen = 0;
```

This comment records the bug in the first version of this file. The worker began by loading the current epoch as "the last one I have seen". Creating a pool and immediately calling `run` could then publish a job *before* a freshly spawned worker thread got its first time slice. That worker would read the new epoch as its starting point, never see a change, never run the job, and `run` would spin forever waiting for it. The very first run of the test suite hung this way. Starting from the constant 0 (the epoch at construction) removes the race. Concurrency bugs like this one depend on timing, so a test that passes once proves little; run concurrent tests many times, and use tools like Miri (below) that explore different interleavings.

<!-- file: src/pool.rs -->
```rust
fn backoff(spins: &mut u32) {
    if *spins < SPINS_BEFORE_YIELD {
        *spins += 1;
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}
```

`std::hint::spin_loop()` compiles to the CPU's "I am spinning" hint (`pause` on x86, `yield` on ARM), which saves power and frees resources for the other hyper-thread on the same core. After 65,536 spins (tens of microseconds), a waiting thread starts calling `yield_now`, which gives up its time slice so other threads can run. That keeps a pool that is idle for long periods from monopolizing the machine completely, at the cost of a slower first response after a long pause. Real engines also let workers sleep properly after a longer idle period.

### 4.4 Handing out disjoint chunks safely

<!-- file: src/pool.rs -->
```rust
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
```

The job closure is shared by all workers (`Fn`, called through `&`), so it cannot simply hand out `&mut` slices: from the compiler's point of view, every worker could call it with any index. Wrapping each chunk in a `Mutex` turns "give worker *i* exclusive access to chunk *i*" into something safe code can express. Each mutex is only ever locked by one worker, so locking it is an uncontended atomic operation of a few nanoseconds. The alternative, raw pointers plus `unsafe`, would save those nanoseconds and give up the compiler's help; this is a place where safe code is fast enough.

`granule` makes every chunk a multiple of some unit, so that for the transposed output buffer of section 3.2 each chunk holds whole rows of `m` outputs.

### 4.5 The kernel on the pool

<!-- file: src/lib.rs -->
```rust
    if m == 1 {
        pool.for_each_chunk_mut(y, 1, |start, y_part| {
            let rows = &w[start * k..(start + y_part.len()) * k];
            for (out, w_row) in y_part.iter_mut().zip(rows.chunks_exact(k)) {
                *out = dot(w_row, x);
            }
        });
        return;
    }
```

For decode (one input row), each thread takes a contiguous range of outputs and the matching weight rows. `dot` is a generic parameter: the same function runs `f32` weights with chapter 6's `dot` and `bf16` weights with `dot_bf16`, and in chapters 18-19 quantized weights with their own dot products. Generics are compiled separately for each `dot` passed in, so there is no function-pointer call in the inner loop.

### 4.6 False sharing, in code

<!-- file: src/lib.rs -->
```rust
#[repr(align(64))]
#[derive(Default)]
pub struct PaddedCounter(pub AtomicU64);
```

`#[repr(align(64))]` makes the type's alignment 64 bytes, and since a type's size is always a multiple of its alignment, it also occupies 64 bytes. An array of `PaddedCounter` therefore puts each counter on its own cache line. Eight bytes of data, 56 of padding: a deliberate trade of memory for speed.

## 5. Run it

```bash
cargo test -p ch07-threads
cargo run --release -p ch07-threads
```

On the reference machine (4 vCPUs):

```text
cores: 4

== 1. cost of one parallel call that does nothing
   spawn + join 4 OS threads:  153.65µs
   rayon, 4 tasks on the pool:   49.49µs
   spin pool, 4 threads:          1.27µs

== 2. y = W x (f32 weights), 4 threads where parallel
   weights              |   serial | spawned threads |    rayon | spin pool | best GB/s
    576x576  (  1.3 MB) |   20.9µs |         230.7µs |   44.7µs |     9.6µs |     138.3
   1536x576  (  3.5 MB) |  173.0µs |         279.8µs |  103.9µs |    21.2µs |     167.1
   4096x4096 ( 67.1 MB) |    3.2ms |           1.4ms |    1.2ms |   932.4µs |      72.0
   8192x8192 (268.4 MB) |   24.2ms |           6.8ms |   12.9ms |     6.9ms |      39.6

== 3. scaling with the number of threads (spin pools of each size)
   threads | matvec f32 256 MB  | matvec bf16 128 MB | matmul f32 128x2048x2048 | matmul bf16
         1 |  29.3ms   9.2 GB/s |  22.6ms   5.9 GB/s |    30.9ms   34.8 GFLOP/s |   30.6 GFLOP/s
         2 |  12.8ms  21.0 GB/s |  10.4ms  13.0 GB/s |    19.6ms   54.8 GFLOP/s |   54.6 GFLOP/s
         3 |   9.8ms  27.4 GB/s |   6.1ms  21.8 GB/s |    14.4ms   74.8 GFLOP/s |   81.8 GFLOP/s
         4 |   6.9ms  39.0 GB/s |   5.5ms  24.5 GB/s |    10.3ms  104.4 GFLOP/s |   81.1 GFLOP/s

== 4. 4 threads each incrementing their own counter 20000000 times
   counters side by side (one cache line):  854.1ms
   each counter on its own cache line:      131.1ms
   the shared cache line costs 6.5x
```

Beyond what section 3 discussed:

- **"Best GB/s" above the DRAM bandwidth** (138-167 GB/s for the small matrices, 72 GB/s for the 67 MB one) means the data came from caches, not DRAM: the four private L2s for the small ones, and the large shared L3 for the 67 MB one. Only the 256 MB matrix really streams from DRAM (39.6 GB/s).
- **Rayon was slower than spawned threads on the 256 MB matvec in this run** (12.9 ms against 6.8). Its 64 KB tasks mean thousands of task hand-offs per call, and on this noisy VM that run went badly; other runs were closer. Coarse, static partitioning (one chunk per thread) suits uniform work like a matvec. Work stealing is for uneven work.
- **The `bf16` matmul stops scaling at 3 threads** in this run (81.8 and 81.1 GFLOP/s). Treat single runs on a shared VM with care; this one is within the noise we have seen elsewhere.

## 6. The Rust behind it

**`Send` and `Sync` are how Rust prevents data races at compile time.** Every type automatically implements them when all its fields do. `&T` can cross threads when `T: Sync`; `&mut T` when `T: Send`. `Rc` (non-atomic reference counting) is neither; `Arc` is both (for `Sync` contents). The raw pointer in `Shared` is neither by default, so we wrote `unsafe impl Sync for Shared` with a comment stating the protocol that makes it true. Writing an `unsafe impl Send/Sync` is a promise the compiler cannot check, which is why such impls should be rare and heavily commented.

**Scoped threads replace `Arc` for fork-join parallelism.** Before `std::thread::scope` (Rust 1.63), sharing borrowed data with threads required `Arc` or crates like `crossbeam`. For "split this work, wait for it, continue", scoped threads are simpler and have no reference-counting cost.

**Atomics and orderings.** `Relaxed` guarantees only that the operation itself is atomic. `Release` (on a write) and `Acquire` (on a read of that value) create a happens-before edge: everything before the release is visible after the acquire. This pair is the building block of every lock, channel and pool. `SeqCst` (sequential consistency) is stronger and easier to reason about, and slightly slower; the spin pool uses the weakest ordering that is correct at each step, and says why in a comment.

**`UnsafeCell` is the only way to mutate through a shared reference.** Every type with interior mutability (`Cell`, `RefCell`, `Mutex`, atomics) is built on it. The spin pool uses it directly for the job pointer, because its access rules (written only while no worker runs, read only after the epoch changes) are enforced by the protocol rather than by a lock.

**Miri checks concurrent `unsafe` code too.** The pool's tests were run under Miri, which checks for data races and invalid pointer use and explores different thread interleavings: `cargo +nightly miri test -p ch07-threads --lib pool::tests`. All five tests passed. Under Miri the pool yields instead of spinning (`#[cfg(miri)]`), because spinning in an interpreter only wastes time; the synchronization being checked is the same.

## 7. Mistakes you will make

- **Parallelizing work that is too small.** If the work takes less than a few times the per-call overhead, parallel is slower than serial. Measure the overhead on your machine first.
- **Oversubscribing cores.** Running a spinning pool with as many threads as cores while other busy processes share the machine (or while another pool spins) makes everyone slower, because spinners steal time from the threads doing real work. When testing, run spinning pools in isolation.
- **Returning early from unsafe coordination code.** Panics, errors and `?` are all early returns. Any code that hands out borrowed pointers must wait for their users on *every* path.
- **Assuming a race-free test run means race-free code.** The epoch bug in section 4.3 passed most runs.
- **False sharing in per-thread arrays.** Pad per-thread counters and accumulators to a cache line.
- **Benchmarking the same small matrix over and over.** It stays in the cores' private caches and shows speed-ups the real model will not get.

## 8. How the professionals do it

- **llama.cpp** creates its worker threads once and keeps them spinning (with a configurable polling level) while a graph is being computed; each operation is split into per-thread chunks with a barrier between operations. That is this chapter's design.
- **PyTorch on CPU** uses OpenMP (intra-op parallelism) with its own thread pool; setting `OMP_NUM_THREADS` too high for the machine is a classic cause of slow CPU inference.
- **ONNX Runtime** has its own thread pool with optional spinning (`allow_spinning` session option), and exposes the power/latency trade-off to users.
- **GPU engines** do not have this problem in the same form: a GPU launches thousands of threads per kernel in hardware. Their equivalent overhead is the kernel launch (several microseconds per kernel), which is why they fuse kernels and capture whole decode steps as CUDA Graphs (chapter 29).
- **Thread pinning** (setting CPU affinity so each worker stays on one core) and **NUMA awareness** (on multi-socket servers, placing each thread near the memory holding its weights) matter on large servers; the reference machine has one NUMA node.

## 9. Exercises

1. **Overhead on your machine.** Run part 1 on your own computer. How do spawn, rayon and spin-pool overheads compare with this VM's? For which matrix size does rayon start to beat serial?
2. **Granularity.** Change `TASK_BYTES` to 4 KB and to 1 MB, and rerun part 2 with rayon. Explain what changes.
3. **Amdahl.** A forward pass takes 10 ms on one thread; 9 ms of it are matvecs that scale perfectly and 1 ms is serial. What is the best possible time on 4 threads? On 64?
4. **Remove the `catch_unwind`.** In `SpinPool::run`, call `job(0)` directly instead of through `catch_unwind`, and explain precisely what could go wrong if the caller's share panics. (You do not need to make it crash to answer.)
5. **Sleeping versus spinning.** Replace `backoff`'s spinning with an immediate `yield_now` (set `SPINS_BEFORE_YIELD` to 0 outside Miri too), and rerun parts 1 and 2. What happens to the overhead?
6. **Pad the chunks.** In `for_each_chunk_mut`, chunk boundaries can fall in the middle of a cache line. When does that cause false sharing, and why is the effect negligible for a matvec's output?

## 10. Check yourself

1. Why does adding cores speed up a memory-bound matvec, when the arithmetic was never the bottleneck?
2. Why can `std::thread::scope` threads borrow local variables, while `std::thread::spawn` threads cannot?
3. What makes rayon and spawned threads slower than serial for a 576 × 576 matvec on this machine?
4. What do `Release` and `Acquire` guarantee in the spin pool's epoch protocol?
5. Why must `SpinPool::run` wait for all workers even when the caller's own share panics?
6. What is false sharing, and how do you prevent it?

## 11. Recap

- Threads multiply usable memory bandwidth: 9 → 39 GB/s from 1 to 4 cores for a DRAM-bound matvec here.
- Split outputs into disjoint `&mut` slices (`chunks_mut`) and the compiler rules out data races.
- Per-call overhead decides everything for small models: 154 µs to spawn threads, 49 µs for rayon here, 1.3 µs for a spinning pool. Only the spinning pool makes SmolLM2-sized matvecs faster.
- Private caches add up: work split across cores can fit in their combined L2 and speed up super-linearly.
- The spin pool's `unsafe` is a lifetime erasure made sound by waiting for every worker on every path, including panics. It was checked with Miri, and its first version had a start-up race.
- Pad per-thread data to 64 bytes to avoid false sharing.

## Answers

**Exercises**

1. Not measured for this course beyond the reference machine. On bare-metal desktops, thread wake-ups usually go through the OS without a hypervisor in the way, so expect rayon's overhead to be much lower than the 49 µs here, and the crossover (the matrix size where rayon beats serial) to move down accordingly. The spin pool's overhead is dominated by cache-line transfers between cores and should stay around a microsecond. Whatever your numbers, the ratio of overhead to work is what decides.
2. What to look for: with 4 KB tasks, a 256 MB matrix becomes 65,536 tasks and per-task overhead becomes visible; with 1 MB tasks, the 1.3 MB SmolLM2 matrix becomes one or two tasks, so most cores get nothing to do. 64 KB is a compromise between the two. The spin pool sidesteps the question for uniform work by giving each thread exactly one contiguous chunk.
3. The time with N threads is at least 1 ms + 9 ms / N. With 4 threads: 1 + 2.25 = 3.25 ms (3.1x). With 64: 1 + 0.14 = 1.14 ms (8.8x). The serial millisecond caps the speed-up below 10x forever.
4. If `job(0)` panics and unwinds out of `run`, `run` returns (by unwinding) while other workers may still be executing the job. The job is a closure that borrows data from the caller's stack frame, and possibly the closure itself lives there. Once `run` unwinds, that frame is destroyed and the workers are reading freed memory. The caller could even reuse the pool and overwrite the job pointer mid-job. All of this is undefined behaviour, and none of it is caught by the compiler, because the lifetime was erased with `transmute`.
5. Measured on the reference machine: almost nothing changes (1.10 µs per empty call, 7.4 µs for the 576 × 576 matvec). On an otherwise idle machine, `yield_now` finds no other thread that wants the core and returns immediately, so the worker is still polling, just through a system call. The difference appears when other work needs the cores: a yielding worker lets it run, a spinning one does not. What makes rayon slow here is not the absence of spinning but that its idle workers *sleep* (block in the kernel) and must be woken.
6. False sharing happens when two threads *write* to the same cache line repeatedly. At a chunk boundary, thread A writes its last few outputs and thread B its first few, possibly in the same line. For a matvec, each output is written exactly once, after a long dot product, so the line bounces at most a couple of times per call: negligible. It matters for data that is written many times, like the counters in part 4.

**Check yourself**

1. Each core can only keep a limited number of memory requests in flight; more cores means more requests in flight in total, and bandwidth = bytes in flight ÷ latency.
2. The scope guarantees that all its threads finish before it returns, so borrowed data outlives the threads. A `spawn`ed thread may outlive the function that spawned it, so it may only hold `'static` data.
3. Their per-call overhead (154 µs to spawn threads, 49 µs to wake rayon's workers on this VM) is larger than the 21 µs the whole serial matvec takes.
4. Everything `run` wrote before its `Release` increment of `epoch` (the job pointer, the `remaining` count, the caller's input data) is visible to a worker after its `Acquire` load sees the new epoch. Symmetrically, each worker's `Release` decrement of `remaining` makes its output writes visible to the caller's `Acquire` load.
5. Because the workers hold a pointer (with an erased lifetime) to a closure that borrows the caller's stack. Unwinding out of `run` would destroy what that pointer refers to while workers might still be using it.
6. Different threads writing different variables that happen to share a cache line, so the line keeps moving between cores. Prevent it by aligning or padding per-thread data to 64 bytes (`#[repr(align(64))]`), or by giving each thread data far apart in memory.

## Further reading

- Mara Bos, *Rust Atomics and Locks* (O'Reilly, 2023; free online). The best explanation of atomics, memory ordering and building synchronization primitives in Rust.
- The rayon documentation and Niko Matsakis's blog posts introducing it.
- The `ggml` thread pool source (`ggml-cpu/ggml-cpu.c` in llama.cpp) for a production spinning pool.
- Next: [Chapter 8: Neural network operators](../08-operators/README.md). With fast linear layers in hand, we build the smaller operators that sit between them.
