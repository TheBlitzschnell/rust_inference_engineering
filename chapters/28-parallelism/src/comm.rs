//! Simulated devices that share nothing and talk only by messages.
//!
//! Each device is a thread. A [`Comm`] (a "communicator", as NCCL and MPI
//! call it) connects it to every other device of its group by channels.
//! Sending copies the data into a new buffer, as a transfer between two
//! devices' memories would; on this machine that copy is the whole cost of
//! a "link". The collectives that real libraries provide are built on
//! sends and receives: all-reduce (ring and naive), all-gather, all-to-all.
//! Each communicator counts its messages, bytes and time spent in
//! communication.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

/// One device's connections to the others in its group.
pub struct Comm {
    rank: usize,
    size: usize,
    /// `to[r]`: the channel to device `r` (unused for `r == rank`).
    to: Vec<Sender<Vec<f32>>>,
    /// `from[r]`: the channel from device `r`. A `Receiver` cannot be
    /// shared between threads; the mutex makes `Comm` `Sync`, which the
    /// tensor-parallel matrices need (it is never contended: only this
    /// device's thread uses it).
    from: Vec<Mutex<Receiver<Vec<f32>>>>,
    messages: AtomicUsize,
    bytes: AtomicUsize,
    nanos: AtomicU64,
}

/// A group of `size` connected devices, one `Comm` each, in rank order.
pub fn group(size: usize) -> Vec<Comm> {
    assert!(size > 0, "a group needs at least one device");
    // to[a][b] sends from device a to device b; from[b][a] receives it.
    let mut to: Vec<Vec<Sender<Vec<f32>>>> = (0..size).map(|_| Vec::new()).collect();
    let mut from: Vec<Vec<Mutex<Receiver<Vec<f32>>>>> = (0..size).map(|_| Vec::new()).collect();
    for sender in &mut to {
        for receiver in &mut from {
            let (s, r) = channel();
            sender.push(s);
            receiver.push(Mutex::new(r));
        }
    }
    to.into_iter()
        .zip(from)
        .enumerate()
        .map(|(rank, (to, from))| Comm {
            rank,
            size,
            to,
            from,
            messages: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            nanos: AtomicU64::new(0),
        })
        .collect()
}

/// Communication counters of one device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommStats {
    pub messages: usize,
    pub bytes: usize,
    /// Time spent inside communication calls: transfers, and waiting for
    /// the other devices to get there.
    pub time: Duration,
}

impl Comm {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn stats(&self) -> CommStats {
        CommStats {
            messages: self.messages.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            time: Duration::from_nanos(self.nanos.load(Ordering::Relaxed)),
        }
    }

    /// Sends a copy of `data` to device `to`.
    pub fn send(&self, to: usize, data: &[f32]) {
        self.messages.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(size_of_val(data), Ordering::Relaxed);
        self.to[to]
            .send(data.to_vec())
            .expect("the other device is gone");
    }

    /// Waits for the next message from device `from`.
    pub fn recv(&self, from: usize) -> Vec<f32> {
        self.from[from]
            .lock()
            .expect("receiver lock")
            .recv()
            .expect("the other device is gone")
    }

    /// Runs `f`, adding its duration to the communication time.
    fn timed<T>(&self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        out
    }

    /// The part of an `n`-element buffer that chunk `i` of `size` covers.
    fn chunk(&self, n: usize, i: usize) -> std::ops::Range<usize> {
        i * n / self.size..(i + 1) * n / self.size
    }

    /// Ring all-reduce: afterwards every device holds the element-wise sum
    /// of everyone's `x`. The buffer is cut into `size` chunks; in
    /// `size - 1` steps each device passes one chunk to the next device and
    /// adds the chunk it receives from the previous one (reduce-scatter),
    /// so that each device ends with one chunk fully summed; in `size - 1`
    /// more steps the summed chunks travel around the ring (all-gather).
    /// Every device sends `2 (size - 1) / size` buffers' worth of data,
    /// however many devices there are.
    pub fn all_reduce(&self, x: &mut [f32]) {
        let (n, p) = (self.size, x.len());
        if n == 1 {
            return;
        }
        self.timed(|| {
            let (next, prev) = ((self.rank + 1) % n, (self.rank + n - 1) % n);
            for step in 0..n - 1 {
                let send = self.chunk(p, (self.rank + n - step) % n);
                self.send(next, &x[send]);
                let recv = self.chunk(p, (self.rank + n - step - 1) % n);
                let incoming = self.recv(prev);
                for (a, b) in x[recv].iter_mut().zip(&incoming) {
                    *a += b;
                }
            }
            for step in 0..n - 1 {
                let send = self.chunk(p, (self.rank + 1 + n - step) % n);
                self.send(next, &x[send]);
                let recv = self.chunk(p, (self.rank + n - step) % n);
                let incoming = self.recv(prev);
                x[recv].copy_from_slice(&incoming);
            }
        });
    }

    /// All-reduce the simple way: everyone sends its buffer to device 0,
    /// which sums them and sends the result back to everyone. Device 0
    /// sends and receives `size - 1` whole buffers.
    pub fn all_reduce_naive(&self, x: &mut [f32]) {
        if self.size == 1 {
            return;
        }
        self.timed(|| {
            if self.rank == 0 {
                for from in 1..self.size {
                    for (a, b) in x.iter_mut().zip(&self.recv(from)) {
                        *a += b;
                    }
                }
                for to in 1..self.size {
                    self.send(to, x);
                }
            } else {
                self.send(0, x);
                x.copy_from_slice(&self.recv(0));
            }
        });
    }

    /// Every device's `mine`, in rank order, on every device (each may
    /// have a different length). Ring: in `size - 1` steps, each device
    /// passes on the piece it received last.
    pub fn all_gather(&self, mine: &[f32]) -> Vec<Vec<f32>> {
        let n = self.size;
        let mut pieces = vec![Vec::new(); n];
        pieces[self.rank] = mine.to_vec();
        if n == 1 {
            return pieces;
        }
        self.timed(|| {
            let (next, prev) = ((self.rank + 1) % n, (self.rank + n - 1) % n);
            for step in 0..n - 1 {
                let send = (self.rank + n - step) % n;
                self.send(next, &pieces[send]);
                pieces[(self.rank + n - step - 1) % n] = self.recv(prev);
            }
        });
        pieces
    }

    /// Sends `outgoing[r]` to every device `r` and returns what each device
    /// sent here, in rank order (the piece for itself stays local).
    pub fn all_to_all(&self, outgoing: &[Vec<f32>]) -> Vec<Vec<f32>> {
        assert_eq!(outgoing.len(), self.size, "one piece per device");
        self.timed(|| {
            for (to, piece) in outgoing.iter().enumerate() {
                if to != self.rank {
                    self.send(to, piece);
                }
            }
            (0..self.size)
                .map(|from| {
                    if from == self.rank {
                        outgoing[from].clone()
                    } else {
                        self.recv(from)
                    }
                })
                .collect()
        })
    }
}

/// Runs `f` on every device of a new group of `size`, each on its own
/// thread, and returns the results in rank order.
pub fn run<T: Send>(size: usize, f: impl Fn(&Comm) -> T + Sync) -> Vec<T> {
    let comms = group(size);
    std::thread::scope(|scope| {
        let handles: Vec<_> = comms
            .iter()
            .map(|comm| {
                let f = &f;
                scope.spawn(move || f(comm))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("device thread"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(rank: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| (rank * 1000 + i) as f32).collect()
    }

    fn sum(size: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (0..size).map(|r| (r * 1000 + i) as f32).sum())
            .collect()
    }

    #[test]
    fn ring_and_naive_all_reduce_sum_everything() {
        for size in 1..=5 {
            for len in [1, 3, 7, 64, 1000] {
                let ring = run(size, |c| {
                    let mut x = data(c.rank(), len);
                    c.all_reduce(&mut x);
                    (x, c.stats())
                });
                let naive = run(size, |c| {
                    let mut x = data(c.rank(), len);
                    c.all_reduce_naive(&mut x);
                    x
                });
                for ((x, _), y) in ring.iter().zip(&naive) {
                    assert_eq!(x, &sum(size, len), "ring, {size} devices, {len}");
                    assert_eq!(y, &sum(size, len), "naive, {size} devices, {len}");
                }
                // Every device sends 2 (size - 1) chunks of about len/size.
                if size > 1 && len >= size {
                    for (_, s) in &ring {
                        assert_eq!(s.messages, 2 * (size - 1));
                        let expect = 2 * (size - 1) * len / size * 4;
                        assert!(s.bytes.abs_diff(expect) <= 2 * size * 4, "{}", s.bytes);
                    }
                }
            }
        }
    }

    #[test]
    fn all_gather_and_all_to_all_deliver_every_piece() {
        let gathered = run(4, |c| c.all_gather(&data(c.rank(), c.rank() + 1)));
        for g in &gathered {
            for (r, piece) in g.iter().enumerate() {
                assert_eq!(piece, &data(r, r + 1));
            }
        }
        let exchanged = run(3, |c| {
            let out: Vec<Vec<f32>> = (0..3).map(|to| vec![(c.rank() * 10 + to) as f32]).collect();
            c.all_to_all(&out)
        });
        for (me, got) in exchanged.iter().enumerate() {
            for (from, piece) in got.iter().enumerate() {
                assert_eq!(piece, &vec![(from * 10 + me) as f32]);
            }
        }
    }
}
