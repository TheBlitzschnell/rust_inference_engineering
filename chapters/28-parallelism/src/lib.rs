//! Chapter 28: more than one device, simulated with threads.
//!
//! - [`comm`]: devices that share nothing, and the collectives they use
//!   (ring and naive all-reduce, all-gather, all-to-all).
//! - [`tp`]: tensor parallelism, every layer split across devices.
//! - [`pp`]: pipeline parallelism, consecutive layers on consecutive
//!   devices.
//! - [`moe`]: a mixture-of-experts layer, and expert parallelism.

pub mod comm;
pub mod moe;
pub mod pp;
pub mod tp;

pub use comm::{Comm, CommStats, group, run};
pub use moe::{EpStats, Moe, forward_ep};
pub use tp::{After, Tp, shard, to_f32};

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Model, Scratch};

    fn model() -> Model<DenseF32> {
        Model::from_reference(&Weights::random(&Config::tiny(), 28))
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
    }

    /// Logits for every position of `tokens` (one prefill), then for one
    /// more token (one decode step).
    fn logits<W: ch14_kv_cache::Matrix>(m: &Model<W>, tokens: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let mut pool = SpinPool::new(1);
        let mut cache = KvCache::new(&m.config, 64);
        let mut scratch = Scratch::new(&m.config, 64, 64);
        let prefill = m
            .forward_all(&mut pool, tokens, &mut cache, &mut scratch)
            .to_vec();
        let decode = m
            .forward_last(&mut pool, &[42], &mut cache, &mut scratch)
            .to_vec();
        (prefill, decode)
    }

    #[test]
    fn tensor_parallel_logits_equal_one_device() {
        let full = model();
        let tokens = [3, 1, 4, 1, 5, 9, 2, 6];
        let (want_prefill, want_decode) = logits(&full, &tokens);
        // The tiny model has 2 KV heads: 1 or 2 devices.
        for devices in [1, 2] {
            let got = run(devices, |comm| {
                let mine = shard(&full, comm, DenseF32::new);
                (logits(&mine, &tokens), comm.stats())
            });
            for ((prefill, decode), stats) in &got {
                assert!(close(prefill, &want_prefill), "{devices} devices, prefill");
                assert!(close(decode, &want_decode), "{devices} devices, decode");
                if devices > 1 {
                    // Per forward pass: 2 all-reduces per layer, 1 all-gather.
                    let per_pass = 2 * full.config.num_layers * 2 * (devices - 1) + (devices - 1);
                    assert_eq!(stats.messages, 2 * per_pass);
                }
            }
        }
    }

    /// Chapter 14's greedy continuation of `prompt`.
    fn greedy(m: &Model<DenseF32>, prompt: &[u32], new: usize) -> Vec<u32> {
        let mut pool = SpinPool::new(1);
        let mut cache = KvCache::new(&m.config, 64);
        let mut scratch = Scratch::new(&m.config, 64, 64);
        let all = ch14_kv_cache::generate_greedy(
            m,
            &mut pool,
            prompt,
            new,
            &mut cache,
            &mut scratch,
            |_, _| {},
        );
        all[prompt.len()..].to_vec()
    }

    #[test]
    fn a_pipeline_generates_what_one_device_generates() {
        let m = model();
        let prompts = vec![vec![1, 2, 3], vec![9, 8, 7, 6, 5], vec![42]];
        let want: Vec<Vec<u32>> = prompts.iter().map(|p| greedy(&m, p, 12)).collect();
        // The tiny model has 2 layers: 1 or 2 stages.
        for stages in [1, 2] {
            let run = pp::generate(&m, stages, &prompts, 12);
            assert_eq!(run.outputs, want, "{stages} stages");
            assert_eq!(run.busy.len(), stages);
        }
    }

    #[test]
    fn expert_parallelism_equals_one_device() {
        let moe = Moe::random(32, 48, 8, 2, 7);
        let m = 40;
        let x: Vec<f32> = (0..m * 32)
            .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
            .collect();
        let mut pool = SpinPool::new(1);
        for capacity in [usize::MAX, 6] {
            let (want, load) = moe.forward(&mut pool, &x, m, capacity);
            assert_eq!(
                load.iter().sum::<usize>() < 2 * m,
                capacity == 6,
                "drops only with a limit"
            );
            for devices in [1, 2, 4] {
                let parts = run(devices, |comm| {
                    let rows = moe::block(m, comm.rank(), devices);
                    let mut pool = SpinPool::new(1);
                    forward_ep(
                        &moe,
                        comm,
                        &mut pool,
                        &x[rows.start * 32..rows.end * 32],
                        capacity,
                    )
                });
                let got: Vec<f32> = parts.iter().flat_map(|(y, _)| y.iter().copied()).collect();
                assert!(close(&got, &want), "{devices} devices, capacity {capacity}");
                let ep_load: Vec<usize> = parts
                    .iter()
                    .flat_map(|(_, s)| s.load.iter().copied())
                    .collect();
                assert_eq!(ep_load, load);
            }
        }
    }
}
