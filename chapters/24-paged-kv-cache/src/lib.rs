//! Chapter 24: the paged KV cache.
//!
//! - [`blocks`]: a pool of fixed-size blocks with reference counts, and a
//!   block table per sequence.
//! - [`prefix`]: finding the blocks of earlier requests that started with
//!   the same tokens.
//! - [`attention`]: attention that reads keys and values block by block.
//! - [`forward`]: chapter 23's batched forward pass over the paged cache.
//! - [`engine`]: the batching engine with blocks: admission by memory,
//!   preemption, prefix caching.

pub mod attention;
pub mod blocks;
pub mod engine;
pub mod forward;
pub mod prefix;

pub use blocks::{BlockPool, BlockTable};
pub use engine::{PagedConfig, Stats, spawn};
pub use forward::{PagedScratch, PagedSeq, forward_paged};
pub use prefix::PrefixCache;

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Model, Scratch};
    use ch15_sampling::{FinishReason, SamplingParams};
    use ch20_flash_attention::FlashOptions;
    use ch21_engine_thread::{Request, collect};
    use std::sync::atomic::Ordering;

    fn model() -> Model<DenseF32> {
        Model::from_reference(&Weights::random(&Config::tiny(), 5))
    }

    fn prompt(len: usize, seed: u32) -> Vec<u32> {
        (0..len as u32).map(|i| (i * 11 + seed * 17) % 97).collect()
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
    }

    /// Chapter 14's logits for the last token of `tokens` after `before`.
    fn alone(model: &Model<DenseF32>, before: &[u32], tokens: &[u32]) -> Vec<f32> {
        let mut pool = SpinPool::new(2);
        let mut cache = KvCache::new(&model.config, 128);
        let mut scratch = Scratch::new(&model.config, 64, 128);
        if !before.is_empty() {
            model.forward_last(&mut pool, before, &mut cache, &mut scratch);
        }
        model
            .forward_last(&mut pool, tokens, &mut cache, &mut scratch)
            .to_vec()
    }

    #[test]
    fn the_paged_forward_pass_computes_what_chapter_14_computes() {
        let model = model();
        let vocab = model.config.vocab_size;
        let mut pool = SpinPool::new(3);
        let opts = FlashOptions::default();
        for block_size in [1, 4, 5, 16] {
            let mut kv = BlockPool::new(&model.config, 64, block_size);
            let mut scratch = PagedScratch::new(&model.config, 64, 3);
            let (a, b) = (prompt(9, 1), prompt(23, 2));
            let mut tables = [BlockTable::default(), BlockTable::default()];
            for (t, p) in tables.iter_mut().zip([&a, &b]) {
                assert!(t.reserve(&mut kv, p.len() + 1));
            }
            // Step 1: both prompts. Step 2: one decode token each.
            let [ta, tb] = &mut tables;
            let mut seqs = vec![
                PagedSeq {
                    tokens: &a,
                    table: ta,
                    logits: true,
                },
                PagedSeq {
                    tokens: &b,
                    table: tb,
                    logits: true,
                },
            ];
            let logits =
                forward_paged(&model, &mut pool, &mut kv, &mut seqs, &mut scratch, &opts).to_vec();
            assert!(
                close(&logits[..vocab], &alone(&model, &[], &a)),
                "block size {block_size}"
            );
            assert!(
                close(&logits[vocab..], &alone(&model, &[], &b)),
                "block size {block_size}"
            );
            let [ta, tb] = &mut tables;
            let mut seqs = vec![
                PagedSeq {
                    tokens: &[7],
                    table: ta,
                    logits: true,
                },
                PagedSeq {
                    tokens: &[8],
                    table: tb,
                    logits: true,
                },
            ];
            let logits =
                forward_paged(&model, &mut pool, &mut kv, &mut seqs, &mut scratch, &opts).to_vec();
            assert!(
                close(&logits[..vocab], &alone(&model, &a, &[7])),
                "block size {block_size}"
            );
            assert!(
                close(&logits[vocab..], &alone(&model, &b, &[8])),
                "block size {block_size}"
            );
        }
    }

    fn request(prompt: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            prompt,
            params: SamplingParams::greedy(),
            max_tokens,
            stop_tokens: Vec::new(),
        }
    }

    fn config(num_blocks: usize, block_size: usize, prefix_caching: bool) -> PagedConfig {
        PagedConfig {
            threads: 2,
            max_batch: 8,
            context: 128,
            max_step_tokens: 16,
            num_blocks,
            block_size,
            prefix_caching,
        }
    }

    /// Every token of `output` must be a greedy choice given the tokens
    /// before it (checked with chapter 14's forward pass).
    fn assert_greedy(prompt: &[u32], output: &[u32]) {
        let model = model();
        let mut pool = SpinPool::new(2);
        let mut cache = KvCache::new(&model.config, 128);
        let mut scratch = Scratch::new(&model.config, 128, 128);
        let all: Vec<u32> = prompt.iter().chain(output).copied().collect();
        let logits = model.forward_all(&mut pool, &all, &mut cache, &mut scratch);
        let vocab = model.config.vocab_size;
        for (j, &token) in output.iter().enumerate() {
            let row = &logits[(prompt.len() - 1 + j) * vocab..][..vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max - row[token as usize] <= 1e-4 * (1.0 + max.abs()),
                "token {j} is not the greedy choice"
            );
        }
    }

    #[test]
    fn requests_squeezed_into_too_few_blocks_are_preempted_and_still_correct() {
        // 6 requests of 12-22 prompt tokens and 30 new tokens, 4-position
        // blocks, 40 blocks (160 positions) for about 270 positions of
        // demand: some requests must give their blocks back and recompute.
        let (handle, _thread, stats) = spawn(model(), config(40, 4, false));
        let prompts: Vec<Vec<u32>> = (0..6).map(|i| prompt(12 + 2 * i, i as u32)).collect();
        let receivers: Vec<_> = prompts
            .iter()
            .map(|p| handle.submit(request(p.clone(), 30)).unwrap())
            .collect();
        for (events, p) in receivers.into_iter().zip(&prompts) {
            let (tokens, summary) = collect(events).unwrap();
            assert_eq!(tokens.len(), 30);
            assert_eq!(summary.finish, FinishReason::Length);
            assert_greedy(p, &tokens);
        }
        assert!(
            stats.preemptions.load(Ordering::Relaxed) > 0,
            "no preemption happened"
        );
        assert!(stats.peak_blocks.load(Ordering::Relaxed) <= 40);
    }

    #[test]
    fn a_shared_prefix_is_computed_once() {
        let (handle, _thread, stats) = spawn(model(), config(64, 4, true));
        let shared = prompt(20, 9);
        let with = |tail: &[u32]| -> Vec<u32> { shared.iter().chain(tail).copied().collect() };
        // The first request fills the cache with the shared blocks...
        let first = with(&[1, 2, 3]);
        let (tokens, _) = collect(handle.submit(request(first.clone(), 5)).unwrap()).unwrap();
        assert_greedy(&first, &tokens);
        // ...which the second finds: 20 shared tokens = 5 full blocks.
        let second = with(&[4, 5]);
        let (tokens, _) = collect(handle.submit(request(second.clone(), 5)).unwrap()).unwrap();
        assert_greedy(&second, &tokens);
        assert_eq!(stats.cached_tokens.load(Ordering::Relaxed), 20);
        // The whole of `first` again: every full block but the one holding
        // the last prompt token, whose logits must be computed.
        let (tokens, _) = collect(handle.submit(request(first.clone(), 5)).unwrap()).unwrap();
        assert_greedy(&first, &tokens);
        assert_eq!(stats.cached_tokens.load(Ordering::Relaxed), 20 + 20);
    }

    #[test]
    fn prompts_that_cannot_fit_are_rejected_and_the_engine_stops() {
        let (handle, thread, _) = spawn(model(), config(8, 4, true));
        // 8 blocks of 4 hold 32 positions.
        assert!(collect(handle.submit(request(prompt(40, 1), 4)).unwrap()).is_err());
        let events = handle.submit(request(prompt(10, 2), 4)).unwrap();
        drop(handle);
        assert_eq!(collect(events).unwrap().0.len(), 4);
        thread.join().unwrap();
    }
}
