//! Chapter 23: continuous batching.
//!
//! - [`forward`]: [`forward_batch`], one forward pass over several
//!   sequences, each with its own tokens and KV cache.
//! - [`engine`]: an engine thread that batches every request it holds at
//!   every step, admitting and retiring requests between steps.
//! - [`packed`]: a `bf16` weight layout and kernel for small batches.

pub mod engine;
pub mod forward;
pub mod packed;

pub use engine::{BatchConfig, spawn};
pub use forward::{BatchScratch, BatchSeq, forward_batch};
pub use packed::PackedBf16;

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Model, Scratch};
    use ch15_sampling::{FinishReason, SamplingParams};
    use ch20_flash_attention::FlashOptions;
    use ch21_engine_thread::{Event, Request, collect};

    fn model() -> Model<DenseF32> {
        Model::from_reference(&Weights::random(&Config::tiny(), 3))
    }

    fn prompt(len: usize, seed: u32) -> Vec<u32> {
        (0..len as u32).map(|i| (i * 7 + seed * 13) % 97).collect()
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
    }

    /// Logits of the last token of `tokens` after `before`, one sequence at
    /// a time with chapter 14's forward pass.
    fn alone(
        model: &Model<DenseF32>,
        pool: &mut SpinPool,
        before: &[u32],
        tokens: &[u32],
    ) -> Vec<f32> {
        let mut cache = KvCache::new(&model.config, 128);
        let mut scratch = Scratch::new(&model.config, 64, 128);
        if !before.is_empty() {
            model.forward_last(pool, before, &mut cache, &mut scratch);
        }
        model
            .forward_last(pool, tokens, &mut cache, &mut scratch)
            .to_vec()
    }

    #[test]
    fn a_batch_computes_what_each_sequence_computes_alone() {
        let model = model();
        let mut pool = SpinPool::new(3);
        let (a, b, c) = (prompt(9, 1), prompt(20, 2), prompt(33, 3));
        let mut caches: Vec<KvCache> = (0..3).map(|_| KvCache::new(&model.config, 128)).collect();
        let mut scratch = BatchScratch::new(&model.config, 64, 3);
        let opts = FlashOptions::default();
        let vocab = model.config.vocab_size;

        // Step 1: `a` and `b` prefill whole prompts, `c` only its first 12
        // tokens (a chunk that needs no logits).
        let [ca, cb, cc] = &mut caches[..] else {
            unreachable!()
        };
        let mut seqs = vec![
            BatchSeq {
                tokens: &a,
                cache: ca,
                logits: true,
            },
            BatchSeq {
                tokens: &b,
                cache: cb,
                logits: true,
            },
            BatchSeq {
                tokens: &c[..12],
                cache: cc,
                logits: false,
            },
        ];
        let logits = forward_batch(&model, &mut pool, &mut seqs, &mut scratch, &opts).to_vec();
        assert_eq!(logits.len(), 2 * vocab);
        assert!(close(&logits[..vocab], &alone(&model, &mut pool, &[], &a)));
        assert!(close(&logits[vocab..], &alone(&model, &mut pool, &[], &b)));

        // Step 2: `a` and `b` decode one token each while `c` finishes its
        // prompt.
        let [ca, cb, cc] = &mut caches[..] else {
            unreachable!()
        };
        let mut seqs = vec![
            BatchSeq {
                tokens: &[5],
                cache: ca,
                logits: true,
            },
            BatchSeq {
                tokens: &c[12..],
                cache: cc,
                logits: true,
            },
            BatchSeq {
                tokens: &[6],
                cache: cb,
                logits: true,
            },
        ];
        let logits = forward_batch(&model, &mut pool, &mut seqs, &mut scratch, &opts).to_vec();
        assert!(close(&logits[..vocab], &alone(&model, &mut pool, &a, &[5])));
        assert!(close(
            &logits[vocab..2 * vocab],
            &alone(&model, &mut pool, &c[..12], &c[12..])
        ));
        assert!(close(
            &logits[2 * vocab..],
            &alone(&model, &mut pool, &b, &[6])
        ));
        assert_eq!(
            caches.iter().map(KvCache::len).collect::<Vec<_>>(),
            [10, 21, 33]
        );
    }

    fn request(prompt: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            prompt,
            params: SamplingParams::greedy(),
            max_tokens,
            stop_tokens: Vec::new(),
        }
    }

    fn config(max_batch: usize, max_step_tokens: usize) -> BatchConfig {
        BatchConfig {
            threads: 2,
            max_batch,
            context: 128,
            max_step_tokens,
        }
    }

    /// Checks that every token of `output` was a greedy choice given what
    /// came before it, by running prompt + output through chapter 14's
    /// forward pass: the chosen token's logit must equal the maximum, up to
    /// rounding (batching changes the order of floating-point additions).
    fn assert_greedy(model: &Model<DenseF32>, prompt: &[u32], output: &[u32]) {
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
                "token {j}: chose {token} with logit {}, the maximum is {max}",
                row[token as usize]
            );
        }
    }

    #[test]
    fn many_requests_at_once_each_get_their_own_greedy_answer() {
        // 5 requests, 3 slots, steps of at most 16 tokens: prompts are
        // prefilled in chunks alongside other requests' decode steps.
        let (handle, _thread) = spawn(model(), config(3, 16));
        let prompts: Vec<Vec<u32>> = (0..5).map(|i| prompt(10 + 7 * i, i as u32)).collect();
        let receivers: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| handle.submit(request(p.clone(), 8 + i)).unwrap())
            .collect();
        let reference = model();
        for (i, (events, p)) in receivers.into_iter().zip(&prompts).enumerate() {
            let (tokens, summary) = collect(events).unwrap();
            assert_eq!(tokens.len(), 8 + i);
            assert_eq!(summary.finish, FinishReason::Length);
            assert_eq!(summary.completion_tokens, 8 + i);
            assert_greedy(&reference, p, &tokens);
        }
    }

    #[test]
    fn a_client_leaving_does_not_disturb_the_others() {
        let (handle, _thread) = spawn(model(), config(4, 32));
        let mut leaving = handle.submit(request(prompt(12, 1), 50)).unwrap();
        let staying = handle.submit(request(prompt(15, 2), 20)).unwrap();
        assert!(matches!(leaving.blocking_recv(), Some(Event::Token(_))));
        drop(leaving);
        let (tokens, _) = collect(staying).unwrap();
        assert_eq!(tokens.len(), 20);
        assert_greedy(&model(), &prompt(15, 2), &tokens);
    }

    #[test]
    fn stop_tokens_end_a_request_and_prompts_that_do_not_fit_are_rejected() {
        let (handle, _thread) = spawn(model(), config(2, 8));
        // Find the first token greedy decoding produces, then stop on it.
        let p = prompt(10, 4);
        let (first, _) = collect(handle.submit(request(p.clone(), 1)).unwrap()).unwrap();
        let mut stopping = request(p, 10);
        stopping.stop_tokens = vec![first[0]];
        let (tokens, summary) = collect(handle.submit(stopping).unwrap()).unwrap();
        assert!(tokens.is_empty());
        assert_eq!(summary.finish, FinishReason::StopToken);
        assert!(collect(handle.submit(request(prompt(200, 5), 4)).unwrap()).is_err());
    }

    #[test]
    fn the_engine_stops_when_every_handle_is_dropped() {
        let (handle, thread) = spawn(model(), config(2, 8));
        let events = handle.submit(request(prompt(5, 1), 3)).unwrap();
        drop(handle);
        // The queued request is still served, then the thread ends.
        assert_eq!(collect(events).unwrap().0.len(), 3);
        thread.join().unwrap();
    }
}
