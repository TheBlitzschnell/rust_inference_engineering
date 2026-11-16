//! Chapter 26: speculative decoding.
//!
//! - [`verify`]: the rules for keeping draft tokens: greedy agreement, and
//!   rejection sampling, which keeps the target's distribution exactly.
//! - [`draft`]: two drafters: a small model, and n-gram lookup in the text
//!   so far.
//! - [`speculative`]: the loop: draft, verify in one target pass, keep,
//!   roll the caches back.

pub mod draft;
pub mod speculative;
pub mod verify;

pub use draft::{Draft, Drafter, Mode, ModelDrafter, NgramDrafter};
pub use speculative::{Generation, SpecStats, Target};
pub use verify::{Proposal, argmax, draw, probabilities, verify_greedy, verify_sampled};

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Model, Scratch};
    use ch15_sampling::Rng;

    fn model(seed: u64) -> Model<DenseF32> {
        Model::from_reference(&Weights::random(&Config::tiny(), seed))
    }

    fn prompt() -> Vec<u32> {
        vec![3, 14, 15, 92, 65, 35, 89, 79]
    }

    fn generation(max_new: usize, k: usize, mode: Mode, seed: u64) -> Generation {
        Generation {
            max_new,
            k,
            mode,
            seed,
            stop: Vec::new(),
        }
    }

    /// Every token of `output` must be the target's greedy choice given the
    /// tokens before it (within rounding: the k-token and 1-token forward
    /// passes may add in a different order).
    fn assert_greedy(target: &Model<DenseF32>, prompt: &[u32], output: &[u32]) {
        let mut pool = SpinPool::new(2);
        let mut cache = KvCache::new(&target.config, 128);
        let mut scratch = Scratch::new(&target.config, 128, 128);
        let all: Vec<u32> = prompt.iter().chain(output).copied().collect();
        let logits = target.forward_all(&mut pool, &all, &mut cache, &mut scratch);
        let vocab = target.config.vocab_size;
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
    fn greedy_speculation_gives_the_targets_own_answer() {
        let target = model(1);
        let mut pool = SpinPool::new(2);
        let mut t = Target::new(&target, 128);
        // A perfect drafter (the target itself), a poor one (another random
        // model), and n-gram lookup; several draft lengths.
        let mut drafters: Vec<(&str, Box<dyn Drafter>)> = vec![
            ("same model", Box::new(ModelDrafter::new(model(1), 128))),
            ("other model", Box::new(ModelDrafter::new(model(2), 128))),
            ("n-gram", Box::new(NgramDrafter { min_n: 1, max_n: 3 })),
        ];
        let plain_decoding = generation(40, 0, Mode::Greedy, 0);
        let (plain, _) = t.generate(
            &mut pool,
            drafters[2].1.as_mut(),
            &prompt(),
            &plain_decoding,
        );
        assert_greedy(&target, &prompt(), &plain);
        for (name, d) in &mut drafters {
            for k in [1, 3, 5] {
                let g = generation(40, k, Mode::Greedy, 0);
                let (out, stats) = t.generate(&mut pool, d.as_mut(), &prompt(), &g);
                assert_eq!(out.len(), 40, "{name}, k = {k}");
                assert_greedy(&target, &prompt(), &out);
                assert_eq!(stats.accepted + stats.target_passes, 40);
                if *name == "same model" {
                    assert_eq!(stats.accepted, stats.drafted, "a perfect drafter");
                }
            }
        }
    }

    /// Total variation distance: half the sum of absolute differences.
    fn tv(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.0
    }

    fn frequencies(counts: &[usize], trials: usize) -> Vec<f64> {
        counts.iter().map(|&c| c as f64 / trials as f64).collect()
    }

    /// The distribution of the token after `p` at `temperature`.
    fn next_token(model: &Model<DenseF32>, p: &[u32], temperature: f32) -> Vec<f32> {
        let mut pool = SpinPool::new(1);
        let mut cache = KvCache::new(&model.config, 64);
        let mut scratch = Scratch::new(&model.config, 64, 64);
        probabilities(
            model.forward_last(&mut pool, p, &mut cache, &mut scratch),
            temperature,
        )
    }

    #[test]
    fn sampled_speculation_draws_from_the_targets_distribution() {
        // The first token after the prompt, many times, with a draft model
        // that disagrees with the target: speculative sampling must still
        // give the target's distribution.
        let temperature = 0.25;
        let (target, draft_model) = (model(1), model(2));
        let p = prompt();
        let want = next_token(&target, &p, temperature);
        let q = next_token(&draft_model, &p, temperature);
        let vocab = target.config.vocab_size;
        let mut pool = SpinPool::new(1);
        let mut t = Target::new(&target, 64);
        let mut drafter = ModelDrafter::new(draft_model, 64);
        let trials = 10_000;
        let mut counts = vec![0; vocab];
        for seed in 0..trials {
            let g = generation(2, 1, Mode::Sample { temperature }, seed as u64);
            let (out, _) = t.generate(&mut pool, &mut drafter, &p, &g);
            counts[out[0] as usize] += 1;
        }
        // How far `trials` draws straight from the target land from it, by
        // chance alone.
        let mut rng = Rng::new(7);
        let mut direct = vec![0; vocab];
        for _ in 0..trials {
            direct[draw(&want, &mut rng) as usize] += 1;
        }
        let want: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
        let q: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();
        let noise = tv(&frequencies(&direct, trials), &want);
        let threshold = 2.0 * noise + 0.01;
        let got = tv(&frequencies(&counts, trials), &want);
        assert!(got < threshold, "{got} against noise {noise}");
        // The test must be able to tell: drawing from the draft model alone
        // would land far beyond the threshold.
        let off = tv(&q, &want);
        assert!(off > 2.0 * threshold, "{off} against threshold {threshold}");
    }

    #[test]
    fn a_perfect_drafter_is_always_accepted_when_sampling() {
        // Same model, same temperature: p = q at every position, so every
        // draft token passes (and the rows of p line up with the drafts).
        let target = model(3);
        let mut pool = SpinPool::new(2);
        let mut t = Target::new(&target, 128);
        let mut drafter = ModelDrafter::new(model(3), 128);
        let g = generation(41, 4, Mode::Sample { temperature: 0.8 }, 5);
        let (out, stats) = t.generate(&mut pool, &mut drafter, &prompt(), &g);
        assert_eq!(out.len(), 41);
        assert_eq!((stats.target_passes, stats.accepted), (9, 32));
    }

    #[test]
    fn generation_ends_before_a_stop_token_even_inside_a_draft() {
        let target = model(1);
        let mut pool = SpinPool::new(2);
        let mut t = Target::new(&target, 128);
        let mut drafter = ModelDrafter::new(model(1), 128);
        let (full, _) = t.generate(
            &mut pool,
            &mut drafter,
            &prompt(),
            &generation(30, 0, Mode::Greedy, 0),
        );
        // Stop at the token generated 10th: with k = 6 and a perfect
        // drafter it arrives in the middle of an accepted draft.
        let mut g = generation(30, 6, Mode::Greedy, 0);
        g.stop = vec![full[9]];
        let (out, _) = t.generate(&mut pool, &mut drafter, &prompt(), &g);
        let first = full.iter().position(|&x| x == full[9]).unwrap();
        assert_eq!(out, full[..first]);
    }
}
