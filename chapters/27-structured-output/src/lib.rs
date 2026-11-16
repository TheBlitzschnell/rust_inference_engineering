//! Chapter 27: structured output.
//!
//! - [`json`]: a byte-level JSON matcher that follows a schema, and knows
//!   the shortest way to finish what it has read.
//! - [`mask`]: which tokens may come next, by testing each token or by
//!   walking a trie of the vocabulary.
//! - [`constrained`]: generation with the mask applied at every step.

pub mod constrained;
pub mod json;
pub mod mask;

pub use constrained::{Output, Vocab, generate_json};
pub use json::{Matcher, Schema};
pub use mask::{TokenTrie, allowed_naive};

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Model, Scratch};
    use ch15_sampling::{Rng, Sampler, SamplingParams};

    /// A toy vocabulary for the tiny model's 97 tokens: 0 is the end of
    /// sequence, 1 to 95 the printable ASCII characters, 96 a two-byte
    /// token.
    fn toy_vocab() -> Vec<Vec<u8>> {
        let mut v = vec![Vec::new()];
        v.extend((0x20..=0x7e_u8).map(|b| vec![b]));
        v.push(b"{\"".to_vec());
        v
    }

    fn schema() -> Schema {
        Schema::Object(vec![
            ("id".into(), Schema::Integer),
            ("ok".into(), Schema::Boolean),
            ("kind".into(), Schema::Enum(vec!["a".into(), "bb".into()])),
            ("xs".into(), Schema::Array(Box::new(Schema::Number))),
            ("note".into(), Schema::String { max_len: 12 }),
            ("extra".into(), Schema::Any),
        ])
    }

    #[test]
    fn the_trie_mask_equals_the_naive_mask() {
        // Random multi-byte tokens over JSON's characters, and the matcher
        // in many states: after each prefix of a few documents.
        let alphabet = b"{}[]\",:0123-.eEtrufalsn kidxbo\\\xc3\xa9\xe4\xb8";
        let mut rng = Rng::new(5);
        let tokens: Vec<Vec<u8>> = (0..3000)
            .map(|_| {
                let len = 1 + (rng.next_u64() % 5) as usize;
                (0..len)
                    .map(|_| alphabet[(rng.next_u64() % alphabet.len() as u64) as usize])
                    .collect()
            })
            .collect();
        let vocab: Vec<(u32, &[u8])> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u32, t.as_slice()))
            .collect();
        let trie = TokenTrie::new(vocab.iter().copied());
        let s = schema();
        let doc = "{\"id\": -12, \"ok\": false, \"kind\": \"bb\", \"xs\": [1.5e3, 0], \"note\": \"a\\\"\u{e9}b\", \"extra\": {\"k\": [null, \"x\u{4e16}\"]}}".as_bytes();
        for schema in [&s, &Schema::Any] {
            for cut in 0..=doc.len() {
                let mut m = Matcher::new(schema);
                assert!(m.push_all(&doc[..cut]));
                let (mut a, mut b) = (vec![false; tokens.len()], vec![false; tokens.len()]);
                let mut c = vec![false; tokens.len()];
                trie.allowed(&m, &mut a);
                allowed_naive(&vocab, &m, &mut b);
                trie.walk(&m, &mut c);
                assert_eq!(a, b, "after {:?}", String::from_utf8_lossy(&doc[..cut]));
                assert_eq!(c, b, "after {:?}", String::from_utf8_lossy(&doc[..cut]));
            }
        }
    }

    fn generate(seed: u64, temperature: f32, max_tokens: usize) -> Output {
        let model: Model<DenseF32> = Model::from_reference(&Weights::random(&Config::tiny(), 3));
        let tokens = toy_vocab();
        let vocab = Vocab::new(tokens.iter().map(Vec::as_slice).collect(), |id| id == 0, 0);
        let mut pool = SpinPool::new(2);
        let mut cache = KvCache::new(&model.config, 256);
        let mut scratch = Scratch::new(&model.config, 16, 256);
        let mut sampler = Sampler::new(SamplingParams {
            temperature,
            seed,
            ..SamplingParams::default()
        });
        let s = schema();
        generate_json(
            &model,
            &mut pool,
            &mut cache,
            &mut scratch,
            &vocab,
            Matcher::new(&s),
            &mut sampler,
            &[5, 6, 7],
            max_tokens,
        )
    }

    fn valid(text: &str) -> bool {
        let s = schema();
        let mut m = Matcher::new(&s);
        serde_json::from_str::<serde_json::Value>(text).is_ok()
            && m.push_all(text.as_bytes())
            && m.is_complete()
    }

    #[test]
    fn a_random_model_always_produces_valid_json() {
        // The tiny model's weights are random: unconstrained, it would
        // print noise. Constrained, every sample is valid.
        for seed in 0..20 {
            let out = generate(seed, 1.0, 150);
            assert!(valid(&out.text), "seed {seed}: {}", out.text);
        }
        let greedy = generate(0, 0.0, 150);
        assert!(valid(&greedy.text), "{}", greedy.text);
    }

    #[test]
    fn a_small_budget_is_completed_by_the_matcher() {
        for max_tokens in [0, 1, 5, 12] {
            let out = generate(1, 1.0, max_tokens);
            assert!(out.tokens.len() <= max_tokens);
            assert!(out.completed > 0, "budget {max_tokens}: {}", out.text);
            assert!(valid(&out.text), "budget {max_tokens}: {}", out.text);
        }
    }
}
