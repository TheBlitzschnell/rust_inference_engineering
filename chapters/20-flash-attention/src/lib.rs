//! Chapter 20: attention in one pass.
//!
//! - [`online`]: the online softmax, and merging partial results.
//! - [`attention`]: [`flash_attention`], tiled, grouped by KV head, with
//!   split-KV for decoding.
//! - [`with_flash`]: plugs it into a chapter 14 model.

use ch14_kv_cache::{Matrix, Model};

pub mod attention;
pub mod online;

pub use attention::{FlashOptions, MAX_TILE, flash_attention, flash_decode_many};

/// The same model, with [`flash_attention`] instead of the built-in
/// attention.
pub fn with_flash<W: Matrix>(model: Model<W>, options: FlashOptions) -> Model<W> {
    model.with_attention(move |pool, input, out| flash_attention(pool, input, out, &options))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch07_threads::SpinPool;
    use ch13_transformer::{Config, Weights};
    use ch14_kv_cache::{DenseF32, KvCache, Scratch};

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
    }

    fn logits(
        model: &Model<DenseF32>,
        tokens: &[u32],
        chunk: usize,
        pool: &mut SpinPool,
    ) -> Vec<f32> {
        let c = &model.config;
        let mut cache = KvCache::new(c, 128);
        let mut scratch = Scratch::new(c, chunk, 128);
        // Prefill all but the last 3 tokens, then decode those one by one.
        let split = tokens.len() - 3;
        let mut all = model
            .forward_all(pool, &tokens[..split.min(chunk)], &mut cache, &mut scratch)
            .to_vec();
        if split > chunk {
            all = model
                .forward_last(pool, &tokens[chunk..split], &mut cache, &mut scratch)
                .to_vec();
        }
        for &t in &tokens[split..] {
            all.extend_from_slice(model.forward_last(pool, &[t], &mut cache, &mut scratch));
        }
        all
    }

    #[test]
    fn the_64_dimension_kernels_compute_the_same_model() {
        // Heads of 64 dimensions take the AVX-512 path where available.
        let config = Config {
            hidden_size: 128,
            intermediate_size: 256,
            num_heads: 6,
            num_kv_heads: 2,
            head_dim: 64,
            ..Config::tiny()
        };
        let w = Weights::random(&config, 2);
        let tokens: Vec<u32> = (0..70).map(|i| (i * 11 + 5) % 97).collect();
        let mut pool = SpinPool::new(4);
        let want = logits(&Model::from_reference(&w), &tokens, 32, &mut pool);
        let flash = with_flash(Model::from_reference(&w), FlashOptions::default());
        assert!(close(&logits(&flash, &tokens, 32, &mut pool), &want));
    }

    #[test]
    fn decoding_many_sequences_at_once_equals_one_at_a_time() {
        use ch14_kv_cache::AttentionInput;
        let config = Config {
            num_heads: 6,
            num_kv_heads: 2,
            head_dim: 64,
            hidden_size: 128,
            ..Config::tiny()
        };
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let mut pool = SpinPool::new(4);
        // Three caches of different lengths, filled with made-up keys and
        // values in layer 0.
        let lengths = [5, 70, 131];
        let caches: Vec<KvCache> = lengths
            .iter()
            .enumerate()
            .map(|(n, &len)| {
                let mut cache = KvCache::new(&config, 200);
                for pos in 0..len {
                    let row = |salt: usize| -> Vec<f32> {
                        (0..kv_dim)
                            .map(|i| {
                                (((i * 7 + pos * 13 + n * 31 + salt) % 23) as f32 - 11.0) / 9.0
                            })
                            .collect()
                    };
                    for layer in 0..config.num_layers {
                        cache.store(layer, pos, &row(1), &row(2));
                    }
                }
                cache.advance(len);
                cache
            })
            .collect();
        let qs: Vec<Vec<f32>> = (0..3)
            .map(|n| {
                (0..q_dim)
                    .map(|i| (((i * 5 + n * 17) % 19) as f32 - 9.0) / 7.0)
                    .collect()
            })
            .collect();
        // The new token is the last position of each cache.
        let inputs: Vec<AttentionInput<'_>> = caches
            .iter()
            .zip(&qs)
            .map(|(cache, q)| AttentionInput {
                layer: 0,
                start: cache.len() - 1,
                m: 1,
                q,
                cache,
                config: &config,
            })
            .collect();
        for opts in [
            FlashOptions::default(),
            FlashOptions {
                simd: false,
                key_block: 16,
                ..FlashOptions::default()
            },
        ] {
            let mut together = vec![0.0; 3 * q_dim];
            let mut states = Vec::new();
            flash_decode_many(&mut pool, &inputs, &mut together, &opts, &mut states);
            for (input, got) in inputs.iter().zip(together.chunks_exact(q_dim)) {
                let mut alone = vec![0.0; q_dim];
                flash_attention(&mut pool, input, &mut alone, &opts);
                assert!(close(got, &alone), "{opts:?}");
            }
        }
    }

    #[test]
    fn flash_attention_computes_the_same_model() {
        let w = Weights::random(&Config::tiny(), 1);
        let tokens: Vec<u32> = (0..60).map(|i| (i * 7 + 3) % 97).collect();
        let mut pool = SpinPool::new(3);
        let reference = Model::from_reference(&w);
        for opts in [
            FlashOptions::default(),
            FlashOptions {
                query_block: 1,
                key_block: 5,
                key_splits: 3,
                simd: true,
            },
            FlashOptions {
                query_block: 7,
                key_block: 256,
                key_splits: 1,
                simd: false,
            },
        ] {
            let flash = with_flash(Model::from_reference(&w), opts);
            for chunk in [8, 64] {
                let want = logits(&reference, &tokens, chunk, &mut pool);
                let got = logits(&flash, &tokens, chunk, &mut pool);
                assert!(close(&got, &want), "{opts:?}, chunk {chunk}");
            }
        }
    }
}
