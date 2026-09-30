//! Continuous batching on SmolLM2-135M-Instruct: what a step costs as the
//! batch grows, one-at-a-time against batched serving, and whether
//! batching changes the answers.
//!
//! Run with: cargo run --release -p ch23-continuous-batching [steps|serving|arrivals|invariance]
//! (needs the model: ./tools/download_model.sh)

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model};
use ch15_sampling::SamplingParams;
use ch16_real_model::{
    DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16, model_dir,
};
use ch17_profiling::{TiledBf16, instrument, map_matrices, measure};
use ch20_flash_attention::{FlashOptions, with_flash};
use ch21_engine_thread::{EngineHandle, Request, Summary, collect};
use ch23_continuous_batching::{
    BatchConfig, BatchScratch, BatchSeq, PackedBf16, forward_batch, spawn,
};
use std::path::Path;
use std::time::{Duration, Instant};

const THREADS: usize = 4;
const CONTEXT: usize = 1024;

fn main() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("steps") {
        steps(&dir);
    }
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    if run("serving") {
        serving(&dir, &tok);
    }
    if run("arrivals") {
        arrivals(&dir, &tok);
    }
    if run("invariance") {
        invariance(&dir, &tok);
    }
}

/// SmolLM2 with chapter 17's tiled `bf16` weights. The batched forward
/// pass calls flash attention itself; `with_flash` is for chapter 21's
/// engine.
fn load(dir: &Path) -> Model<TiledBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        TiledBf16,
    )
}

/// SmolLM2 with every matrix repacked for small batches.
fn load_packed(dir: &Path) -> Model<PackedBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        |w: DenseBf16| PackedBf16::new(w.values(), w.rows(), w.cols()),
    )
}

fn ms(d: Duration) -> String {
    format!("{:.0} ms", d.as_secs_f64() * 1e3)
}

/// Part 1: one decode step for B sequences, each with 256 tokens of
/// context, with both weight layouts.
fn steps(dir: &Path) {
    println!("== 1. one decode step for B sequences (256 tokens of context each, 4 threads)");
    println!("   chapter 17's tiled bf16 kernel:");
    steps_with(load(dir));
    println!("   packed bf16 weights:");
    steps_with(load_packed(dir));
    println!();
}

fn steps_with<W: Matrix>(model: Model<W>) {
    // Chapter 17's timer around every matrix: how much of a step is matrix
    // products, and how much is everything else (mostly attention).
    let (model, timings) = instrument(model);
    let model = &model;
    let mut pool = SpinPool::with_all_cores();
    let context = 256;
    let most = 64;
    // Prefill one sequence, then copy its cache: every sequence has the
    // same 256 tokens of context, which is all the step's cost depends on.
    let prompt: Vec<u32> = (0..context as u32)
        .map(|i| (i * 131 + 7) % 49_000)
        .collect();
    let mut first = KvCache::new(&model.config, context + 16);
    let mut scratch = BatchScratch::new(&model.config, context.max(most), most);
    let opts = FlashOptions::default();
    forward_batch(
        model,
        &mut pool,
        &mut [BatchSeq {
            tokens: &prompt,
            cache: &mut first,
            logits: false,
        }],
        &mut scratch,
        &opts,
    );
    let mut caches = vec![first; most];
    let mut base = None;
    println!("         B   step time   tokens/s   vs B = 1   matrix products   the rest");
    for b in [1, 2, 4, 8, 16, 32, 64] {
        let tokens: Vec<[u32; 1]> = (0..b as u32).map(|i| [100 + i]).collect();
        let mut run = || {
            let mut seqs: Vec<BatchSeq<'_>> = caches[..b]
                .iter_mut()
                .zip(&tokens)
                .map(|(cache, t)| {
                    cache.truncate(context);
                    BatchSeq {
                        tokens: t,
                        cache,
                        logits: true,
                    }
                })
                .collect();
            forward_batch(model, &mut pool, &mut seqs, &mut scratch, &opts);
        };
        run();
        run();
        timings.reset();
        let runs = 10;
        let stats = measure(0, runs, run);
        let step = stats.median;
        let matmul: Duration =
            timings.totals().iter().map(|t| t.time).sum::<Duration>() / runs as u32;
        let rate = b as f64 / step.as_secs_f64();
        let one = *base.get_or_insert(rate);
        let ms1 = |d: Duration| format!("{:.1} ms", d.as_secs_f64() * 1e3);
        println!(
            "      {b:>4} {:>11} {rate:>10.0} {:>9.1}x {:>17} {:>10}",
            ms1(step),
            rate / one,
            ms1(matmul),
            ms1(step.saturating_sub(matmul))
        );
    }
}

fn question_prompts(tok: &Tokenizer) -> Vec<Vec<u32>> {
    [
        "What is the capital of Italy?",
        "Name a large ocean.",
        "Explain what a noun is.",
        "Who painted the Mona Lisa?",
        "Why do leaves change color in autumn?",
        "Name three planets.",
        "What is ice made of?",
        "How many legs does a spider have?",
    ]
    .iter()
    .map(|q| {
        tok.encode(&chat_prompt(&[Message {
            role: "user",
            content: q,
        }]))
    })
    .collect()
}

/// Submits every prompt at once and waits for all; returns each request's
/// summary and tokens, and the wall time.
fn run_all(
    engine: &EngineHandle,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    stop_tokens: &[u32],
) -> (Vec<(Vec<u32>, Summary)>, Duration) {
    let start = Instant::now();
    let receivers: Vec<_> = prompts
        .iter()
        .map(|p| {
            engine
                .submit(Request {
                    prompt: p.clone(),
                    params: SamplingParams::greedy(),
                    max_tokens,
                    stop_tokens: stop_tokens.to_vec(),
                })
                .expect("engine running")
        })
        .collect();
    let results = receivers
        .into_iter()
        .map(|r| collect(r).expect("request"))
        .collect();
    (results, start.elapsed())
}

/// Part 2: eight clients at once, 64 tokens each, on chapter 21's engine
/// and on the batching engine.
fn serving(dir: &Path, tok: &Tokenizer) {
    println!("== 2. eight clients at once, 64 tokens each");
    let prompts = question_prompts(tok);
    // Both with packed weights and flash attention: the only difference is
    // batching.
    let one_at_a_time = ch21_engine_thread::spawn(
        with_flash(load_packed(dir), FlashOptions::default()),
        THREADS,
        CONTEXT,
    )
    .0;
    let batched = spawn(
        load_packed(dir),
        BatchConfig {
            threads: THREADS,
            max_batch: 8,
            context: CONTEXT,
            max_step_tokens: 512,
        },
    )
    .0;
    for (name, engine) in [("one at a time", &one_at_a_time), ("batched", &batched)] {
        // No stop tokens, so every answer is exactly 64 tokens.
        let (results, wall) = run_all(engine, &prompts, 64, &[]);
        let tokens: usize = results.iter().map(|(t, _)| t.len()).sum();
        println!(
            "   {name}: {tokens} tokens in {}, {:.0} tokens/s",
            ms(wall),
            tokens as f64 / wall.as_secs_f64()
        );
        println!("      client  first token      done");
        for (i, (_, s)) in results.iter().enumerate() {
            println!(
                "      {i:>6} {:>12} {:>9}",
                ms(s.time_to_first_token),
                ms(s.total)
            );
        }
    }
    println!();
}

/// Part 3: clients arriving one after another. With continuous batching a
/// newcomer joins the running batch at the next step.
fn arrivals(dir: &Path, tok: &Tokenizer) {
    println!("== 3. eight clients arriving 150 ms apart, 64 tokens each");
    let prompts = question_prompts(tok);
    let one_at_a_time = ch21_engine_thread::spawn(
        with_flash(load_packed(dir), FlashOptions::default()),
        THREADS,
        CONTEXT,
    )
    .0;
    let batched = spawn(
        load_packed(dir),
        BatchConfig {
            threads: THREADS,
            max_batch: 8,
            context: CONTEXT,
            max_step_tokens: 512,
        },
    )
    .0;
    for (name, engine) in [("one at a time", &one_at_a_time), ("batched", &batched)] {
        let start = Instant::now();
        let results: Vec<(Duration, Summary)> = std::thread::scope(|scope| {
            let clients: Vec<_> = prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    scope.spawn(move || {
                        std::thread::sleep(Duration::from_millis(150) * i as u32);
                        let arrived = start.elapsed();
                        let events = engine
                            .submit(Request {
                                prompt: p.clone(),
                                params: SamplingParams::greedy(),
                                max_tokens: 64,
                                stop_tokens: Vec::new(),
                            })
                            .expect("engine running");
                        (arrived, collect(events).expect("request").1)
                    })
                })
                .collect();
            clients
                .into_iter()
                .map(|c| c.join().expect("client"))
                .collect()
        });
        println!("   {name}:");
        println!("      client  arrives   waits  first token after  done at");
        for (i, (arrived, s)) in results.iter().enumerate() {
            println!(
                "      {i:>6} {:>8} {:>7} {:>18} {:>8}",
                ms(*arrived),
                ms(s.queued),
                ms(s.time_to_first_token),
                ms(*arrived + s.total)
            );
        }
    }
    println!();
}

/// Part 4: does batching change the numbers? Logits of one request alone
/// and inside a batch of eight, then whole greedy answers, for both weight
/// layouts.
fn invariance(dir: &Path, tok: &Tokenizer) {
    println!("== 4. request 0 alone vs in a batch of eight: are the results the same?");
    let prompts = question_prompts(tok);
    let eos = load_bf16(dir, Placement::Mapped)
        .expect("model")
        .1
        .eos_tokens;
    println!("   chapter 17's tiled bf16 kernel:");
    invariance_with(&|| load(dir), &prompts, &eos, tok);
    println!("   packed bf16 weights:");
    invariance_with(&|| load_packed(dir), &prompts, &eos, tok);
}

fn invariance_with<W: Matrix + 'static>(
    load: &dyn Fn() -> Model<W>,
    prompts: &[Vec<u32>],
    eos: &[u32],
    tok: &Tokenizer,
) {
    let model = load();
    let mut pool = SpinPool::with_all_cores();
    let mut scratch = BatchScratch::new(&model.config, 512, 8);
    let opts = FlashOptions::default();
    let new_cache = || KvCache::new(&model.config, CONTEXT);

    // Prefill: prompt 0 alone, then all eight prompts in one step.
    let mut alone_cache = new_cache();
    let alone = forward_batch(
        &model,
        &mut pool,
        &mut [BatchSeq {
            tokens: &prompts[0],
            cache: &mut alone_cache,
            logits: true,
        }],
        &mut scratch,
        &opts,
    )
    .to_vec();
    let mut caches: Vec<KvCache> = prompts.iter().map(|_| new_cache()).collect();
    let mut seqs: Vec<BatchSeq<'_>> = caches
        .iter_mut()
        .zip(prompts)
        .map(|(cache, p)| BatchSeq {
            tokens: p,
            cache,
            logits: true,
        })
        .collect();
    let together = forward_batch(&model, &mut pool, &mut seqs, &mut scratch, &opts);
    drop(seqs);
    report("prefill logits", &alone, &together[..alone.len()]);

    // One decode step: the same token, alone and in the batch.
    let token = [100u32];
    let alone = forward_batch(
        &model,
        &mut pool,
        &mut [BatchSeq {
            tokens: &token,
            cache: &mut alone_cache,
            logits: true,
        }],
        &mut scratch,
        &opts,
    )
    .to_vec();
    let mut seqs: Vec<BatchSeq<'_>> = caches
        .iter_mut()
        .map(|cache| BatchSeq {
            tokens: &token,
            cache,
            logits: true,
        })
        .collect();
    let together = forward_batch(&model, &mut pool, &mut seqs, &mut scratch, &opts);
    drop(seqs);
    report("decode logits ", &alone, &together[..alone.len()]);

    // Whole answers through the engine: each alone, then all together.
    let engine = |max_batch| {
        spawn(
            load(),
            BatchConfig {
                threads: THREADS,
                max_batch,
                context: CONTEXT,
                max_step_tokens: 512,
            },
        )
        .0
    };
    let (alone, _) = run_all(&engine(1), prompts, 96, eos);
    let (together, _) = run_all(&engine(8), prompts, 96, eos);
    let mut same = 0;
    for (i, ((a, _), (b, _))) in alone.iter().zip(&together).enumerate() {
        if a == b {
            same += 1;
            continue;
        }
        let at = a.iter().zip(b).take_while(|(x, y)| x == y).count();
        let around = |t: &[u32]| {
            tok.decode(&t[at.saturating_sub(4)..(at + 6).min(t.len())])
                .replace('\n', " ")
        };
        println!(
            "      answer {i} differs from token {at}: alone \"...{}\", batched \"...{}\"",
            around(a),
            around(b)
        );
    }
    println!(
        "      greedy answers of up to 96 tokens: {same} of {} identical",
        prompts.len()
    );
}

/// How two rows of logits differ.
fn report(what: &str, a: &[f32], b: &[f32]) {
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let largest = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    if differ == 0 {
        println!("      {what}: bit for bit identical");
    } else {
        println!(
            "      {what}: {differ} of {} differ, by at most {largest:.1e}",
            a.len()
        );
    }
}
