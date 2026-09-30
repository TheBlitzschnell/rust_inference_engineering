//! The paged KV cache on SmolLM2-135M-Instruct: memory, the cost of paged
//! attention, prefix caching, preemption.
//!
//! Run with: cargo run --release -p ch24-paged-kv-cache [memory|attention|prefix|preemption]
//! (needs the model: ./tools/download_model.sh)

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model};
use ch15_sampling::SamplingParams;
use ch16_real_model::{
    DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16, model_dir,
};
use ch17_profiling::{map_matrices, measure};
use ch20_flash_attention::FlashOptions;
use ch21_engine_thread::{EngineHandle, Request, Summary, collect};
use ch23_continuous_batching::{BatchConfig, BatchScratch, BatchSeq, PackedBf16, forward_batch};
use ch24_paged_kv_cache::{
    BlockPool, BlockTable, PagedConfig, PagedScratch, PagedSeq, Stats, forward_paged, spawn,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const THREADS: usize = 4;
const TEXT: &str = include_str!("../../15-sampling/data/pride-and-prejudice-1-6.txt");

fn main() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return;
    }
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("memory") {
        memory(&dir, &tok);
    }
    if run("attention") {
        attention(&dir);
    }
    if run("prefix") {
        prefix(&dir, &tok);
    }
    if run("preemption") {
        preemption(&dir, &tok);
    }
}

fn load(dir: &Path) -> Model<PackedBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        |w: DenseBf16| PackedBf16::new(w.values(), w.rows(), w.cols()),
    )
}

fn ms(d: Duration) -> String {
    format!("{:.0} ms", d.as_secs_f64() * 1e3)
}

/// Bytes of keys and values per position for SmolLM2-135M.
fn bytes_per_position(model: &Model<PackedBf16>) -> usize {
    BlockPool::bytes_per_block(&model.config, 1)
}

/// A request to summarize a passage of `words` words of the novel,
/// starting at word `at`.
fn summarize(tok: &Tokenizer, at: usize, words: usize) -> Vec<u32> {
    let passage: Vec<&str> = TEXT.split_whitespace().skip(at).take(words).collect();
    let question = format!(
        "Summarize this passage in one sentence:\n\n{}",
        passage.join(" ")
    );
    tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: &question,
    }]))
}

/// Submits `(prompt, max_tokens)` requests, all at once, and waits.
fn run_all(engine: &EngineHandle, work: &[(Vec<u32>, usize)]) -> (Vec<Summary>, Duration) {
    let start = Instant::now();
    let receivers: Vec<_> = work
        .iter()
        .map(|(p, n)| {
            engine
                .submit(Request {
                    prompt: p.clone(),
                    params: SamplingParams::greedy(),
                    max_tokens: *n,
                    stop_tokens: Vec::new(),
                })
                .expect("engine running")
        })
        .collect();
    let summaries = receivers
        .into_iter()
        .map(|r| collect(r).expect("request").1)
        .collect();
    (summaries, start.elapsed())
}

fn report(name: &str, summaries: &[Summary], wall: Duration) {
    let tokens: usize = summaries.iter().map(|s| s.completion_tokens).sum();
    let mut ttft: Vec<Duration> = summaries.iter().map(|s| s.time_to_first_token).collect();
    ttft.sort();
    println!(
        "   {name}: {tokens} tokens in {}, {:.0} tokens/s; first token: median {}, slowest {}",
        ms(wall),
        tokens as f64 / wall.as_secs_f64(),
        ms(ttft[ttft.len() / 2]),
        ms(ttft[ttft.len() - 1])
    );
}

/// Part 1: the same memory, as slots or as blocks. The budget holds two
/// full-context slots; the requests are short, so most of a slot would be
/// empty.
fn memory(dir: &Path, tok: &Tokenizer) {
    let model = load(dir);
    let per_position = bytes_per_position(&model);
    // 16 requests: passages of 10 to 40 words, answers of 64 to 124 tokens.
    let work: Vec<(Vec<u32>, usize)> = (0..16)
        .map(|i| {
            (
                summarize(tok, i * 500, 10 + (i * 7) % 31),
                64 + (i * 13) % 61,
            )
        })
        .collect();
    let used: usize = work.iter().map(|(p, n)| p.len() + n).sum();
    let (context, positions) = (1024, 2048);
    println!(
        "== 1. 16 requests needing {used} positions in all; {} MB of KV cache ({positions} positions)",
        (positions * per_position) >> 20
    );
    println!(
        "   {per_position} bytes per position: 2 slots of {context} positions, or {} blocks of 16",
        positions / 16
    );
    // Chapter 23: one slot of the full context per request.
    let slots = ch23_continuous_batching::spawn(
        load(dir),
        BatchConfig {
            threads: THREADS,
            max_batch: positions / context,
            context,
            max_step_tokens: 512,
        },
    )
    .0;
    let (s, wall) = run_all(&slots, &work);
    report("2 slots   ", &s, wall);
    let (paged, _thread, stats) = spawn(
        load(dir),
        PagedConfig {
            threads: THREADS,
            max_batch: 16,
            context,
            max_step_tokens: 512,
            num_blocks: positions / 16,
            block_size: 16,
            prefix_caching: false,
        },
    );
    let (s, wall) = run_all(&paged, &work);
    report("128 blocks", &s, wall);
    println!(
        "      up to {} requests in a step, up to {} of 128 blocks in use, {} preemptions",
        Stats::get(&stats.peak_batch),
        Stats::get(&stats.peak_blocks),
        Stats::get(&stats.preemptions)
    );
    println!();
}

/// Part 2: what reading keys block by block costs. One decode step for 8
/// sequences with 1,024 tokens of context each.
fn attention(dir: &Path) {
    println!("== 2. one decode step, 8 sequences with 1,024 tokens of context each");
    let model = load(dir);
    let mut pool = SpinPool::with_all_cores();
    let opts = FlashOptions::default();
    let context = 1024;
    let prompt: Vec<u32> = (0..context as u32)
        .map(|i| (i * 131 + 7) % 49_000)
        .collect();
    let token = [100u32];

    // Contiguous (chapter 23): prefill one cache, copy it.
    let mut first = KvCache::new(&model.config, context + 8);
    let mut scratch = BatchScratch::new(&model.config, context, 8);
    forward_batch(
        &model,
        &mut pool,
        &mut [BatchSeq {
            tokens: &prompt,
            cache: &mut first,
            logits: false,
        }],
        &mut scratch,
        &opts,
    );
    let mut caches = vec![first; 8];
    let stats = measure(2, 10, || {
        let mut seqs: Vec<BatchSeq<'_>> = caches
            .iter_mut()
            .map(|cache| {
                cache.truncate(context);
                BatchSeq {
                    tokens: &token,
                    cache,
                    logits: true,
                }
            })
            .collect();
        forward_batch(&model, &mut pool, &mut seqs, &mut scratch, &opts);
    });
    println!(
        "   contiguous caches:      {:.1} ms",
        stats.median.as_secs_f64() * 1e3
    );

    for block_size in [16, 64, 256] {
        // Each sequence prefills its own blocks.
        let blocks = 8 * (context + 8).div_ceil(block_size);
        let mut kv = BlockPool::new(&model.config, blocks, block_size);
        let mut scratch = PagedScratch::new(&model.config, context, 8);
        let mut tables: Vec<BlockTable> = (0..8).map(|_| BlockTable::default()).collect();
        for t in &mut tables {
            assert!(t.reserve(&mut kv, context + 1));
            forward_paged(
                &model,
                &mut pool,
                &mut kv,
                &mut [PagedSeq {
                    tokens: &prompt,
                    table: t,
                    logits: false,
                }],
                &mut scratch,
                &opts,
            );
        }
        let stats = measure(2, 10, || {
            let mut seqs: Vec<PagedSeq<'_>> = tables
                .iter_mut()
                .map(|table| {
                    table.len = context;
                    PagedSeq {
                        tokens: &token,
                        table,
                        logits: true,
                    }
                })
                .collect();
            forward_paged(&model, &mut pool, &mut kv, &mut seqs, &mut scratch, &opts);
        });
        println!(
            "   blocks of {block_size:>3} positions: {:.1} ms",
            stats.median.as_secs_f64() * 1e3
        );
    }
    println!();
}

/// A long system prompt shared by every request, then a short question.
fn with_system(tok: &Tokenizer, system: &str, question: &str) -> Vec<u32> {
    tok.encode(&chat_prompt(&[
        Message {
            role: "system",
            content: system,
        },
        Message {
            role: "user",
            content: question,
        },
    ]))
}

/// Part 3: eight requests sharing a long system prompt, 300 ms apart.
fn prefix(dir: &Path, tok: &Tokenizer) {
    let words: Vec<&str> = TEXT.split_whitespace().take(450).collect();
    let system = format!(
        "You answer questions about this text, briefly.\n\n{}",
        words.join(" ")
    );
    let questions = [
        "Who is Mr. Bingley?",
        "Where does the story take place?",
        "What does Mrs. Bennet want?",
        "How many daughters are there?",
        "Who is Mr. Darcy?",
        "What is Netherfield?",
        "What is the first sentence about?",
        "Who speaks first?",
    ];
    let prompts: Vec<Vec<u32>> = questions
        .iter()
        .map(|q| with_system(tok, &system, q))
        .collect();
    println!(
        "== 3. eight requests sharing a {}-token system prompt, arriving 300 ms apart",
        prompts[0].len() - with_system(tok, "", questions[0]).len()
    );
    for caching in [false, true] {
        let (engine, _thread, stats) = spawn(
            load(dir),
            PagedConfig {
                threads: THREADS,
                max_batch: 16,
                context: 2048,
                max_step_tokens: 512,
                num_blocks: 1024,
                block_size: 16,
                prefix_caching: caching,
            },
        );
        let start = Instant::now();
        let results: Vec<Summary> = std::thread::scope(|scope| {
            let clients: Vec<_> = prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let engine = &engine;
                    scope.spawn(move || {
                        std::thread::sleep(Duration::from_millis(300) * i as u32);
                        let events = engine
                            .submit(Request {
                                prompt: p.clone(),
                                params: SamplingParams::greedy(),
                                max_tokens: 32,
                                stop_tokens: Vec::new(),
                            })
                            .expect("engine running");
                        collect(events).expect("request").1
                    })
                })
                .collect();
            clients
                .into_iter()
                .map(|c| c.join().expect("client"))
                .collect()
        });
        let ttft: Vec<String> = results.iter().map(|s| ms(s.time_to_first_token)).collect();
        println!(
            "   prefix caching {}: first token after {}",
            if caching { "on " } else { "off" },
            ttft.join(", ")
        );
        println!(
            "      {} of {} prompt tokens taken from the cache; all done after {}",
            Stats::get(&stats.cached_tokens),
            Stats::get(&stats.prompt_tokens),
            ms(start.elapsed())
        );
    }
    println!();
}

/// Part 4: more demand than memory. Eight requests of about 60 prompt
/// tokens and 200 new tokens (about 2,100 positions in all) with room for
/// 800 positions: requests are preempted and recomputed.
fn preemption(dir: &Path, tok: &Tokenizer) {
    println!("== 4. eight requests needing about 2,100 positions, with room for 800 or for 4,096");
    let work: Vec<(Vec<u32>, usize)> = (0..8).map(|i| (summarize(tok, i * 900, 30), 200)).collect();
    let mut answers: Vec<Vec<u32>> = Vec::new();
    for blocks in [50, 256] {
        let (engine, _thread, stats): (EngineHandle, _, Arc<Stats>) = spawn(
            load(dir),
            PagedConfig {
                threads: THREADS,
                max_batch: 8,
                context: 1024,
                max_step_tokens: 256,
                num_blocks: blocks,
                block_size: 16,
                prefix_caching: false,
            },
        );
        let start = Instant::now();
        let receivers: Vec<_> = work
            .iter()
            .map(|(p, n)| {
                engine
                    .submit(Request {
                        prompt: p.clone(),
                        params: SamplingParams::greedy(),
                        max_tokens: *n,
                        stop_tokens: Vec::new(),
                    })
                    .expect("engine running")
            })
            .collect();
        let outputs: Vec<Vec<u32>> = receivers
            .into_iter()
            .map(|r| collect(r).expect("request").0)
            .collect();
        let tokens: usize = outputs.iter().map(Vec::len).sum();
        println!(
            "   {blocks:>3} blocks of 16: {tokens} tokens in {}, {} preemptions, {} positions recomputed, up to {} requests per step",
            ms(start.elapsed()),
            Stats::get(&stats.preemptions),
            Stats::get(&stats.recomputed_tokens),
            Stats::get(&stats.peak_batch)
        );
        answers.push(outputs.concat());
    }
    println!(
        "   the answers are {}",
        if answers[0] == answers[1] {
            "identical"
        } else {
            "different"
        }
    );
}
