//! Talk to SmolLM2-135M-Instruct, and measure how it runs.
//!
//! ```text
//! cargo run --release -p ch16-real-model -- ask "What is the capital of France?"
//! cargo run --release -p ch16-real-model -- ask --temperature 0.7 --seed 3 "Tell me a joke"
//! cargo run --release -p ch16-real-model -- chat
//! cargo run --release -p ch16-real-model -- bench
//! ```
//!
//! Options for `ask` and `chat`: `--model 135m|360m`,
//! `--weights mapped|aligned|f32`, `--temperature T`, `--top-k K`,
//! `--top-p P`, `--seed S`, `--max-tokens N`, `--system TEXT`.
//! `bench` takes an optional model size: `bench 360m`.

use ch07_threads::SpinPool;
use ch09_safetensors::smollm2_dir;
use ch11_tokenization::StreamDecoder;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::{FinishReason, Sampler, SamplingParams, generate};
use ch16_real_model::{Message, ModelInfo, Placement, Tokenizer, chat_prompt, load_bf16, load_f32};
use std::io::{BufRead, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Everything `ask` and `chat` can be told.
struct Options {
    dir: PathBuf,
    weights: String,
    params: SamplingParams,
    max_tokens: usize,
    system: Option<String>,
    rest: Vec<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("ask") => parse(&args[1..]).and_then(|o| ask(&o)),
        Some("chat") => parse(&args[1..]).and_then(|o| chat(&o)),
        Some("bench") => bench(args.get(1).map_or("135m", String::as_str)),
        Some("bench-one") => bench_one(
            args.get(1).map_or("mapped", String::as_str),
            args.get(2).map_or("135m", String::as_str),
        ),
        _ => Err("usage: ch16-real-model ask [options] QUESTION | chat [options] | bench".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn parse(args: &[String]) -> Result<Options> {
    let mut o = Options {
        dir: smollm2_dir("135m"),
        weights: "mapped".into(),
        params: SamplingParams::greedy(),
        max_tokens: 256,
        system: None,
        rest: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => o.dir = smollm2_dir(value()?),
            "--weights" => o.weights.clone_from(value()?),
            "--temperature" => o.params.temperature = value()?.parse()?,
            "--top-k" => o.params.top_k = value()?.parse()?,
            "--top-p" => o.params.top_p = value()?.parse()?,
            "--seed" => o.params.seed = value()?.parse()?,
            "--max-tokens" => o.max_tokens = value()?.parse()?,
            "--system" => o.system = Some(value()?.clone()),
            _ => o.rest.push(arg.clone()),
        }
    }
    Ok(o)
}

/// Loads the model with the requested weight format and runs `f` on it.
/// `Model<DenseBf16>` and `Model<DenseF32>` are different types, so `f` is
/// generic and compiled once for each.
fn with_model<R>(dir: &Path, weights: &str, f: impl WithModel<R>) -> Result<R> {
    let start = Instant::now();
    match weights {
        "mapped" | "aligned" => {
            let placement = if weights == "mapped" {
                Placement::Mapped
            } else {
                Placement::Aligned
            };
            let (model, info) = load_bf16(dir, placement)?;
            f.run(&model, &info, start.elapsed())
        }
        "f32" => {
            let (model, info) = load_f32(dir)?;
            f.run(&model, &info, start.elapsed())
        }
        other => Err(format!("unknown weight format {other:?} (mapped, aligned or f32)").into()),
    }
}

/// A generic callback: closures cannot be generic over `W`, traits can.
trait WithModel<R> {
    fn run<W: Matrix>(self, model: &Model<W>, info: &ModelInfo, load_time: Duration) -> Result<R>;
}

/// One conversation turn: encode, generate while streaming text to stdout,
/// return the answer and timing.
struct Turn {
    answer: String,
    prompt_tokens: usize,
    new_tokens: usize,
    first_token: Duration,
    total: Duration,
    finish: FinishReason,
}

#[expect(clippy::too_many_arguments, reason = "the pieces of one generation")]
fn turn<W: Matrix>(
    model: &Model<W>,
    info: &ModelInfo,
    tok: &Tokenizer,
    pool: &mut SpinPool,
    cache: &mut KvCache,
    scratch: &mut Scratch,
    options: &Options,
    messages: &[Message<'_>],
) -> Result<Turn> {
    let prompt = tok.encode(&chat_prompt(messages));
    let room = cache.capacity().saturating_sub(prompt.len());
    if room == 0 {
        return Err("the conversation no longer fits in the context".into());
    }
    let mut sampler = Sampler::new(options.params.clone());
    let mut decoder = StreamDecoder::new();
    let mut answer = String::new();
    let mut stdout = std::io::stdout().lock();
    let start = Instant::now();
    let mut first_token = None;
    let mut new_tokens = 0;
    let finish = generate(
        model,
        pool,
        &prompt,
        &mut sampler,
        cache,
        scratch,
        options.max_tokens.min(room),
        &info.eos_tokens,
        |token| {
            first_token.get_or_insert_with(|| start.elapsed());
            new_tokens += 1;
            let text = decoder.push(tok.token_bytes(token));
            answer.push_str(&text);
            // A closed stdout (the reader went away) ends generation.
            if write!(stdout, "{text}")
                .and_then(|()| stdout.flush())
                .is_err()
            {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        },
    );
    let tail = decoder.finish();
    answer.push_str(&tail);
    writeln!(stdout, "{tail}")?;
    Ok(Turn {
        answer,
        prompt_tokens: prompt.len(),
        new_tokens,
        first_token: first_token.unwrap_or_default(),
        total: start.elapsed(),
        finish,
    })
}

fn report(t: &Turn) {
    let decode = t.total.saturating_sub(t.first_token);
    let rate = if t.new_tokens > 1 {
        (t.new_tokens - 1) as f64 / decode.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "[{} prompt tokens, {} new; first token after {:.0?}, then {rate:.1} tokens/s; finish: {:?}]",
        t.prompt_tokens, t.new_tokens, t.first_token, t.finish
    );
}

fn tokenizer(dir: &Path) -> Result<Tokenizer> {
    Ok(Tokenizer::from_file(&dir.join("tokenizer.json"))?)
}

fn ask(options: &Options) -> Result<()> {
    struct Ask<'a>(&'a Options);
    impl WithModel<()> for Ask<'_> {
        fn run<W: Matrix>(self, model: &Model<W>, info: &ModelInfo, load: Duration) -> Result<()> {
            let o = self.0;
            let question = o.rest.join(" ");
            if question.is_empty() {
                return Err("ask needs a question".into());
            }
            eprintln!("[loaded {} weights in {load:.0?}]", o.weights);
            let tok = tokenizer(&o.dir)?;
            let mut messages = Vec::new();
            if let Some(system) = &o.system {
                messages.push(Message {
                    role: "system",
                    content: system,
                });
            }
            messages.push(Message {
                role: "user",
                content: &question,
            });
            let context = 2048;
            let mut pool = SpinPool::with_all_cores();
            let mut cache = KvCache::new(&model.config, context);
            let mut scratch = Scratch::new(&model.config, 256, context);
            let t = turn(
                model,
                info,
                &tok,
                &mut pool,
                &mut cache,
                &mut scratch,
                o,
                &messages,
            )?;
            report(&t);
            Ok(())
        }
    }
    with_model(&options.dir, &options.weights, Ask(options))
}

fn chat(options: &Options) -> Result<()> {
    struct Chat<'a>(&'a Options);
    impl WithModel<()> for Chat<'_> {
        fn run<W: Matrix>(self, model: &Model<W>, info: &ModelInfo, load: Duration) -> Result<()> {
            let o = self.0;
            eprintln!(
                "[loaded {} weights in {load:.0?}; empty line to quit]",
                o.weights
            );
            let tok = tokenizer(&o.dir)?;
            let context = 4096;
            let mut pool = SpinPool::with_all_cores();
            let mut cache = KvCache::new(&model.config, context);
            let mut scratch = Scratch::new(&model.config, 256, context);
            // The whole conversation, as owned strings: (role, content).
            let mut history: Vec<(&str, String)> = Vec::new();
            if let Some(system) = &o.system {
                history.push(("system", system.clone()));
            }
            let stdin = std::io::stdin();
            loop {
                eprint!("> ");
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
                    return Ok(());
                }
                history.push(("user", line.trim().to_owned()));
                let messages: Vec<Message<'_>> = history
                    .iter()
                    .map(|(role, content)| Message { role, content })
                    .collect();
                // Every turn re-processes the whole conversation. Chapter 24
                // keeps the cache between turns instead (prefix caching).
                let t = turn(
                    model,
                    info,
                    &tok,
                    &mut pool,
                    &mut cache,
                    &mut scratch,
                    o,
                    &messages,
                )?;
                report(&t);
                history.push(("assistant", t.answer));
            }
        }
    }
    with_model(&options.dir, &options.weights, Chat(options))
}

/// Runs `bench-one` in a fresh process for each weight format, so that each
/// starts with nothing loaded and its memory use can be measured alone.
fn bench(size: &str) -> Result<()> {
    let exe = std::env::current_exe()?;
    println!(
        "SmolLM2-{}-Instruct, {} threads",
        size.to_uppercase(),
        SpinPool::with_all_cores().threads()
    );
    println!(
        "{:<8} {:>9} {:>11} {:>10} {:>13} {:>12} {:>9} {:>10} {:>10}",
        "weights",
        "load",
        "1st prefill",
        "prefill",
        "prefill tok/s",
        "decode tok/s",
        "GB/s",
        "RSS anon",
        "RSS file"
    );
    for placement in ["mapped", "aligned", "f32"] {
        let status = std::process::Command::new(&exe)
            .args(["bench-one", placement, size])
            .status()?;
        if !status.success() {
            return Err(format!("bench-one {placement} failed").into());
        }
    }
    tokenizer_speed(&smollm2_dir(size))
}

fn bench_one(weights: &str, size: &str) -> Result<()> {
    struct Bench<'a>(&'a str, &'a Path);
    impl WithModel<()> for Bench<'_> {
        fn run<W: Matrix>(self, model: &Model<W>, _: &ModelInfo, load: Duration) -> Result<()> {
            let tok = tokenizer(self.1)?;
            let prompt = tok.encode(&chat_prompt(&[Message {
                role: "user",
                content: "Explain in two sentences why the sky is blue.",
            }]));
            let mut pool = SpinPool::with_all_cores();
            let mut cache = KvCache::new(&model.config, 512);
            let mut scratch = Scratch::new(&model.config, 256, 512);
            let timed = |f: &mut dyn FnMut()| {
                let start = Instant::now();
                f();
                start.elapsed()
            };
            // The first pass touches every weight: for mapped weights that is
            // when the pages are actually read.
            let first = timed(&mut || {
                cache.clear();
                model.forward_last(&mut pool, &prompt, &mut cache, &mut scratch);
            });
            let prefill = (0..3)
                .map(|_| {
                    timed(&mut || {
                        cache.clear();
                        model.forward_last(&mut pool, &prompt, &mut cache, &mut scratch);
                    })
                })
                .min()
                .unwrap_or_default();
            let mut steps: Vec<Duration> = (0..64)
                .map(|i| {
                    timed(&mut || {
                        model.forward_last(
                            &mut pool,
                            &[(i * 37 + 100) % 49_000],
                            &mut cache,
                            &mut scratch,
                        );
                    })
                })
                .collect();
            steps.sort_unstable();
            let decode = steps[steps.len() / 2];
            let (anon, file) = rss();
            println!(
                "{:<8} {:>9.0?} {:>11.0?} {:>10.1?} {:>13.0} {:>12.1} {:>9.1} {:>7} MB {:>7} MB",
                self.0,
                load,
                first,
                prefill,
                prompt.len() as f64 / prefill.as_secs_f64(),
                1.0 / decode.as_secs_f64(),
                model.weight_bytes_per_token() as f64 / decode.as_secs_f64() / 1e9,
                anon / 1024,
                file / 1024,
            );
            Ok(())
        }
    }
    let dir = smollm2_dir(size);
    with_model(&dir, weights, Bench(weights, &dir))
}

/// Resident memory in KiB, split into anonymous (the process's own) and
/// file-backed (mapped files, shared with the page cache). Linux only.
fn rss() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    (field("RssAnon:"), field("RssFile:"))
}

fn tokenizer_speed(dir: &Path) -> Result<()> {
    let start = Instant::now();
    let tok = tokenizer(dir)?;
    let load = start.elapsed();
    let text = include_str!("../../15-sampling/data/pride-and-prejudice-1-6.txt");
    let start = Instant::now();
    let ids = tok.encode(text);
    let t = start.elapsed();
    println!(
        "tokenizer: loaded in {load:.0?}; {} bytes -> {} tokens in {t:.1?} ({:.1} MB/s, {:.2} bytes per token)",
        text.len(),
        ids.len(),
        text.len() as f64 / t.as_secs_f64() / 1e6,
        text.len() as f64 / ids.len() as f64
    );
    Ok(())
}
