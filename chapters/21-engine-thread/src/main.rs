//! The engine thread with SmolLM2-135M-Instruct: streaming, queueing,
//! cancellation, and why the model must not run on the async runtime.
//!
//! Run with: cargo run --release -p ch21-engine-thread [stream|queue|cancel|idle|runtime]
//! (needs the model: ./tools/download_model.sh)

use ch07_threads::SpinPool;
use ch11_tokenization::StreamDecoder;
use ch14_kv_cache::{KvCache, Model, Scratch};
use ch15_sampling::{Sampler, SamplingParams, generate};
use ch16_real_model::{Message, Placement, Tokenizer, chat_prompt, load_bf16, model_dir};
use ch17_profiling::{TiledBf16, map_matrices};
use ch20_flash_attention::{FlashOptions, with_flash};
use ch21_engine_thread::{EngineHandle, Event, Request, Summary, spawn};
use std::io::Write;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedReceiver;

/// Engine threads and context length used throughout.
const THREADS: usize = 4;
const CONTEXT: usize = 2048;

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
    let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let (model, eos) = load(&dir);
    let (engine, thread) = spawn(model, THREADS, CONTEXT);
    // An async runtime with a single thread: the clients below are tasks on
    // it, like the request handlers of chapter 22's server.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime");
    let ask = |question: &str, max_tokens: usize| Request {
        prompt: tok.encode(&chat_prompt(&[Message {
            role: "user",
            content: question,
        }])),
        params: SamplingParams::greedy(),
        max_tokens,
        stop_tokens: eos.clone(),
    };
    if run("stream") {
        rt.block_on(stream(&engine, &tok, ask("Why is the sky blue?", 64)));
    }
    if run("queue") {
        rt.block_on(queue(&engine, &ask));
    }
    if run("cancel") {
        rt.block_on(cancel(&engine, &ask));
    }
    if run("idle") {
        idle();
    }
    if run("runtime") {
        rt.block_on(on_engine(&engine, ask("Describe the ocean.", 32)));
    }
    // Dropping the last handle stops the engine; wait for it to finish.
    drop(engine);
    thread.join().expect("engine thread");
    if run("runtime") {
        rt.block_on(inline(&dir, &ask("Describe the ocean.", 32)));
    }
}

/// A duration in milliseconds, with one decimal below 10 ms.
fn ms(d: Duration) -> String {
    let v = d.as_secs_f64() * 1e3;
    if v < 10.0 {
        format!("{v:.1} ms")
    } else {
        format!("{v:.0} ms")
    }
}

/// SmolLM2 with chapter 17's tiled `bf16` weights and chapter 20's flash
/// attention (the fastest `bf16` model so far), and its end-of-turn tokens.
fn load(dir: &Path) -> (Model<TiledBf16>, Vec<u32>) {
    let (model, info) = load_bf16(dir, Placement::Mapped).expect("model");
    let model = with_flash(map_matrices(model, TiledBf16), FlashOptions::default());
    (model, info.eos_tokens)
}

/// Part 1: one request, printed as its tokens arrive.
async fn stream(engine: &EngineHandle, tok: &Tokenizer, request: Request) {
    println!("== 1. one request, streamed");
    let mut events = engine.submit(request).expect("engine running");
    let mut decoder = StreamDecoder::new();
    print!("   ");
    let s = loop {
        match events.recv().await {
            Some(Event::Token(t)) => {
                print!("{}", decoder.push(tok.token_bytes(t)));
                std::io::stdout().flush().expect("stdout");
            }
            Some(Event::Done(s)) => break s,
            Some(Event::Rejected(why)) => panic!("rejected: {why}"),
            None => panic!("the engine stopped"),
        }
    };
    println!("{}", decoder.finish());
    let decode = s.total.saturating_sub(s.time_to_first_token);
    println!(
        "   {} prompt tokens, {} generated; first token after {}, then {:.1} tokens/s ({:?})",
        s.prompt_tokens,
        s.completion_tokens,
        ms(s.time_to_first_token),
        s.completion_tokens.saturating_sub(1) as f64 / decode.as_secs_f64(),
        s.finish
    );
    println!();
}

/// Part 2: four clients at once. The engine serves one request at a time,
/// so each waits for everyone ahead of it.
async fn queue(engine: &EngineHandle, ask: &impl Fn(&str, usize) -> Request) {
    println!("== 2. four clients submit at the same moment, 32 tokens each");
    let questions = [
        "What is the capital of France?",
        "Name three primary colors.",
        "What is 12 times 12?",
        "Who wrote Pride and Prejudice?",
    ];
    let receivers: Vec<_> = questions
        .iter()
        .map(|q| engine.submit(ask(q, 32)).expect("engine running"))
        .collect();
    println!("   client     queued  first token      total  tokens");
    for (i, events) in receivers.into_iter().enumerate() {
        // The channels are unbounded, so reading them one after another
        // does not slow the engine down.
        let s = summary(events).await;
        println!(
            "   {i:>6} {:>10} {:>12} {:>10} {:>7}",
            ms(s.queued),
            ms(s.time_to_first_token),
            ms(s.total),
            s.completion_tokens
        );
    }
    println!();
}

/// Part 3: a client that goes away. Client A asks for 200 tokens; client B
/// is queued behind it.
async fn cancel(engine: &EngineHandle, ask: &impl Fn(&str, usize) -> Request) {
    println!("== 3. client A asks for 200 tokens, client B is queued behind it");
    // No stop tokens: A really runs 200 tokens unless cancelled.
    let mut long = ask("Write a long story about a lighthouse keeper.", 200);
    long.stop_tokens.clear();

    // First, A reads everything.
    let a = engine.submit(long.clone()).expect("engine running");
    let b = engine.submit(ask("Say hello.", 8)).expect("engine running");
    let a_done = summary(a).await;
    let b_done = summary(b).await;
    println!(
        "   A reads all {} tokens:   B waits {:>8} to start",
        a_done.completion_tokens,
        ms(b_done.queued)
    );

    // Then A drops its receiver after 10 tokens.
    let mut a = engine.submit(long).expect("engine running");
    let b_submitted = Instant::now();
    let b = engine.submit(ask("Say hello.", 8)).expect("engine running");
    let mut seen = 0;
    while seen < 10 {
        match a.recv().await {
            Some(Event::Token(_)) => seen += 1,
            Some(_) | None => break,
        }
    }
    let left = Instant::now();
    drop(a);
    let b_done = summary(b).await;
    let b_started = b_submitted + b_done.queued;
    println!(
        "   A leaves after {seen} tokens: B waits {:>8} to start, {} after A left",
        ms(b_done.queued),
        ms(b_started.saturating_duration_since(left))
    );
    println!();
}

/// Waits for a request's final event.
async fn summary(mut events: UnboundedReceiver<Event>) -> Summary {
    while let Some(event) = events.recv().await {
        match event {
            Event::Token(_) => {}
            Event::Done(s) => return s,
            Event::Rejected(why) => panic!("rejected: {why}"),
        }
    }
    panic!("the engine stopped");
}

/// Part 5a: a timer task ticks every 10 ms on the runtime while the
/// engine thread generates 32 tokens.
async fn on_engine(engine: &EngineHandle, request: Request) {
    println!("== 5. a 10 ms timer on the async runtime while 32 tokens are generated");
    let stop = Arc::new(AtomicBool::new(false));
    let timer = tokio::spawn(ticker(Arc::clone(&stop)));
    let s = summary(engine.submit(request).expect("engine running")).await;
    stop.store(true, Ordering::Relaxed);
    let gap = timer.await.expect("timer task");
    println!(
        "   on the engine thread:  generation {:>7}, longest gap between ticks {:>7}",
        ms(s.total),
        ms(gap)
    );
}

/// Part 5b: the same generation inside an async task, after the engine has
/// stopped (so the two do not compete for cores). The runtime's only thread
/// is busy, so no other task (the timer, other clients) runs meanwhile.
async fn inline(dir: &Path, request: &Request) {
    let (model, _) = load(dir);
    let mut pool = SpinPool::new(THREADS);
    let mut cache = KvCache::new(&model.config, CONTEXT);
    let mut scratch = Scratch::new(&model.config, 256, CONTEXT);
    let stop = Arc::new(AtomicBool::new(false));
    let timer = tokio::spawn(ticker(Arc::clone(&stop)));
    // Let the timer start before blocking.
    tokio::time::sleep(Duration::from_millis(15)).await;
    let start = Instant::now();
    generate(
        &model,
        &mut pool,
        &request.prompt,
        &mut Sampler::new(request.params.clone()),
        &mut cache,
        &mut scratch,
        request.max_tokens,
        &request.stop_tokens,
        |_| ControlFlow::Continue(()),
    );
    let took = start.elapsed();
    stop.store(true, Ordering::Relaxed);
    let gap = timer.await.expect("timer task");
    println!(
        "   inside an async task:  generation {:>7}, longest gap between ticks {:>7}",
        ms(took),
        ms(gap)
    );
    println!();
}

/// Part 4: what waiting costs. CPU time used by this process during one
/// second in which no request runs.
fn idle() {
    println!("== 4. CPU time used during 1 s with no requests");
    let second = || {
        let before = cpu_time();
        std::thread::sleep(Duration::from_secs(1));
        cpu_time()
            .zip(before)
            .map(|(after, before)| after.saturating_sub(before))
    };
    let show = |what: &str, used: Option<Duration>| match used {
        Some(d) => println!("   {what}: {:.2} s of CPU per second", d.as_secs_f64()),
        None => println!("   {what}: not measured (needs Linux's /proc)"),
    };
    show("the engine, waiting for requests ", second());
    let pool = SpinPool::new(THREADS);
    show("plus an idle SpinPool (4 threads)", second());
    drop(pool);
    let start = Instant::now();
    for _ in 0..100 {
        drop(SpinPool::new(THREADS));
    }
    println!(
        "   starting and stopping a 4-thread pool: {:.0} µs",
        start.elapsed().as_secs_f64() * 1e6 / 100.0
    );
    println!();
}

/// User plus system CPU time of this process so far, from `/proc/self/stat`
/// (fields 14 and 15, in clock ticks of 1/100 s on Linux).
fn cpu_time() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name (field 2) is in parentheses and may contain spaces;
    // fields are counted from after its closing parenthesis (field 3).
    let rest = &stat[stat.rfind(')')? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks: u64 = fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?;
    Some(Duration::from_millis(ticks * 10))
}

/// Sleeps 10 ms at a time until `stop` is set; returns the longest time
/// between two wake-ups.
async fn ticker(stop: Arc<AtomicBool>) -> Duration {
    let mut last = Instant::now();
    let mut longest = Duration::ZERO;
    while !stop.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(10)).await;
        longest = longest.max(last.elapsed());
        last = Instant::now();
    }
    longest
}
