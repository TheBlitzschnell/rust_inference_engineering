//! An OpenAI-compatible server for SmolLM2-135M-Instruct.
//!
//! ```text
//! cargo run --release -p ch22-http-server -- serve [--port 8080] [--max-in-flight 8]
//! cargo run --release -p ch22-http-server -- demo
//! ```
//!
//! `serve` runs until Ctrl-C. `demo` starts the server on a free local port
//! and exercises it with the client from `client.rs`.
//! (needs the model: ./tools/download_model.sh)

use ch16_real_model::{Placement, Tokenizer, load_bf16, model_dir};
use ch17_profiling::{TiledBf16, map_matrices};
use ch20_flash_attention::{FlashOptions, with_flash};
use ch21_engine_thread::spawn;
use ch22_http_server::client;
use ch22_http_server::{AppState, ServerConfig, serve, shutdown_signal};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const THREADS: usize = 4;
const CONTEXT: usize = 2048;

fn main() -> ExitCode {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            dir.display()
        );
        return ExitCode::FAILURE;
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let port = value("--port").map_or(Ok(8080), |v| v.parse::<u16>());
    let max_in_flight = value("--max-in-flight").map_or(Ok(8), |v| v.parse::<usize>());
    let (Ok(port), Ok(max_in_flight)) = (port, max_in_flight) else {
        eprintln!("--port and --max-in-flight take numbers");
        return ExitCode::FAILURE;
    };
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    match args.first().map(String::as_str) {
        Some("serve") => rt.block_on(run_server(&dir, port, max_in_flight)),
        Some("demo") | None => rt.block_on(demo(&dir)),
        Some(other) => {
            eprintln!("unknown command {other:?}: use serve or demo");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

/// Loads SmolLM2 (tiled `bf16`, flash attention), starts its engine thread
/// and builds the handlers' state. The engine stops when the last handle
/// (inside the state) is dropped, after finishing queued requests.
fn state(dir: &Path, max_in_flight: usize) -> AppState {
    let (model, info) = load_bf16(dir, Placement::Mapped).expect("model");
    let model = with_flash(map_matrices(model, TiledBf16), FlashOptions::default());
    let (engine, _thread) = spawn(model, THREADS, CONTEXT);
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    AppState::new(
        engine,
        tokenizer,
        info.eos_tokens,
        ServerConfig {
            model_name: "SmolLM2-135M-Instruct".into(),
            context: CONTEXT,
            max_in_flight,
        },
    )
}

async fn run_server(dir: &Path, port: u16, max_in_flight: usize) {
    let state = state(dir, max_in_flight);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("port in use?");
    eprintln!(
        "listening on http://127.0.0.1:{port} (at most {max_in_flight} requests in flight); Ctrl-C stops"
    );
    serve(listener, state, shutdown_signal())
        .await
        .expect("server");
    eprintln!("stopped");
}

async fn demo(dir: &Path) {
    let state = state(dir, 4);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("address");
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(serve(listener, state, async move {
        let _ = stopped.await;
    }));
    println!("server on http://{addr}, at most 4 requests in flight\n");
    whole(addr).await;
    streamed(addr).await;
    overhead(addr).await;
    overload(addr).await;
    shutdown(addr, stop, server).await;
}

fn chat_body(question: &str, max_tokens: usize, stream: bool) -> String {
    json!({
        "model": "SmolLM2-135M-Instruct",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream
    })
    .to_string()
}

async fn post(addr: SocketAddr, body: &str) -> client::Response {
    client::send(addr, "POST", "/v1/chat/completions", Some(body))
        .await
        .expect("request")
}

/// Whether a streamed event carries some of the answer's text.
fn has_text(data: &str) -> bool {
    serde_json::from_str::<Value>(data).is_ok_and(|v| {
        v["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    })
}

fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1e3)
}

/// Part 1: one request, the whole answer as one JSON document.
async fn whole(addr: SocketAddr) {
    println!("== 1. POST /v1/chat/completions");
    let start = Instant::now();
    let r = post(
        addr,
        &chat_body("What is the capital of France?", 32, false),
    )
    .await;
    let status = r.status;
    let body: Value = serde_json::from_str(&r.text().await.expect("body")).expect("JSON");
    println!("   status {status} after {}", ms(start.elapsed()));
    for line in serde_json::to_string_pretty(&body).expect("JSON").lines() {
        println!("   {line}");
    }
    println!();
}

/// Part 2: the same API streamed: the raw events, with their arrival times.
async fn streamed(addr: SocketAddr) {
    println!("== 2. the same with \"stream\": true (server-sent events as they arrive)");
    let start = Instant::now();
    let r = post(addr, &chat_body("Name three primary colors.", 24, true)).await;
    println!(
        "   status {} after {}, content-type {}",
        r.status,
        ms(start.elapsed()),
        r.header("content-type").unwrap_or("?")
    );
    let mut events = r.events();
    let mut all = Vec::new();
    let mut first_text = None;
    while let Some(data) = events.next().await.expect("events") {
        if first_text.is_none() && has_text(&data) {
            first_text = Some(start.elapsed());
        }
        all.push((start.elapsed(), data));
    }
    let show = |(t, data): &(Duration, String)| println!("   {:>9}  data: {data}", ms(*t));
    all.iter().take(4).for_each(show);
    println!("   ...");
    all.iter().skip(all.len().saturating_sub(3)).for_each(show);
    println!(
        "   {} events; first text after {}",
        all.len(),
        first_text.map_or_else(|| "-".into(), ms)
    );
    println!();
}

/// Part 3: what HTTP itself costs: `GET /health` round trips (a new TCP
/// connection each time), no model involved.
async fn overhead(addr: SocketAddr) {
    println!("== 3. GET /health, 200 times in a row (new connection each time)");
    let mut times = Vec::new();
    for _ in 0..200 {
        let start = Instant::now();
        let r = client::send(addr, "GET", "/health", None)
            .await
            .expect("health");
        r.text().await.expect("body");
        times.push(start.elapsed());
    }
    times.sort();
    println!(
        "   median {}, 99th percentile {}",
        ms(times[times.len() / 2]),
        ms(times[times.len() * 99 / 100])
    );
    println!();
}

/// Part 4: ten clients at once against a limit of four in flight.
async fn overload(addr: SocketAddr) {
    println!("== 4. ten clients at once, streaming 32 tokens each, 4 allowed in flight");
    let questions = [
        "What is the capital of Italy?",
        "Name a large ocean.",
        "What is 7 times 8?",
        "Who painted the Mona Lisa?",
        "What color is the sky?",
        "Name a planet.",
        "What is ice made of?",
        "How many legs does a spider have?",
        "What is the capital of Japan?",
        "Name a fruit.",
    ];
    let start = Instant::now();
    let clients: Vec<_> = questions
        .iter()
        .map(|q| {
            let body = chat_body(q, 32, true);
            tokio::spawn(async move {
                let r = post(addr, &body).await;
                let status = r.status;
                let answered = start.elapsed();
                let mut first_text = None;
                let mut events = r.events();
                while let Some(data) = events.next().await.expect("events") {
                    if first_text.is_none() && has_text(&data) {
                        first_text = Some(start.elapsed());
                    }
                }
                (status, answered, first_text, start.elapsed())
            })
        })
        .collect();
    println!("   client  status  response after  first text after  done after");
    for (i, c) in clients.into_iter().enumerate() {
        let (status, answered, first, done) = c.await.expect("client");
        println!(
            "   {i:>6}  {status:>6}  {:>14}  {:>16}  {:>10}",
            ms(answered),
            first.map_or_else(|| "-".into(), ms),
            ms(done)
        );
    }
    println!();
}

/// Part 5: graceful shutdown while a request is streaming.
async fn shutdown(
    addr: SocketAddr,
    stop: oneshot::Sender<()>,
    server: tokio::task::JoinHandle<std::io::Result<()>>,
) {
    println!("== 5. shutdown while a request is streaming");
    let mut events = post(addr, &chat_body("Describe a forest.", 48, true))
        .await
        .events();
    events.next().await.expect("events");
    let asked = Instant::now();
    stop.send(()).expect("server running");
    let mut n = 0;
    let mut last = String::new();
    while let Some(data) = events.next().await.expect("events") {
        n += 1;
        last = data;
    }
    println!(
        "   the request kept streaming: {n} more events, the last {last:?}, {} after shutdown began",
        ms(asked.elapsed())
    );
    server.await.expect("server task").expect("server");
    println!(
        "   server stopped {} after shutdown began",
        ms(asked.elapsed())
    );
    let refused = tokio::net::TcpStream::connect(addr).await.is_err();
    println!("   new connections refused: {refused}");
}
