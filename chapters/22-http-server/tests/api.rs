//! The server, end to end over real sockets, with a tiny random model and a
//! byte-level tokenizer built here (256 byte tokens plus the two chat
//! markers, no merges).

use ch07_threads::SpinPool;
use ch13_transformer::{Config, Weights};
use ch14_kv_cache::{DenseF32, KvCache, Matrix, Model, Scratch};
use ch15_sampling::{Sampler, SamplingParams, generate};
use ch16_real_model::tokenizer::byte_to_char;
use ch16_real_model::{Message, Tokenizer, chat_prompt};
use ch17_profiling::map_matrices;
use ch21_engine_thread::spawn;
use ch22_http_server::client::{self, Response};
use ch22_http_server::{AppState, ServerConfig, serve};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const END: u32 = 257;
const CONTEXT: usize = 256;

fn config() -> Config {
    Config {
        vocab_size: 258,
        ..Config::tiny()
    }
}

fn tokenizer() -> Tokenizer {
    let chars = byte_to_char();
    let vocab: serde_json::Map<String, Value> =
        (0..256).map(|b| (chars[b].to_string(), json!(b))).collect();
    let file = json!({
        "added_tokens": [
            {"id": 256, "content": "<|im_start|>"},
            {"id": END, "content": "<|im_end|>"}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "model": {"type": "BPE", "vocab": vocab, "merges": []}
    });
    Tokenizer::from_json(&file.to_string()).expect("test tokenizer")
}

/// A matrix that takes at least 200 µs per product, so that requests last
/// long enough to overlap.
struct Slow(DenseF32);

impl Matrix for Slow {
    fn rows(&self) -> usize {
        self.0.rows()
    }
    fn cols(&self) -> usize {
        self.0.cols()
    }
    fn bytes(&self) -> usize {
        self.0.bytes()
    }
    fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        self.0.row_to_f32(r, out);
    }
    fn matmul(&self, pool: &mut SpinPool, x: &[f32], y: &mut [f32], m: usize, s: &mut Vec<f32>) {
        std::thread::sleep(Duration::from_micros(200));
        self.0.matmul(pool, x, y, m, s);
    }
}

struct Server {
    addr: SocketAddr,
    state: AppState,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

/// Starts a server on a free port. `slow` wraps the weights in `Slow`;
/// `stop_at_end` makes `<|im_end|>` end answers (turn it off so that
/// answers run to `max_tokens`).
async fn start(max_in_flight: usize, slow: bool, stop_at_end: bool) -> Server {
    let w = Weights::random(&config(), 7);
    let model = Model::from_reference(&w);
    let engine = if slow {
        spawn(map_matrices(model, Slow), 2, CONTEXT).0
    } else {
        spawn(model, 2, CONTEXT).0
    };
    let state = AppState::new(
        engine,
        tokenizer(),
        if stop_at_end { vec![END] } else { Vec::new() },
        ServerConfig {
            model_name: "tiny".into(),
            context: CONTEXT,
            max_in_flight,
        },
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    let task = tokio::spawn(serve(listener, state.clone(), async move {
        let _ = stopped.await;
    }));
    Server {
        addr,
        state,
        stop: Some(stop),
        task,
    }
}

/// What the model says to "hello" with greedy decoding, computed without
/// the server: (text, number of tokens).
fn direct_answer(max_tokens: usize) -> (String, usize) {
    let tok = tokenizer();
    let model = Model::from_reference(&Weights::random(&config(), 7));
    let prompt = tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: "hello",
    }]));
    let mut pool = SpinPool::new(2);
    let mut cache = KvCache::new(&model.config, CONTEXT);
    let mut scratch = Scratch::new(&model.config, 256, CONTEXT);
    let mut tokens = Vec::new();
    generate(
        &model,
        &mut pool,
        &prompt,
        &mut Sampler::new(SamplingParams::greedy()),
        &mut cache,
        &mut scratch,
        max_tokens,
        &[END],
        |t| {
            tokens.push(t);
            ControlFlow::Continue(())
        },
    );
    (tok.decode(&tokens), tokens.len())
}

fn hello(extra: Value) -> String {
    let mut body = json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0
    });
    let Value::Object(fields) = extra else {
        panic!("extra fields must be an object")
    };
    for (k, v) in fields {
        body[k] = v;
    }
    body.to_string()
}

async fn post(addr: SocketAddr, body: &str) -> Response {
    client::send(addr, "POST", "/v1/chat/completions", Some(body))
        .await
        .unwrap()
}

async fn json_of(r: Response) -> Value {
    serde_json::from_str(&r.text().await.unwrap()).unwrap()
}

#[tokio::test]
async fn a_whole_answer_matches_direct_generation() {
    let server = start(4, false, true).await;
    let (want, n) = direct_answer(24);
    let r = post(server.addr, &hello(json!({"max_tokens": 24}))).await;
    assert_eq!(r.status, 200);
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("application/json")
    );
    let v = json_of(r).await;
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "tiny");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], want.as_str());
    let finish = if n == 24 { "length" } else { "stop" };
    assert_eq!(v["choices"][0]["finish_reason"], finish);
    assert_eq!(v["usage"]["completion_tokens"], n);
    assert_eq!(
        v["usage"]["total_tokens"],
        v["usage"]["prompt_tokens"].as_u64().unwrap() + n as u64
    );
}

#[tokio::test]
async fn a_streamed_answer_has_the_same_text() {
    let server = start(4, false, true).await;
    let (want, n) = direct_answer(24);
    let body =
        hello(json!({"max_tokens": 24, "stream": true, "stream_options": {"include_usage": true}}));
    let r = post(server.addr, &body).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    let mut events = r.events();
    let mut all = Vec::new();
    while let Some(data) = events.next().await.unwrap() {
        all.push(data);
    }
    assert_eq!(all.last().map(String::as_str), Some("[DONE]"));
    let chunks: Vec<Value> = all[..all.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, want);
    // The chunk with the finish reason, then the usage chunk.
    let finish = &chunks[chunks.len() - 2]["choices"][0]["finish_reason"];
    assert_eq!(finish, if n == 24 { "length" } else { "stop" });
    let usage = &chunks[chunks.len() - 1];
    assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
    assert_eq!(usage["usage"]["completion_tokens"], n);
}

#[tokio::test]
async fn a_stop_string_ends_the_answer_before_it() {
    let server = start(4, false, false).await;
    let (full, _) = direct_answer(60);
    let chars: Vec<char> = full.chars().collect();
    assert!(
        chars.len() >= 20,
        "the test needs a longer answer: {full:?}"
    );
    let stop: String = chars[12..15].iter().collect();
    let want = &full[..full.find(&stop).unwrap()];
    let r = post(
        server.addr,
        &hello(json!({"max_tokens": 60, "stop": [stop, "never matches"]})),
    )
    .await;
    let v = json_of(r).await;
    assert_eq!(v["choices"][0]["message"]["content"], want);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn bad_requests_get_400_with_a_reason() {
    let server = start(4, false, true).await;
    let cases = [
        "{not json",
        r#"{"messages": "hi"}"#,
        r#"{"messages": []}"#,
        r#"{"messages": [{"role": "robot", "content": "hi"}]}"#,
        r#"{"messages": [{"role": "user", "content": "hi"}], "n": 2}"#,
        r#"{"messages": [{"role": "user", "content": "hi"}], "temperature": 3}"#,
        r#"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}"#,
    ];
    for body in cases {
        let r = post(server.addr, body).await;
        assert_eq!(r.status, 400, "{body}");
        let v = json_of(r).await;
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|m| !m.is_empty()),
            "{body}"
        );
        assert_eq!(v["error"]["type"], "invalid_request_error", "{body}");
    }
    // Too long for the context: a specific code clients can act on.
    let r = post(server.addr, &hello(json!({"max_tokens": CONTEXT}))).await;
    assert_eq!(r.status, 400);
    assert_eq!(json_of(r).await["error"]["code"], "context_length_exceeded");
}

#[tokio::test]
async fn requests_beyond_capacity_get_429_until_a_place_frees_up() {
    let server = start(1, true, false).await;
    // One long streamed request takes the only place.
    let long = hello(json!({"max_tokens": 100, "stream": true}));
    let mut first = post(server.addr, &long).await.events();
    assert!(first.next().await.unwrap().is_some());
    assert_eq!(server.state.in_flight(), 1);

    let r = post(server.addr, &hello(json!({"max_tokens": 2}))).await;
    assert_eq!(r.status, 429);
    assert_eq!(r.header("retry-after"), Some("1"));
    assert_eq!(json_of(r).await["error"]["type"], "server_overloaded");

    // The client of the first request goes away: its place is freed (and
    // the engine stops generating for it) without waiting for 100 tokens.
    let left = Instant::now();
    drop(first);
    while server.state.in_flight() > 0 {
        assert!(left.elapsed() < Duration::from_secs(5), "place never freed");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let r = post(server.addr, &hello(json!({"max_tokens": 2}))).await;
    assert_eq!(r.status, 200);
}

#[tokio::test]
async fn models_and_health() {
    let server = start(3, false, true).await;
    let r = client::send(server.addr, "GET", "/v1/models", None)
        .await
        .unwrap();
    assert_eq!(json_of(r).await["data"][0]["id"], "tiny");
    let r = client::send(server.addr, "GET", "/health", None)
        .await
        .unwrap();
    let v = json_of(r).await;
    assert_eq!(v["status"], "ok");
    assert_eq!(v["in_flight"], 0);
    assert_eq!(v["max_in_flight"], 3);
}

#[tokio::test]
async fn shutdown_lets_the_running_request_finish() {
    let mut server = start(2, true, false).await;
    let mut events = post(
        server.addr,
        &hello(json!({"max_tokens": 20, "stream": true})),
    )
    .await
    .events();
    assert!(events.next().await.unwrap().is_some());
    server.stop.take().unwrap().send(()).unwrap();
    // The stream still runs to the end.
    let mut last = None;
    while let Some(data) = events.next().await.unwrap() {
        last = Some(data);
    }
    assert_eq!(last.as_deref(), Some("[DONE]"));
    server.task.await.unwrap().unwrap();
    // And the server no longer accepts connections.
    assert!(tokio::net::TcpStream::connect(server.addr).await.is_err());
}

#[tokio::test]
async fn a_client_that_leaves_before_a_whole_answer_frees_its_place() {
    use tokio::io::AsyncWriteExt;
    let server = start(1, true, false).await;
    // A request for a whole (not streamed) answer of 100 tokens, which the
    // slow model needs well over 300 ms for.
    let body = hello(json!({"max_tokens": 100}));
    let mut stream = tokio::net::TcpStream::connect(server.addr).await.unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(body.as_bytes()).await.unwrap();
    while server.state.in_flight() == 0 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let left = Instant::now();
    drop(stream);
    while server.state.in_flight() > 0 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // hyper notices the closed connection and drops the handler, which
    // drops the request's `TextStream`: typically within a few ms.
    assert!(left.elapsed() < Duration::from_millis(200));
}
