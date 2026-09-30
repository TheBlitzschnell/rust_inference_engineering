//! Chapter 22: an OpenAI-compatible HTTP server in front of chapter 21's
//! engine.
//!
//! - [`api`]: the JSON types of the chat completions API.
//! - [`router`]: `POST /v1/chat/completions` (whole answers or streamed as
//!   server-sent events), `GET /v1/models`, `GET /health`.
//! - [`TextStream`]: engine tokens to text, with stop strings.
//! - [`client`]: a minimal HTTP client for the demo and the tests.
//!
//! The server holds no model. Handlers validate requests, turn messages
//! into prompt tokens, pass them to the engine thread and relay what comes
//! back; many requests can be connected at once while the engine runs one.

pub mod api;
pub mod client;

use api::{
    ChatChunk, ChatCompletion, ChatMessage, ChatRequest, Choice, ChunkChoice, Delta, ErrorBody,
    ErrorDetail, ModelCard, ModelList, Usage,
};
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ch11_tokenization::StreamDecoder;
use ch15_sampling::{FinishReason, Matched, SamplingParams, StopMatcher};
use ch16_real_model::{Message, Tokenizer, chat_prompt};
use ch21_engine_thread::{EngineHandle, Event, Request};
use futures_util::{Stream, StreamExt, stream};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Reported in responses and by `/v1/models`.
    pub model_name: String,
    /// The engine's context length: prompt plus answer, in tokens.
    pub context: usize,
    /// Requests admitted at once, running or waiting in the engine's queue.
    /// Beyond that, requests are refused with `429 Too Many Requests`.
    pub max_in_flight: usize,
}

/// Everything the handlers share.
struct Shared {
    engine: EngineHandle,
    tokenizer: Tokenizer,
    /// Tokens that end the model's turn.
    stop_tokens: Vec<u32>,
    config: ServerConfig,
    admission: Arc<Semaphore>,
    next_id: AtomicU64,
}

/// The handlers' state: cheap to clone (one `Arc`).
#[derive(Clone)]
pub struct AppState(Arc<Shared>);

impl AppState {
    pub fn new(
        engine: EngineHandle,
        tokenizer: Tokenizer,
        stop_tokens: Vec<u32>,
        config: ServerConfig,
    ) -> Self {
        let admission = Arc::new(Semaphore::new(config.max_in_flight));
        Self(Arc::new(Shared {
            engine,
            tokenizer,
            stop_tokens,
            config,
            admission,
            next_id: AtomicU64::new(1),
        }))
    }

    /// Requests currently admitted (running or queued).
    pub fn in_flight(&self) -> usize {
        self.0.config.max_in_flight - self.0.admission.available_permits()
    }
}

/// The routes.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
}

/// Serves until `shutdown` completes, then stops accepting connections and
/// waits for the requests in progress to finish.
pub async fn serve(
    listener: TcpListener,
    state: AppState,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}

/// Resolves on Ctrl-C, or on SIGTERM (what `docker stop` and Kubernetes
/// send) on Unix.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

/// An error response: an HTTP status and OpenAI's `{"error": {...}}` body.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: Option<&'static str>,
    message: String,
}

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: None,
            message: message.into(),
        }
    }

    fn too_long(message: String) -> Self {
        Self {
            code: Some("context_length_exceeded"),
            ..Self::invalid(message)
        }
    }

    fn busy() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "server_overloaded",
            code: None,
            message: "the server is at capacity; retry later".into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            code: None,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorBody {
            error: ErrorDetail {
                message: self.message,
                kind: self.kind,
                code: self.code,
            },
        });
        let mut response = (self.status, body).into_response();
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            // Tells well-behaved clients how long to back off, in seconds.
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

/// `POST /v1/chat/completions`.
async fn chat(
    State(state): State<AppState>,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    // Malformed JSON, or JSON that does not fit `ChatRequest`: axum's
    // message says which field is wrong.
    let Json(req) = body.map_err(|e| ApiError::invalid(e.body_text()))?;
    let shared = &state.0;
    let prepared = prepare(shared, req)?;
    // Admission control: a bounded number of requests in flight. Queueing
    // without a limit would only turn overload into timeouts for everyone.
    let permit = Arc::clone(&shared.admission)
        .try_acquire_owned()
        .map_err(|_| ApiError::busy())?;
    let events = shared
        .engine
        .submit(prepared.request)
        .map_err(|_| ApiError::unavailable("the engine has stopped"))?;
    let text = TextStream {
        events,
        state: state.clone(),
        decoder: StreamDecoder::new(),
        stops: StopMatcher::new(&prepared.stops),
        tokens: 0,
        prompt_tokens: prepared.prompt_tokens,
        queue: VecDeque::new(),
        finished: false,
        _permit: permit,
    };
    let meta = Meta {
        id: format!(
            "chatcmpl-{}",
            shared.next_id.fetch_add(1, Ordering::Relaxed)
        ),
        created: unix_time(),
        model: prepared.model,
    };
    if prepared.stream {
        Ok(stream_response(text, meta, prepared.include_usage).into_response())
    } else {
        Ok(Json(complete(text, meta).await?).into_response())
    }
}

/// A validated request, ready for the engine.
struct Prepared {
    request: Request,
    prompt_tokens: usize,
    stops: Vec<String>,
    stream: bool,
    include_usage: bool,
    model: String,
}

/// Checks the request and builds the engine's `Request`. Everything that
/// can be refused is refused here, before the request takes a place in
/// the queue.
fn prepare(shared: &Shared, req: ChatRequest) -> Result<Prepared, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::invalid("`messages` must not be empty"));
    }
    if let Some(m) = req
        .messages
        .iter()
        .find(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
    {
        return Err(ApiError::invalid(format!("unknown role {:?}", m.role)));
    }
    if req.n.is_some_and(|n| n != 1) {
        return Err(ApiError::invalid("only `n` = 1 is supported"));
    }
    let temperature = req.temperature.unwrap_or(1.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::invalid("`temperature` must be between 0 and 2"));
    }
    let top_p = req.top_p.unwrap_or(1.0);
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::invalid("`top_p` must be in (0, 1]"));
    }
    let stops = req.stop.map(api::Stop::into_vec).unwrap_or_default();
    if stops.len() > 4 {
        return Err(ApiError::invalid("at most 4 stop sequences"));
    }

    let messages: Vec<Message<'_>> = req
        .messages
        .iter()
        .map(|m| Message {
            role: &m.role,
            content: &m.content,
        })
        .collect();
    let prompt = shared.tokenizer.encode(&chat_prompt(&messages));
    let context = shared.config.context;
    let room = context.saturating_sub(prompt.len());
    if room == 0 {
        return Err(ApiError::too_long(format!(
            "the prompt is {} tokens; this model's context is {context}",
            prompt.len()
        )));
    }
    let max_tokens = req.max_tokens.unwrap_or(room);
    if max_tokens == 0 {
        return Err(ApiError::invalid("`max_tokens` must be at least 1"));
    }
    if max_tokens > room {
        return Err(ApiError::too_long(format!(
            "the prompt is {} tokens and `max_tokens` is {max_tokens}; \
             together they exceed this model's context of {context}",
            prompt.len()
        )));
    }
    let params = SamplingParams {
        temperature,
        top_p,
        top_k: req.top_k.unwrap_or(0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        // Without a seed, each request gets a different one.
        seed: req.seed.unwrap_or_else(|| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            let n = shared.next_id.load(Ordering::Relaxed);
            (now.as_secs() << 30)
                ^ u64::from(now.subsec_nanos())
                ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        }),
        ..SamplingParams::default()
    };
    Ok(Prepared {
        prompt_tokens: prompt.len(),
        request: Request {
            prompt,
            params,
            max_tokens,
            stop_tokens: shared.stop_tokens.clone(),
        },
        stops,
        stream: req.stream,
        include_usage: req.stream_options.is_some_and(|o| o.include_usage),
        model: req
            .model
            .unwrap_or_else(|| shared.config.model_name.clone()),
    })
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// What a response needs besides the text.
struct Meta {
    id: String,
    created: u64,
    model: String,
}

impl Meta {
    fn chunk(&self, delta: Delta, finish_reason: Option<&'static str>) -> ChatChunk {
        ChatChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        }
    }
}

/// What [`TextStream::next`] produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    /// More of the answer.
    Text(String),
    /// The answer is complete.
    End {
        finish_reason: &'static str,
        usage: Usage,
    },
    /// The request failed after it was accepted.
    Failed(String),
}

/// One request's engine events, turned into text: tokens are decoded
/// without splitting UTF-8 characters, and stop strings end the answer.
///
/// Owns the admission permit, so the request counts as in flight exactly
/// as long as this exists. Dropping it (the client went away) drops the
/// event receiver, which cancels the request in the engine.
pub struct TextStream {
    events: UnboundedReceiver<Event>,
    state: AppState,
    decoder: StreamDecoder,
    stops: StopMatcher,
    tokens: usize,
    prompt_tokens: usize,
    queue: VecDeque<Piece>,
    finished: bool,
    _permit: OwnedSemaphorePermit,
}

impl TextStream {
    /// The next piece, or `None` after `End` or `Failed`.
    pub async fn next(&mut self) -> Option<Piece> {
        loop {
            if let Some(piece) = self.queue.pop_front() {
                return Some(piece);
            }
            if self.finished {
                return None;
            }
            match self.events.recv().await {
                Some(Event::Token(t)) => {
                    self.tokens += 1;
                    let text = self.decoder.push(self.state.0.tokenizer.token_bytes(t));
                    if self.push_text(&text) {
                        // A stop string: end now, and close the channel so
                        // the engine stops at its next token.
                        self.events.close();
                        self.end("stop");
                    }
                }
                Some(Event::Done(summary)) => {
                    let rest = std::mem::take(&mut self.decoder).finish();
                    let stopped = self.push_text(&rest);
                    if !stopped {
                        let held = self.stops.finish();
                        self.queue_text(held);
                    }
                    self.end(match summary.finish {
                        FinishReason::Length if !stopped => "length",
                        _ => "stop",
                    });
                }
                Some(Event::Rejected(why)) => self.fail(why),
                None => self.fail("the engine stopped".into()),
            }
        }
    }

    /// Passes new text through the stop matcher; true if a stop string
    /// was found.
    fn push_text(&mut self, text: &str) -> bool {
        match self.stops.push(text) {
            Matched::Continue(s) => {
                self.queue_text(s);
                false
            }
            Matched::Stop(s) => {
                self.queue_text(s);
                true
            }
        }
    }

    fn queue_text(&mut self, s: String) {
        if !s.is_empty() {
            self.queue.push_back(Piece::Text(s));
        }
    }

    fn end(&mut self, finish_reason: &'static str) {
        self.queue.push_back(Piece::End {
            finish_reason,
            usage: Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.tokens,
                total_tokens: self.prompt_tokens + self.tokens,
            },
        });
        self.finished = true;
    }

    fn fail(&mut self, why: String) {
        self.queue.push_back(Piece::Failed(why));
        self.finished = true;
    }
}

/// The whole answer as one JSON response.
async fn complete(mut text: TextStream, meta: Meta) -> Result<ChatCompletion, ApiError> {
    let mut content = String::new();
    while let Some(piece) = text.next().await {
        match piece {
            Piece::Text(s) => content.push_str(&s),
            Piece::End {
                finish_reason,
                usage,
            } => {
                return Ok(ChatCompletion {
                    id: meta.id,
                    object: "chat.completion",
                    created: meta.created,
                    model: meta.model,
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content,
                        },
                        finish_reason,
                    }],
                    usage,
                });
            }
            Piece::Failed(why) => return Err(ApiError::unavailable(why)),
        }
    }
    Err(ApiError::unavailable("the engine stopped"))
}

/// The answer as server-sent events: a chunk with the role, one per piece
/// of text, one with the finish reason, optionally one with the usage,
/// then `[DONE]`.
fn stream_response(
    text: TextStream,
    meta: Meta,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let json = |chunk: &ChatChunk| {
        SseEvent::default().data(serde_json::to_string(chunk).expect("chunks serialize"))
    };
    let first = json(&meta.chunk(
        Delta {
            role: Some("assistant"),
            content: Some(String::new()),
        },
        None,
    ));
    // `unfold` turns "call `next` again and again" into a `Stream`. Each
    // step yields the events for one piece.
    let rest = stream::unfold((text, meta), move |(mut text, meta)| async move {
        let events = match text.next().await? {
            Piece::Text(s) => vec![json(&meta.chunk(
                Delta {
                    content: Some(s),
                    ..Delta::default()
                },
                None,
            ))],
            Piece::End {
                finish_reason,
                usage,
            } => {
                let mut events = vec![json(&meta.chunk(Delta::default(), Some(finish_reason)))];
                if include_usage {
                    let mut last = meta.chunk(Delta::default(), None);
                    last.choices.clear();
                    last.usage = Some(usage);
                    events.push(json(&last));
                }
                events.push(SseEvent::default().data("[DONE]"));
                events
            }
            Piece::Failed(why) => {
                let body = ErrorBody {
                    error: ErrorDetail {
                        message: why,
                        kind: "server_error",
                        code: None,
                    },
                };
                vec![SseEvent::default().data(serde_json::to_string(&body).expect("serializes"))]
            }
        };
        Some((events, (text, meta)))
    })
    .flat_map(|events| stream::iter(events.into_iter().map(Ok)));
    // Keep-alive comments stop proxies from closing a connection that is
    // silent while its request waits in the queue.
    Sse::new(stream::once(async move { Ok(first) }).chain(rest)).keep_alive(KeepAlive::default())
}

/// `GET /v1/models`.
async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.0.config.model_name.clone(),
            object: "model",
            created: 0,
            owned_by: "rust-inference-engineering",
        }],
    })
}

/// `GET /health`: for load balancers and orchestrators.
async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "in_flight": state.in_flight(),
        "max_in_flight": state.0.config.max_in_flight,
    }))
}
