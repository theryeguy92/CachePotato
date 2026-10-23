//! `cachepotato serve`: an OpenAI-compatible API and a browser UI over one model.
//!
//! One inference thread owns the model: the engine is one-model-per-process and pins the
//! thread that opens it, so the model is opened there and never moves. HTTP handlers (tokio,
//! two worker threads, unpinned) send it jobs over a channel and stream tokens back, so
//! requests are served one at a time in arrival order. A client that disconnects stops its
//! own generation after the current token.
//!
//! OpenAI clients resend the whole conversation on every request. The engine keeps the
//! conversation in its KV cache, so the thread remembers what it has fed (`transcript`): when
//! a request's earlier messages are exactly that, only the new user message is processed;
//! otherwise the cache is reset and the earlier messages are replayed before answering.
//!
//! Decoding is greedy, always. Sampling parameters (temperature, top_p, seed, ...) are
//! accepted and ignored rather than rejected, because common clients send them by default.

use std::convert::Infallible;
use std::ffi::c_long;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc as tmpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::doctor;
use crate::ffi::{Model, Options, Stats, Stop};
use crate::models;
use crate::run;

const UI: &str = include_str!("../ui/index.html");

pub struct ServeArgs {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub max_tokens: usize,
    pub budget_mb: Option<u64>,
    pub threads: Option<usize>,
    pub pin_cores: bool,
    pub yes: bool,
}

// ---------------------------------------------------------------------------- state

#[derive(Clone, Serialize)]
struct LastTurn {
    prompt_tokens: usize,
    generated: usize,
    prefill_s: f64,
    decode_s: f64,
    tok_s: f64,
    finish: &'static str,
}

/// What the Monitor shows. Written by the inference thread, read by `/api/stats`.
#[derive(Clone, Default, Serialize)]
struct Snapshot {
    model: String,
    /// loading | idle | prefill | decode
    phase: &'static str,
    queued: usize,
    requests: u64,
    // the forward pass in flight
    layer: usize,
    n_layers: usize,
    prompt_tokens: usize,
    generated: usize,
    tok_s: f64,
    turn_bytes_fetched: i64,
    last_turn: Option<LastTurn>,
    // conversation
    context_tokens: usize,
    max_ctx: usize,
    // machine and pager
    threads: i32,
    pinned: Vec<i32>,
    budget_mb: c_long,
    file_bytes: i64,
    hits: c_long,
    misses: c_long,
    evictions: c_long,
    bytes_fetched: i64,
    resident_bytes: i64,
    #[serde(skip)]
    turn_start_bytes: i64,
}

impl Snapshot {
    fn apply(&mut self, s: &Stats) {
        self.n_layers = s.n_layers as usize;
        self.max_ctx = s.max_ctx as usize;
        self.threads = s.threads;
        self.pinned = s.pinned().to_vec();
        self.budget_mb = s.budget_mb;
        self.file_bytes = s.file_bytes;
        self.hits = s.hits;
        self.misses = s.misses;
        self.evictions = s.evictions;
        self.bytes_fetched = s.bytes_fetched;
        self.resident_bytes = s.resident_bytes;
        self.turn_bytes_fetched = s.bytes_fetched - self.turn_start_bytes;
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Msg {
    role: String,
    content: String,
}

enum Ev {
    Token(String),
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize },
    Error(String),
}

enum Job {
    Chat { messages: Vec<Msg>, max_tokens: usize, events: tmpsc::UnboundedSender<Ev> },
    Reset,
}

struct AppState {
    snap: Arc<Mutex<Snapshot>>,
    jobs: Mutex<mpsc::Sender<Job>>,
    api_key: Option<String>,
    model: String,
    default_max_tokens: usize,
    next_id: AtomicU64,
}

// ---------------------------------------------------------------------------- the inference thread

/// Token bytes can end inside a UTF-8 sequence; JSON needs whole characters.
#[derive(Default)]
struct Utf8Buf {
    pending: Vec<u8>,
}

impl Utf8Buf {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..ok]).unwrap_or_default());
                    match e.error_len() {
                        None => {
                            self.pending.drain(..ok); // an incomplete tail: wait for more
                            return out;
                        }
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..ok + bad);
                        }
                    }
                }
            }
        }
    }
    fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// How an earlier message is replayed into the cache: the same shape the live path leaves
/// behind (cp_chat's user framing; an assistant answer as its bare text).
fn chatml(m: &Msg) -> String {
    match m.role.as_str() {
        "system" => format!("<|im_start|>system\n{}<|im_end|>\n", m.content),
        "user" => format!("<|im_start|>user\n{} /no_think<|im_end|>\n<|im_start|>assistant\n", m.content),
        _ => m.content.clone(),
    }
}

fn inference(path: PathBuf, opts: Options, snap: Arc<Mutex<Snapshot>>, jobs: mpsc::Receiver<Job>, ready: mpsc::Sender<Result<()>>) {
    let mut model = match Model::open(&path, &opts) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    {
        let mut g = snap.lock().unwrap();
        g.apply(&model.stats());
        g.phase = "idle";
    }
    let s2 = snap.clone();
    model.set_progress(move |p| {
        let mut g = s2.lock().unwrap();
        if p.tokens > 1 {
            g.phase = "prefill";
            g.layer = p.layer;
            g.n_layers = p.n_layers;
        }
        if let Some(s) = p.stats {
            g.apply(&s);
        }
    });
    let _ = ready.send(Ok(()));

    let mut transcript: Vec<Msg> = vec![];
    for job in jobs {
        match job {
            Job::Reset => {
                model.reset();
                transcript.clear();
                snap.lock().unwrap().context_tokens = 0;
            }
            Job::Chat { messages, max_tokens, events } => {
                {
                    let mut g = snap.lock().unwrap();
                    g.queued = g.queued.saturating_sub(1);
                }
                chat_job(&mut model, &mut transcript, &snap, messages, max_tokens, &events);
                let mut g = snap.lock().unwrap();
                g.phase = "idle";
                g.requests += 1;
                g.context_tokens = model.context_tokens();
                g.apply(&model.stats());
            }
        }
    }
}

fn chat_job(model: &mut Model, transcript: &mut Vec<Msg>, snap: &Mutex<Snapshot>, messages: Vec<Msg>, max_tokens: usize, events: &tmpsc::UnboundedSender<Ev>) {
    let (last, history) = messages.split_last().expect("validated non-empty");
    {
        let mut g = snap.lock().unwrap();
        g.phase = "prefill";
        g.layer = 0;
        g.generated = 0;
        g.tok_s = 0.0;
        g.turn_start_bytes = g.bytes_fetched;
        g.turn_bytes_fetched = 0;
    }
    let t_start = Instant::now();
    let mut replayed = 0;
    if history != transcript.as_slice() {
        model.reset();
        transcript.clear();
        for m in history {
            match model.prefill_text(&chatml(m)) {
                Ok(n) => replayed += n,
                Err(e) => {
                    model.reset();
                    let _ = events.send(Ev::Error(format!("{e:#} (2048-token context)")));
                    return;
                }
            }
            transcript.push(m.clone());
        }
    }

    let mut utf = Utf8Buf::default();
    let mut answer = String::new();
    let mut started = false; // Qwen3 opens with an empty think block: drop leading blank lines
    let mut first: Option<Instant> = None;
    let mut generated = 0usize;
    let result = model.chat(&last.content, max_tokens, |bytes| {
        let now = Instant::now();
        let t0 = *first.get_or_insert(now);
        generated += 1;
        {
            let mut g = snap.lock().unwrap();
            g.phase = "decode";
            g.generated = generated;
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.0 {
                g.tok_s = (generated - 1) as f64 / dt;
            }
        }
        let mut text = utf.push(bytes);
        if !started {
            let t = text.trim_start();
            if t.is_empty() {
                return true;
            }
            text = t.to_string();
            started = true;
        }
        answer.push_str(&text);
        events.send(Ev::Token(text)).is_ok()
    });

    match result {
        Ok(turn) => {
            let tail = utf.finish();
            if !tail.is_empty() {
                answer.push_str(&tail);
                let _ = events.send(Ev::Token(tail));
            }
            if turn.context_reset != 0 {
                transcript.clear(); // the engine restarted the conversation to make room
            }
            transcript.push(last.clone());
            transcript.push(Msg { role: "assistant".into(), content: answer });
            let finish = match turn.stop {
                Stop::MaxTokens | Stop::Context => "length",
                _ => "stop",
            };
            let prompt_tokens = replayed + turn.prompt_tokens as usize;
            let gen = turn.generated_tokens as usize;
            snap.lock().unwrap().last_turn = Some(LastTurn {
                prompt_tokens,
                generated: gen,
                prefill_s: t_start.elapsed().as_secs_f64() - turn.decode_s,
                decode_s: turn.decode_s,
                tok_s: if turn.decode_s > 0.0 { gen as f64 / turn.decode_s } else { 0.0 },
                finish,
            });
            let _ = events.send(Ev::Done { finish, prompt_tokens, completion_tokens: gen });
        }
        Err(e) => {
            model.reset();
            transcript.clear();
            let _ = events.send(Ev::Error(format!("{e:#}")));
        }
    }
}

// ---------------------------------------------------------------------------- HTTP

fn api_error(status: StatusCode, msg: impl Into<String>) -> Response {
    let kind = if status == StatusCode::UNAUTHORIZED { "authentication_error" } else { "invalid_request_error" };
    (status, Json(json!({ "error": { "message": msg.into(), "type": kind } }))).into_response()
}

fn content_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| (p.get("type")?.as_str()? == "text").then(|| p.get("text")?.as_str()).flatten())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        Value::Null => Some(String::new()),
        _ => None,
    }
}

fn parse_messages(body: &Value) -> std::result::Result<Vec<Msg>, String> {
    let arr = body.get("messages").and_then(Value::as_array).ok_or("`messages` must be an array")?;
    let mut out = vec![];
    for m in arr {
        let role = m.get("role").and_then(Value::as_str).ok_or("each message needs a `role`")?;
        let role = match role {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            other => return Err(format!("role `{other}` is not supported (system, user, assistant)")),
        };
        let content = content_text(m.get("content").unwrap_or(&Value::Null)).ok_or("`content` must be a string or text parts")?;
        if content.contains('\0') {
            return Err("message content contains a NUL byte".into());
        }
        out.push(Msg { role: role.into(), content });
    }
    match out.last() {
        None => Err("`messages` is empty".into()),
        Some(m) if m.role != "user" => Err("the last message must be from the user".into()),
        _ => Ok(out),
    }
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

async fn index() -> Html<&'static str> {
    Html(UI)
}

async fn list_models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({ "object": "list", "data": [ { "id": st.model, "object": "model", "created": 0, "owned_by": "cachepotato" } ] }))
}

async fn stats(State(st): State<Arc<AppState>>) -> Json<Snapshot> {
    Json(st.snap.lock().unwrap().clone())
}

async fn reset(State(st): State<Arc<AppState>>) -> Response {
    match st.jobs.lock().unwrap().send(Job::Reset) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "the model is not running"),
    }
}

async fn chat_completions(State(st): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let messages = match parse_messages(&body) {
        Ok(m) => m,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e),
    };
    let max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .map_or(st.default_max_tokens, |n| n.clamp(1, 1 << 16) as usize);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let (tx, mut rx) = tmpsc::unbounded_channel();
    st.snap.lock().unwrap().queued += 1;
    if st.jobs.lock().unwrap().send(Job::Chat { messages, max_tokens, events: tx }).is_err() {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "the model is not running");
    }
    let id = format!("chatcmpl-{}", st.next_id.fetch_add(1, Ordering::Relaxed));
    let created = unix_now();
    let model = st.model.clone();

    if stream {
        let chunk = move |delta: Value, finish: Option<&str>, usage: Option<Value>| {
            let mut c = json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [ { "index": 0, "delta": delta, "finish_reason": finish } ]
            });
            if let Some(u) = usage {
                c["usage"] = u;
            }
            Event::default().data(c.to_string())
        };
        let first = chunk(json!({ "role": "assistant", "content": "" }), None, None);
        let body = UnboundedReceiverStream::new(rx).map(move |ev| match ev {
            Ev::Token(t) => chunk(json!({ "content": t }), None, None),
            Ev::Done { finish, prompt_tokens, completion_tokens } => chunk(
                json!({}),
                Some(finish),
                Some(json!({ "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens })),
            ),
            Ev::Error(e) => Event::default().data(json!({ "error": { "message": e, "type": "server_error" } }).to_string()),
        });
        let events = tokio_stream::once(first)
            .chain(body)
            .chain(tokio_stream::once(Event::default().data("[DONE]")))
            .map(Ok::<_, Infallible>);
        return Sse::new(events).keep_alive(KeepAlive::default()).into_response();
    }

    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Ev::Token(t) => text.push_str(&t),
            Ev::Error(e) => return api_error(StatusCode::BAD_REQUEST, e),
            Ev::Done { finish, prompt_tokens, completion_tokens } => {
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [ { "index": 0, "message": { "role": "assistant", "content": text },
                                   "finish_reason": finish } ],
                    "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                               "total_tokens": prompt_tokens + completion_tokens }
                }))
                .into_response();
            }
        }
    }
    api_error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended without a result")
}

async fn auth(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if let Some(key) = &st.api_key {
        if req.uri().path() != "/" {
            let ok = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .is_some_and(|v| v == key);
            if !ok {
                return api_error(StatusCode::UNAUTHORIZED, "missing or wrong API key (Authorization: Bearer <key>)");
            }
        }
    }
    next.run(req).await
}

// ---------------------------------------------------------------------------- entry point

pub fn serve(a: ServeArgs) -> Result<()> {
    let home = models::home()?;
    if let Some(b) = doctor::blockers(&doctor::probe(&home)).first() {
        bail!("{b} (see `cachepotato doctor`)");
    }
    let ip: IpAddr = a.host.parse().with_context(|| format!("--host {} is not an IP address", a.host))?;
    if !ip.is_loopback() && a.api_key.is_none() {
        bail!("--host {ip} exposes the server beyond this machine; set --api-key <secret> as well");
    }
    let path = run::locate(&a.model, a.yes)?;
    let name = if Path::new(&a.model).is_file() {
        path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| a.model.clone())
    } else {
        models::resolve(&a.model)?.display()
    };

    let snap = Arc::new(Mutex::new(Snapshot { model: name.clone(), phase: "loading", ..Default::default() }));
    let (job_tx, job_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let opts = Options { budget_mb: a.budget_mb, threads: a.threads, pin_cores: Some(a.pin_cores), tokenizer_dir: None };
    let s2 = snap.clone();
    eprintln!("loading {} …", path.display());
    std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || inference(path, opts, s2, job_rx, ready_tx))?;
    ready_rx.recv().map_err(|_| anyhow!("the inference thread exited during loading"))??;

    {
        let g = snap.lock().unwrap();
        let pinned = if g.pinned.is_empty() {
            String::new()
        } else {
            format!(", pinned to CPUs {}", g.pinned.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","))
        };
        let budget = if g.budget_mb < 0 { "no pager".into() } else { format!("{:.1} GB pager budget", g.budget_mb as f64 / 1024.0) };
        eprintln!("{name}: {} layers, {budget}, {} threads{pinned}.", g.n_layers, g.threads);
    }

    let model_id = name.clone();
    let state = Arc::new(AppState {
        snap,
        jobs: Mutex::new(job_tx),
        api_key: a.api_key,
        model: name,
        default_max_tokens: a.max_tokens,
        next_id: AtomicU64::new(1),
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/stats", get(stats))
        .route("/api/reset", post(reset))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state);

    // two runtime threads: the engine's pinned threads should have the cores to themselves
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(async move {
        let addr = SocketAddr::new(ip, a.port);
        let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("cannot listen on {addr}"))?;
        let shown = if ip.is_loopback() { format!("http://localhost:{}", a.port) } else { format!("http://{addr}") };
        eprintln!();
        eprintln!("  open        {shown}");
        eprintln!("  OpenAI API  {shown}/v1   (model id `{model_id}`)");
        eprintln!("  ctrl-C stops the server.");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .context("server error")
    })
}
