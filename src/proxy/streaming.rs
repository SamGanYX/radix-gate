use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_util::Stream;
use parking_lot::RwLock;
use serde_json::json;

use crate::balancer::RoutingDecision;
use crate::shadow_cache::WorkerReplica;

/// RAII guard ensuring a worker's active in-flight request counter is decremented
/// exactly once, even if a streaming connection terminates prematurely or panics.
pub struct InFlightGuard {
    worker: Arc<RwLock<WorkerReplica>>,
    active: bool,
}

impl InFlightGuard {
    pub fn new(worker: Arc<RwLock<WorkerReplica>>) -> Self {
        worker.read().active_requests.fetch_add(1, Ordering::SeqCst);
        Self {
            worker,
            active: true,
        }
    }

    /// Creates a guard for a worker whose active in-flight counter was already incremented.
    pub fn from_existing(worker: Arc<RwLock<WorkerReplica>>) -> Self {
        Self {
            worker,
            active: true,
        }
    }

    /// Disarms the guard when manual decrement is explicitly performed.
    pub fn disarm(&mut self) {
        if self.active {
            self.worker
                .read()
                .active_requests
                .fetch_sub(1, Ordering::SeqCst);
            self.active = false;
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if self.active {
            self.worker
                .read()
                .active_requests
                .fetch_sub(1, Ordering::SeqCst);
            self.active = false;
        }
    }
}

/// A stream wrapper that binds an active in-flight guard to the lifetime of the response stream.
/// When the stream terminates, errors, or is cancelled by client disconnect, the guard drops.
pub struct TrackedStream<S> {
    inner: S,
    _guard: InFlightGuard,
}

impl<S> TrackedStream<S> {
    pub fn new(inner: S, guard: InFlightGuard) -> Self {
        Self {
            inner,
            _guard: guard,
        }
    }
}

impl<S, E> Stream for TrackedStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Constructs a compliant OpenAI SSE event stream (`text/event-stream`).
pub fn create_openai_sse_stream(
    request_id: &str,
    model: &str,
    decision: &RoutingDecision,
    block_size: usize,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + 'static {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let total_prompt_tokens = decision.total_blocks * block_size;
    let cached_tokens = decision.matched_blocks * block_size;
    let worker_id = decision.worker_id.clone();
    let req_id = request_id.to_string();
    let model_name = model.to_string();

    let mut chunks: Vec<Bytes> = Vec::new();

    // 1. Initial role declaration chunk
    let init_chunk = json!({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": null
        }]
    });
    chunks.push(Bytes::from(format!("data: {}\n\n", init_chunk)));

    // 2. Token text deltas
    let tokens = [
        "Response ",
        "streamed ",
        "via ",
        "RadixGate ",
        "[routed ",
        "to ",
        &worker_id,
        "].",
    ];

    for tok in tokens {
        let delta_chunk = json!({
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": tok
                },
                "finish_reason": null
            }]
        });
        chunks.push(Bytes::from(format!("data: {}\n\n", delta_chunk)));
    }

    // 3. Final chunk with stop finish_reason and usage metadata
    let completion_tokens = tokens.len();
    let total_tokens = total_prompt_tokens + completion_tokens;
    let finish_chunk = json!({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens
            }
        }
    });
    chunks.push(Bytes::from(format!("data: {}\n\n", finish_chunk)));

    // 4. Stream terminator
    chunks.push(Bytes::from("data: [DONE]\n\n"));

    futures_util::stream::iter(chunks.into_iter().map(Ok))
}

/// Constructs a compliant Anthropic SSE event stream (`text/event-stream`).
pub fn create_anthropic_sse_stream(
    request_id: &str,
    model: &str,
    decision: &RoutingDecision,
    block_size: usize,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + 'static {
    let total_prompt_tokens = decision.total_blocks * block_size;
    let cached_tokens = decision.matched_blocks * block_size;
    let uncached_tokens = total_prompt_tokens.saturating_sub(cached_tokens);
    let worker_id = decision.worker_id.clone();
    let req_id = request_id.to_string();
    let model_name = model.to_string();

    let mut chunks: Vec<Bytes> = Vec::new();

    // 1. message_start
    let msg_start = json!({
        "type": "message_start",
        "message": {
            "id": req_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "usage": {
                "input_tokens": total_prompt_tokens,
                "output_tokens": 1,
                "cache_creation_input_tokens": uncached_tokens,
                "cache_read_input_tokens": cached_tokens
            }
        }
    });
    chunks.push(Bytes::from(format!(
        "event: message_start\ndata: {}\n\n",
        msg_start
    )));

    // 2. content_block_start
    let block_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": ""
        }
    });
    chunks.push(Bytes::from(format!(
        "event: content_block_start\ndata: {}\n\n",
        block_start
    )));

    // 3. content_block_delta chunks
    let tokens = [
        "Response ",
        "streamed ",
        "via ",
        "RadixGate ",
        "[routed ",
        "to ",
        &worker_id,
        "].",
    ];

    for tok in tokens {
        let delta = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": tok
            }
        });
        chunks.push(Bytes::from(format!(
            "event: content_block_delta\ndata: {}\n\n",
            delta
        )));
    }

    // 4. content_block_stop
    let block_stop = json!({
        "type": "content_block_stop",
        "index": 0
    });
    chunks.push(Bytes::from(format!(
        "event: content_block_stop\ndata: {}\n\n",
        block_stop
    )));

    // 5. message_delta
    let msg_delta = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": "end_turn",
            "stop_sequence": null
        },
        "usage": {
            "output_tokens": tokens.len()
        }
    });
    chunks.push(Bytes::from(format!(
        "event: message_delta\ndata: {}\n\n",
        msg_delta
    )));

    // 6. message_stop
    let msg_stop = json!({
        "type": "message_stop"
    });
    chunks.push(Bytes::from(format!(
        "event: message_stop\ndata: {}\n\n",
        msg_stop
    )));

    futures_util::stream::iter(chunks.into_iter().map(Ok))
}
