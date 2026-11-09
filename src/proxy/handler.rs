use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn};

use crate::balancer::RoutingPolicy;
use crate::balancer::policies::PrefixAwarePolicy;
use crate::proxy::types::ChatRequest;
use crate::shadow_cache::WorkerPool;
use crate::tokenizer::BlockChunker;
use crate::trie::{BlockHash, RadixTree};

/// Shared application state across the async Axum web server.
pub struct AppState {
    pub radix_tree: Arc<RwLock<RadixTree>>,
    pub worker_pool: Arc<WorkerPool>,
    pub chunker: BlockChunker,
    pub policy: Arc<dyn RoutingPolicy>,
    pub client: reqwest::Client,

    // Performance metrics
    pub total_requests: AtomicU64,
    pub total_matched_blocks: AtomicU64,
    pub total_prompt_blocks: AtomicU64,
}

impl AppState {
    pub fn new(
        radix_tree: Arc<RwLock<RadixTree>>,
        worker_pool: Arc<WorkerPool>,
        chunker: BlockChunker,
    ) -> Self {
        Self {
            radix_tree,
            worker_pool,
            chunker,
            policy: Arc::new(PrefixAwarePolicy::default()),
            client: reqwest::Client::builder()
                .pool_max_idle_per_host(32)
                .tcp_nodelay(true)
                .build()
                .unwrap(),
            total_requests: AtomicU64::new(0),
            total_matched_blocks: AtomicU64::new(0),
            total_prompt_blocks: AtomicU64::new(0),
        }
    }
}

/// Health check endpoint.
pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "service": "radix-gate" }))
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CacheSyncPayload {
    pub worker_id: String,
    pub evicted_blocks: Vec<BlockHash>,
}

/// Active worker feedback endpoint: allows upstream engines (vLLM / Triton)
/// to broadcast dynamic VRAM cache evictions directly to the gateway.
pub async fn handle_cache_sync(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CacheSyncPayload>,
) -> impl IntoResponse {
    info!(
        worker = %payload.worker_id,
        evicted = payload.evicted_blocks.len(),
        "Received upstream cache eviction sync"
    );

    let mut tree = state.radix_tree.write();
    tree.evict_blocks(&payload.worker_id, &payload.evicted_blocks);

    Json(json!({
        "status": "synchronized",
        "worker_id": payload.worker_id,
        "evicted_blocks_purged": payload.evicted_blocks.len(),
    }))
}

/// Global cluster and Radix Tree metrics endpoint.
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (nodes, active_workers) = {
        let tree = state.radix_tree.read();
        tree.stats()
    };

    let total_reqs = state.total_requests.load(Ordering::Relaxed);
    let matched_b = state.total_matched_blocks.load(Ordering::Relaxed);
    let total_b = state.total_prompt_blocks.load(Ordering::Relaxed);

    let hit_ratio = if total_b > 0 {
        matched_b as f64 / total_b as f64
    } else {
        0.0
    };

    let worker_ids = state.worker_pool.get_worker_ids();
    let mut workers_summary = Vec::new();
    for id in &worker_ids {
        if let Some(w_lock) = state.worker_pool.get_worker(id) {
            let w = w_lock.read();
            workers_summary.push(json!({
                "id": w.id,
                "endpoint": w.endpoint_url,
                "load": w.load(),
                "cache_utilization": format!("{:.1}%", w.cache_utilization() * 100.0),
                "total_routed": w.total_requests_routed,
                "matched_blocks": w.total_matched_blocks,
                "prefilled_blocks": w.total_prefilled_blocks,
            }));
        }
    }

    Json(json!({
        "status": "healthy",
        "radix_tree": {
            "total_nodes": nodes,
            "unique_workers": active_workers,
        },
        "performance": {
            "total_requests": total_reqs,
            "total_matched_blocks": matched_b,
            "total_prompt_blocks": total_b,
            "cluster_cache_hit_ratio": format!("{:.2}%", hit_ratio * 100.0),
        },
        "workers": workers_summary
    }))
}

/// Primary OpenAI-compatible chat routing handler (`/v1/chat/completions`).
pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<ChatRequest>,
) -> Response {
    handle_routing(state, headers, payload, false).await
}

/// Anthropic Messages API routing handler (`/v1/messages`).
pub async fn handle_anthropic_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<ChatRequest>,
) -> Response {
    handle_routing(state, headers, payload, true).await
}

async fn handle_routing(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: ChatRequest,
    is_anthropic: bool,
) -> Response {
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // 1. Extract prompt and compute token block hashes
    let prompt_text = payload.extract_prompt_text();
    let block_hashes = state.chunker.chunk_text(&prompt_text);
    let total_blocks = block_hashes.len();

    // 2. Select optimal worker via Radix-Aware balancer
    let decision = {
        let tree = state.radix_tree.read();
        state
            .policy
            .select_worker(&state.worker_pool, &tree, &block_hashes)
    };

    let decision = match decision {
        Some(d) => d,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": "No healthy inference workers available in cluster" })),
            )
                .into_response();
        }
    };

    info!(
        worker = %decision.worker_id,
        matched = decision.matched_blocks,
        total = decision.total_blocks,
        score = %decision.affinity_score,
        "Routed request"
    );

    // Record metrics
    state
        .total_matched_blocks
        .fetch_add(decision.matched_blocks as u64, Ordering::Relaxed);
    state
        .total_prompt_blocks
        .fetch_add(total_blocks as u64, Ordering::Relaxed);

    // 3. Increment active load on selected worker and acquire optimistic in-flight prefill lease
    let worker_arc = match state.worker_pool.get_worker(&decision.worker_id) {
        Some(w) => w,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "Worker disappeared during routing" })),
            )
                .into_response();
        }
    };

    {
        let worker = worker_arc.read();
        worker
            .active_requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    // Optimistically reserve prefix in trie to prevent concurrent thundering-herd cold prefills
    {
        let mut tree = state.radix_tree.write();
        tree.reserve_in_flight(&block_hashes, &decision.worker_id);
    }

    // 4. Update shadow cache and Radix Tree with newly cached blocks
    {
        let mut worker = worker_arc.write();
        let evicted = worker.record_request(&block_hashes, decision.matched_blocks);

        let mut tree = state.radix_tree.write();
        tree.insert(&block_hashes, &decision.worker_id);

        if !evicted.is_empty() {
            tree.prune_empty_nodes();
        }
    }

    // 5. Attempt live upstream forwarding if endpoint is a real upstream server
    let endpoint_url = decision.endpoint_url.clone();
    let is_real_upstream =
        endpoint_url.starts_with("http://") || endpoint_url.starts_with("https://");
    let is_mock_port = endpoint_url.contains(":800"); // default mock ports 8001-8004

    let live_response = if is_real_upstream && !is_mock_port {
        try_forward_upstream(
            &state.client,
            &endpoint_url,
            &headers,
            &payload,
            is_anthropic,
        )
        .await
    } else {
        None
    };

    // Decrement worker active requests
    {
        let worker = worker_arc.read();
        worker
            .active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    let hit_ratio = if total_blocks > 0 {
        decision.matched_blocks as f64 / total_blocks as f64
    } else {
        0.0
    };

    // 6. Return live response if upstream succeeded, else return high-fidelity mock
    let mut response = match live_response {
        Some(resp) => {
            // Check for upstream dynamic eviction header feedback
            if let Some(evicted_str) = resp
                .headers()
                .get("x-evicted-blocks")
                .and_then(|h| h.to_str().ok())
            {
                let blocks: Vec<BlockHash> = evicted_str
                    .split(',')
                    .filter_map(|s| s.trim().parse::<u64>().ok())
                    .collect();
                if !blocks.is_empty() {
                    let mut tree = state.radix_tree.write();
                    tree.evict_blocks(&decision.worker_id, &blocks);
                }
            }
            resp
        }
        None => build_mock_response(&state, &payload, &decision, is_anthropic),
    };

    // Release in-flight reservation lease and transition prefix to confirmed state
    {
        let mut tree = state.radix_tree.write();
        tree.release_in_flight(&block_hashes, &decision.worker_id, true);
    }

    // Inject Radix Gateway telemetry headers
    let headers_mut = response.headers_mut();
    headers_mut.insert(
        "X-Radix-Worker",
        decision
            .worker_id
            .parse()
            .unwrap_or_else(|_| "unknown".parse().unwrap()),
    );
    headers_mut.insert(
        "X-Radix-Matched-Blocks",
        decision.matched_blocks.to_string().parse().unwrap(),
    );
    headers_mut.insert(
        "X-Radix-Total-Blocks",
        decision.total_blocks.to_string().parse().unwrap(),
    );
    headers_mut.insert(
        "X-Radix-Hit-Ratio",
        format!("{:.2}", hit_ratio).parse().unwrap(),
    );

    response
}

/// Attempts to proxy the request directly to a live upstream vLLM/Anthropic worker.
async fn try_forward_upstream(
    client: &reqwest::Client,
    endpoint_base: &str,
    client_headers: &HeaderMap,
    payload: &ChatRequest,
    is_anthropic: bool,
) -> Option<Response> {
    let path = if is_anthropic {
        "/v1/messages"
    } else {
        "/v1/chat/completions"
    };
    let target_url = format!("{}{}", endpoint_base.trim_end_matches('/'), path);

    let mut req_builder = client.post(&target_url).json(payload);

    if let Some(auth) = client_headers.get("authorization") {
        req_builder = req_builder.header("authorization", auth);
    }
    if let Some(api_key) = client_headers.get("x-api-key") {
        req_builder = req_builder.header("x-api-key", api_key);
    }
    if let Some(anthropic_ver) = client_headers.get("anthropic-version") {
        req_builder = req_builder.header("anthropic-version", anthropic_ver);
    }

    match req_builder.send().await {
        Ok(upstream_resp) => {
            let status =
                StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::OK);
            let content_type = upstream_resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();

            let body = Body::from_stream(upstream_resp.bytes_stream());
            let response = Response::builder()
                .status(status)
                .header("content-type", content_type)
                .body(body)
                .unwrap_or_else(|_| {
                    (StatusCode::INTERNAL_SERVER_ERROR, "Stream building error").into_response()
                });

            Some(response)
        }
        Err(err) => {
            warn!(url = %target_url, error = %err, "Upstream connection failed; falling back to simulated response");
            None
        }
    }
}

/// Generates a realistic mock response matching OpenAI / Anthropic specification.
fn build_mock_response(
    state: &AppState,
    payload: &ChatRequest,
    decision: &crate::balancer::RoutingDecision,
    is_anthropic: bool,
) -> Response {
    let total_blocks = decision.total_blocks;
    let block_size = state.chunker.block_size;

    let response_body = if is_anthropic {
        json!({
            "id": format!("msg_{}", uuid_simple()),
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": format!("Response generated via RadixGate [routed to {}].", decision.worker_id)
            }],
            "model": payload.model,
            "usage": {
                "input_tokens": total_blocks * block_size,
                "output_tokens": 42,
                "cache_creation_input_tokens": (total_blocks.saturating_sub(decision.matched_blocks)) * block_size,
                "cache_read_input_tokens": decision.matched_blocks * block_size
            }
        })
    } else {
        json!({
            "id": format!("chatcmpl_{}", uuid_simple()),
            "object": "chat.completion",
            "model": payload.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": format!("Response generated via RadixGate [routed to {}].", decision.worker_id)
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": total_blocks * block_size,
                "completion_tokens": 42,
                "total_tokens": (total_blocks * block_size) + 42,
                "prompt_tokens_details": {
                    "cached_tokens": decision.matched_blocks * block_size
                }
            }
        })
    };

    Json(response_body).into_response()
}

fn uuid_simple() -> String {
    format!("{:08x}{:08x}", rand::random::<u32>(), rand::random::<u32>())
}
