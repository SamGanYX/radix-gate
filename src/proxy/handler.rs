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

/// Renders the active Radix Tree branches as plain text ASCII.
pub async fn handle_tree(State(state): State<Arc<AppState>>) -> Response {
    let tree = state.radix_tree.read();
    let ascii = tree.render_ascii();
    Response::builder()
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from(ascii))
        .unwrap_or_else(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, "Tree render error").into_response()
        })
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
/// Returns standard Prometheus text format by default (or JSON if Accept: application/json).
pub async fn metrics(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
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
    let accept = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if accept.contains("application/json") {
        let mut workers_summary = Vec::new();
        for id in &worker_ids {
            if let Some(w_lock) = state.worker_pool.get_worker(id) {
                let w = w_lock.read();
                workers_summary.push(json!({
                    "id": w.id,
                    "endpoint": w.endpoint_url,
                    "load": w.load(),
                    "role": format!("{:?}", w.role),
                    "role_badge": w.role.badge(),
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
        .into_response()
    } else {
        // Prometheus / OpenMetrics text format
        let mut prom = String::new();
        prom.push_str(
            "# HELP radixgate_requests_total Total number of routed inference requests\n",
        );
        prom.push_str("# TYPE radixgate_requests_total counter\n");
        prom.push_str(&format!("radixgate_requests_total {}\n\n", total_reqs));

        prom.push_str(
            "# HELP radixgate_cache_hit_ratio Global cluster KV-cache prefix hit ratio\n",
        );
        prom.push_str("# TYPE radixgate_cache_hit_ratio gauge\n");
        prom.push_str(&format!("radixgate_cache_hit_ratio {:.4}\n\n", hit_ratio));

        prom.push_str(
            "# HELP radixgate_matched_blocks_total Total token blocks matched across cache\n",
        );
        prom.push_str("# TYPE radixgate_matched_blocks_total counter\n");
        prom.push_str(&format!("radixgate_matched_blocks_total {}\n\n", matched_b));

        prom.push_str("# HELP radixgate_prompt_blocks_total Total prompt token blocks received\n");
        prom.push_str("# TYPE radixgate_prompt_blocks_total counter\n");
        prom.push_str(&format!("radixgate_prompt_blocks_total {}\n\n", total_b));

        prom.push_str("# HELP radixgate_radix_tree_nodes Current number of nodes in Radix Tree\n");
        prom.push_str("# TYPE radixgate_radix_tree_nodes gauge\n");
        prom.push_str(&format!("radixgate_radix_tree_nodes {}\n\n", nodes));

        prom.push_str("# HELP radixgate_worker_load Active in-flight requests per worker\n");
        prom.push_str("# TYPE radixgate_worker_load gauge\n");
        for id in &worker_ids {
            if let Some(w_lock) = state.worker_pool.get_worker(id) {
                let w = w_lock.read();
                prom.push_str(&format!(
                    "radixgate_worker_load{{worker=\"{}\"}} {}\n",
                    w.id,
                    w.load()
                ));
            }
        }
        prom.push('\n');

        prom.push_str("# HELP radixgate_worker_kv_utilization Ratio of GPU KV blocks utilized\n");
        prom.push_str("# TYPE radixgate_worker_kv_utilization gauge\n");
        for id in &worker_ids {
            if let Some(w_lock) = state.worker_pool.get_worker(id) {
                let w = w_lock.read();
                prom.push_str(&format!(
                    "radixgate_worker_kv_utilization{{worker=\"{}\"}} {:.4}\n",
                    w.id,
                    w.cache_utilization()
                ));
            }
        }
        prom.push('\n');

        prom.push_str("# HELP radixgate_worker_health Health status of worker (1 = healthy, 0 = circuit-broken)\n");
        prom.push_str("# TYPE radixgate_worker_health gauge\n");
        for id in &worker_ids {
            if let Some(w_lock) = state.worker_pool.get_worker(id) {
                let w = w_lock.read();
                let health_val = if w.is_healthy { 1 } else { 0 };
                prom.push_str(&format!(
                    "radixgate_worker_health{{worker=\"{}\"}} {}\n",
                    w.id, health_val
                ));
            }
        }

        Response::builder()
            .header("content-type", "text/plain; version=0.0.4; charset=utf-8")
            .body(Body::from(prom))
            .unwrap_or_else(|_| {
                (StatusCode::INTERNAL_SERVER_ERROR, "Prometheus format error").into_response()
            })
    }
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

    // Proactive Hotspot Replication: pre-warm secondary replica if viral prefix congestion is detected
    if let Some(ref target_id) = decision.replicate_to_worker
        && let Some(target_arc) = state.worker_pool.get_worker(target_id)
    {
        let mut target_worker = target_arc.write();
        let _ = target_worker.record_request(&block_hashes, 0);
        let mut tree = state.radix_tree.write();
        tree.insert(&block_hashes, target_id);
        info!(
            hotspot_from = %decision.worker_id,
            replicated_to = %target_id,
            "Proactively replicated viral prefix hotspot to secondary replica"
        );
    }

    // Record metrics
    state
        .total_matched_blocks
        .fetch_add(decision.matched_blocks as u64, Ordering::Relaxed);
    state
        .total_prompt_blocks
        .fetch_add(total_blocks as u64, Ordering::Relaxed);

    // 3. Acquire optimistic in-flight prefill lease on selected worker
    let mut current_worker_id = decision.worker_id.clone();
    let mut current_worker_arc = match state.worker_pool.get_worker(&current_worker_id) {
        Some(w) => w,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "Worker disappeared during routing" })),
            )
                .into_response();
        }
    };

    current_worker_arc
        .read()
        .active_requests
        .fetch_add(1, Ordering::SeqCst);

    // Optimistically reserve prefix in trie to prevent concurrent thundering-herd cold prefills
    {
        let mut tree = state.radix_tree.write();
        tree.reserve_in_flight(&block_hashes, &current_worker_id);
    }

    let endpoint_url = decision.endpoint_url.clone();
    let is_real_upstream =
        endpoint_url.starts_with("http://") || endpoint_url.starts_with("https://");
    let is_mock_port = endpoint_url.contains(":800"); // default mock ports 8001-8004

    let mut final_response: Option<Response> = None;

    if is_real_upstream && !is_mock_port {
        // Attempt primary worker
        match try_forward_upstream(
            &state.client,
            &endpoint_url,
            &headers,
            &payload,
            is_anthropic,
        )
        .await
        {
            Ok(resp) => {
                current_worker_arc.write().record_success();
                final_response = Some(resp);
            }
            Err(err) => {
                warn!(
                    worker = %current_worker_id,
                    error = %err,
                    "Primary upstream worker failed; checking circuit breaker and initiating failover"
                );
                let tripped = current_worker_arc.write().record_failure();
                {
                    let mut tree = state.radix_tree.write();
                    tree.release_in_flight(&block_hashes, &current_worker_id, false);
                    if tripped {
                        warn!(
                            worker = %current_worker_id,
                            "Circuit breaker TRIPPED! Evicting worker from Radix Tree"
                        );
                        tree.evict_worker(&current_worker_id);
                    }
                }
                current_worker_arc
                    .read()
                    .active_requests
                    .fetch_sub(1, Ordering::SeqCst);

                // Failover Retry Loop: Attempt other healthy workers in cluster
                let candidates = state.worker_pool.get_healthy_worker_ids();
                for candidate_id in candidates {
                    if candidate_id == current_worker_id {
                        continue;
                    }
                    if let Some(candidate_arc) = state.worker_pool.get_worker(&candidate_id) {
                        let candidate_url = candidate_arc.read().endpoint_url.clone();
                        candidate_arc
                            .read()
                            .active_requests
                            .fetch_add(1, Ordering::SeqCst);
                        state
                            .radix_tree
                            .write()
                            .reserve_in_flight(&block_hashes, &candidate_id);

                        info!(from = %current_worker_id, to = %candidate_id, "Retrying request on failover worker");

                        match try_forward_upstream(
                            &state.client,
                            &candidate_url,
                            &headers,
                            &payload,
                            is_anthropic,
                        )
                        .await
                        {
                            Ok(resp) => {
                                candidate_arc.write().record_success();
                                current_worker_id = candidate_id.clone();
                                current_worker_arc = candidate_arc;
                                final_response = Some(resp);
                                break;
                            }
                            Err(retry_err) => {
                                warn!(candidate = %candidate_id, error = %retry_err, "Failover candidate failed");
                                let tripped = candidate_arc.write().record_failure();
                                {
                                    let mut tree = state.radix_tree.write();
                                    tree.release_in_flight(&block_hashes, &candidate_id, false);
                                    if tripped {
                                        tree.evict_worker(&candidate_id);
                                    }
                                }
                                candidate_arc
                                    .read()
                                    .active_requests
                                    .fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Mock worker mode (for local benchmark verification and unit tests)
        final_response = Some(build_mock_response(
            &state,
            &payload,
            &decision,
            is_anthropic,
        ));
    }

    // If final_response is still None, all real upstreams failed -> Return HTTP 502 Bad Gateway
    let mut response = match final_response {
        Some(resp) => resp,
        None => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": {
                        "message": "Upstream inference cluster unavailable: all candidate replicas failed or timed out",
                        "type": "bad_gateway",
                        "code": 502
                    }
                })),
            )
                .into_response();
        }
    };

    // Decrement worker active requests
    current_worker_arc
        .read()
        .active_requests
        .fetch_sub(1, Ordering::SeqCst);

    let hit_ratio = if total_blocks > 0 {
        decision.matched_blocks as f64 / total_blocks as f64
    } else {
        0.0
    };

    // 4. Update shadow cache & atomically commit in-flight lease
    {
        let mut worker = current_worker_arc.write();
        let evicted = worker.record_request(&block_hashes, decision.matched_blocks);

        let mut tree = state.radix_tree.write();
        tree.commit_in_flight(&block_hashes, &current_worker_id);

        if !evicted.is_empty() {
            tree.prune_empty_nodes();
        }

        // Check for upstream dynamic eviction header feedback
        if let Some(evicted_str) = response
            .headers()
            .get("x-evicted-blocks")
            .and_then(|h| h.to_str().ok())
        {
            let blocks: Vec<BlockHash> = evicted_str
                .split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect();
            if !blocks.is_empty() {
                tree.evict_blocks(&current_worker_id, &blocks);
            }
        }
    }

    // Inject Radix Gateway telemetry headers
    let headers_mut = response.headers_mut();
    headers_mut.insert(
        "X-Radix-Worker",
        current_worker_id
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
) -> Result<Response, String> {
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

            if status.is_server_error() {
                return Err(format!("Upstream returned HTTP server error {}", status));
            }

            let content_type = upstream_resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();

            let mut response_builder = Response::builder()
                .status(status)
                .header("content-type", content_type);

            if let Some(evicted) = upstream_resp.headers().get("x-evicted-blocks") {
                response_builder = response_builder.header("x-evicted-blocks", evicted);
            }

            let body = Body::from_stream(upstream_resp.bytes_stream());
            response_builder
                .body(body)
                .map_err(|e| format!("Stream building error: {}", e))
        }
        Err(err) => Err(format!("Upstream connection failed: {}", err)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::create_router;
    use crate::shadow_cache::{WorkerPool, WorkerReplica};
    use crate::tokenizer::BlockChunker;
    use crate::trie::RadixTree;
    use parking_lot::RwLock;

    #[tokio::test]
    async fn test_upstream_failover_and_502_bad_gateway() {
        // Setup healthy secondary worker server returning 200 OK
        let healthy_app = axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(|| async {
                Json(json!({
                    "id": "msg_healthy",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "healthy response"}],
                    "model": "claude-3-5-sonnet",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let healthy_port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, healthy_app).await.unwrap();
        });

        // Worker 1: dead port (19999)
        // Worker 2: healthy port
        let radix_tree = Arc::new(RwLock::new(RadixTree::new(64)));
        let worker_pool = Arc::new(WorkerPool::new());
        worker_pool.add_worker(WorkerReplica::new(
            "worker-dead".into(),
            "http://127.0.0.1:19999".into(),
            1000,
            16,
        ));
        worker_pool.add_worker(WorkerReplica::new(
            "worker-healthy".into(),
            format!("http://127.0.0.1:{}", healthy_port),
            1000,
            16,
        ));

        // Warm worker-dead with the prefix so it is initially selected by the radix router
        let chunker = BlockChunker::new(64);
        let prompt = "Initial shared prompt for testing failover.";
        let blocks = chunker.chunk_text(prompt);
        radix_tree.write().insert(&blocks, "worker-dead");

        let state = Arc::new(AppState::new(
            radix_tree.clone(),
            worker_pool.clone(),
            chunker,
        ));
        let router = create_router(state);

        let test_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let test_port = test_listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(test_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let req_body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": prompt}]
        });

        // 1. Send request: worker-dead fails, router automatically fails over to worker-healthy!
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/messages", test_port))
            .json(&req_body)
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let routed_worker = resp
            .headers()
            .get("x-radix-worker")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(routed_worker, "worker-healthy");

        // 2. Test 502 Bad Gateway when all upstreams are dead
        let dead_pool = Arc::new(WorkerPool::new());
        dead_pool.add_worker(WorkerReplica::new(
            "dead-only".into(),
            "http://127.0.0.1:19998".into(),
            1000,
            16,
        ));
        let dead_state = Arc::new(AppState::new(
            Arc::new(RwLock::new(RadixTree::new(64))),
            dead_pool,
            BlockChunker::new(64),
        ));
        let dead_router = create_router(dead_state);
        let dead_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_port = dead_listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(dead_listener, dead_router).await.unwrap();
        });

        let dead_resp = client
            .post(format!("http://127.0.0.1:{}/v1/messages", dead_port))
            .json(&req_body)
            .send()
            .await
            .unwrap();

        assert_eq!(dead_resp.status(), StatusCode::BAD_GATEWAY);
    }
}
