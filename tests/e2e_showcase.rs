use std::sync::Arc;
use std::time::Duration;

use axum::extract::Json;
use parking_lot::RwLock;
use serde_json::json;

use radix_gate::proxy::{AppState, create_router};
use radix_gate::shadow_cache::{WorkerPool, WorkerReplica, WorkerRole};
use radix_gate::tokenizer::BlockChunker;
use radix_gate::trie::RadixTree;
use radix_gate::{GatewayMesh, LoadTestConfig, LoadTestEndpoint, LoadTestRunner, MeshConfig};

/// End-to-end integration test validating the complete distributed cluster lifecycle:
/// 1. Dual-gateway UDP gossip mesh peering.
/// 2. Disaggregated Prefill vs Decode worker routing.
/// 3. Cross-gateway cache coherence without client re-routing.
#[tokio::test]
async fn test_e2e_dual_gateway_mesh_and_disaggregation() {
    // 1. Launch a mock Prefill worker (high compute, port A) and Decode worker (high memory, port B)
    let prefill_app = axum::Router::new().route(
        "/v1/messages",
        axum::routing::post(|| async {
            Json(json!({
                "id": "msg_prefill",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "prefill computation complete"}],
                "model": "claude-3-5-sonnet",
                "usage": {"input_tokens": 100, "output_tokens": 10}
            }))
        }),
    );
    let prefill_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let prefill_port = prefill_listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(prefill_listener, prefill_app).await.unwrap();
    });

    let decode_app = axum::Router::new().route(
        "/v1/messages",
        axum::routing::post(|| async {
            Json(json!({
                "id": "msg_decode",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "decode token generation complete"}],
                "model": "claude-3-5-sonnet",
                "usage": {"input_tokens": 10, "output_tokens": 40}
            }))
        }),
    );
    let decode_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let decode_port = decode_listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(decode_listener, decode_app).await.unwrap();
    });

    // 2. Setup Worker Pool with disaggregated roles
    let worker_pool_1 = Arc::new(WorkerPool::new());
    worker_pool_1.add_worker(WorkerReplica::new_with_role(
        "worker-prefill".into(),
        format!("http://127.0.0.1:{}", prefill_port),
        1000,
        16,
        WorkerRole::Prefill,
    ));
    worker_pool_1.add_worker(WorkerReplica::new_with_role(
        "worker-decode".into(),
        format!("http://127.0.0.1:{}", decode_port),
        1000,
        16,
        WorkerRole::Decode,
    ));

    let worker_pool_2 = Arc::new(WorkerPool::new());
    worker_pool_2.add_worker(WorkerReplica::new_with_role(
        "worker-prefill".into(),
        format!("http://127.0.0.1:{}", prefill_port),
        1000,
        16,
        WorkerRole::Prefill,
    ));
    worker_pool_2.add_worker(WorkerReplica::new_with_role(
        "worker-decode".into(),
        format!("http://127.0.0.1:{}", decode_port),
        1000,
        16,
        WorkerRole::Decode,
    ));

    // 3. Initialize Gateway 2 with UDP Mesh listener on ephemeral socket
    let tree_2 = Arc::new(RwLock::new(RadixTree::new(64)));
    let config_mesh_2 = MeshConfig {
        node_id: "gateway-02".to_string(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        peer_addrs: Vec::new(),
    };
    let mesh_2 = Arc::new(
        GatewayMesh::bind(config_mesh_2, tree_2.clone())
            .await
            .unwrap(),
    );
    let mesh_addr_2 = mesh_2.local_addr().unwrap();
    let _mesh_listener_2 = mesh_2.clone().start_listener();

    let state_2 = Arc::new(
        AppState::new(tree_2.clone(), worker_pool_2, BlockChunker::new(64)).with_mesh(mesh_2),
    );
    let http_listener_2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_port_2 = http_listener_2.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(http_listener_2, create_router(state_2))
            .await
            .unwrap();
    });

    // 4. Initialize Gateway 1 peered with Gateway 2's UDP socket
    let tree_1 = Arc::new(RwLock::new(RadixTree::new(64)));
    let config_mesh_1 = MeshConfig {
        node_id: "gateway-01".to_string(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        peer_addrs: vec![mesh_addr_2],
    };
    let mesh_1 = Arc::new(
        GatewayMesh::bind(config_mesh_1, tree_1.clone())
            .await
            .unwrap(),
    );
    let _mesh_listener_1 = mesh_1.clone().start_listener();

    let state_1 = Arc::new(
        AppState::new(tree_1.clone(), worker_pool_1, BlockChunker::new(64)).with_mesh(mesh_1),
    );
    let http_listener_1 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_port_1 = http_listener_1.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(http_listener_1, create_router(state_1))
            .await
            .unwrap();
    });

    let client = reqwest::Client::new();

    // 5. Send cold prompt (0% hit, > 2 blocks) to Gateway 1 -> Router should select Prefill worker!
    let prompt = "A massive software architecture document requiring heavy matrix multiplication prefill and large context window processing across distributed vLLM instances. ".repeat(16);
    let req_1 = json!({
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": prompt}]
    });

    let resp_1 = client
        .post(format!("http://127.0.0.1:{}/v1/messages", http_port_1))
        .json(&req_1)
        .send()
        .await
        .unwrap();

    assert_eq!(resp_1.status(), axum::http::StatusCode::OK);
    let worker_selected = resp_1
        .headers()
        .get("x-radix-worker")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(worker_selected, "worker-prefill");

    // Allow UDP gossip packet transit between Gateway 1 and Gateway 2
    tokio::time::sleep(Duration::from_millis(60)).await;

    // 6. Verify Gateway 2 has synchronized the prefix via gossip WITHOUT ever seeing the client request!
    let chunker = BlockChunker::new(64);
    let full_prompt = format!("<|user|>\n{}\n<|end|>\n", prompt);
    let blocks = chunker.chunk_text(&full_prompt);
    assert!(blocks.len() >= 3, "Expected at least 3 blocks for prompt");

    let match_on_gateway_2 = tree_2.read().find_longest_prefix(&blocks);
    assert_eq!(match_on_gateway_2.max_matched_blocks, blocks.len());
    assert_eq!(
        match_on_gateway_2.worker_matches.get("worker-prefill"),
        Some(&blocks.len())
    );

    // 7. Verify routing on Gateway 2 leverages the gossiped cache state
    let resp_2 = client
        .post(format!("http://127.0.0.1:{}/v1/messages", http_port_2))
        .json(&req_1)
        .send()
        .await
        .unwrap();
    assert_eq!(resp_2.status(), axum::http::StatusCode::OK);
    let worker_on_gw2 = resp_2
        .headers()
        .get("x-radix-worker")
        .unwrap()
        .to_str()
        .unwrap();
    // Since worker-prefill holds the warm KV cache (100% affinity), Gateway 2 routes to it!
    assert_eq!(worker_on_gw2, "worker-prefill");
}

/// End-to-end integration test validating high-concurrency socket load generation,
/// percentile tracking, and Prometheus metrics export.
#[tokio::test]
async fn test_e2e_load_test_and_openmetrics_verification() {
    let tree = Arc::new(RwLock::new(RadixTree::new(64)));
    let pool = Arc::new(WorkerPool::new());
    pool.add_worker(WorkerReplica::new(
        "worker-01".into(),
        "http://127.0.0.1:8001".into(),
        1000,
        16,
    ));
    pool.add_worker(WorkerReplica::new(
        "worker-02".into(),
        "http://127.0.0.1:8002".into(),
        1000,
        16,
    ));

    let state = Arc::new(AppState::new(tree, pool, BlockChunker::new(64)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, create_router(state)).await.unwrap();
    });

    let config = LoadTestConfig {
        target_url: format!("http://127.0.0.1:{}", port),
        concurrency: 5,
        total_requests: 30,
        endpoint: LoadTestEndpoint::Messages,
        num_shared_prefixes: 2,
        turns_per_session: 3,
        request_timeout: Duration::from_secs(5),
    };

    let runner = LoadTestRunner::new(config);
    let summary = runner.run().await.unwrap();

    assert_eq!(summary.total_requests, 30);
    assert_eq!(summary.success_count, 30);
    assert_eq!(summary.error_count, 0);
    assert!(summary.requests_per_sec > 0.0);
    assert!(
        summary.matched_blocks > 0,
        "Multi-turn prompts must hit KV-cache!"
    );
    assert!(summary.p50_ms > 0.0);
    assert!(summary.p99_ms >= summary.p50_ms);

    // Verify OpenMetrics Prometheus scrape
    let client = reqwest::Client::new();
    let prom_resp = client
        .get(format!("http://127.0.0.1:{}/metrics", port))
        .send()
        .await
        .unwrap();

    assert_eq!(prom_resp.status(), axum::http::StatusCode::OK);
    let text = prom_resp.text().await.unwrap();
    assert!(text.contains("# TYPE radixgate_requests_total counter"));
    assert!(text.contains("radixgate_requests_total 30"));
    assert!(text.contains("# TYPE radixgate_cache_hit_ratio gauge"));
    assert!(text.contains("radixgate_worker_load{worker=\"worker-01\"}"));
}
