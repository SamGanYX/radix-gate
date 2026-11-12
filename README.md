# RadixGate

Deterministic, prefix-aware reverse proxy and radix router for distributed LLM inference clusters.

[![CI](https://github.com/SamGanYX/radix-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/SamGanYX/radix-gate/actions)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-informational.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

---

## Overview

Modern inference engines (vLLM, SGLang, and Claude) implement prompt KV-cache reuse (PagedAttention, RadixAttention) to bypass prefill matrix multiplications for shared prefixes. In multi-replica deployments, default layer-7 load balancers (round-robin, random, or least-connections) route traffic without prefix awareness. When multi-turn agent turns or queries sharing a common system prompt are scattered across arbitrary workers, cluster cache hit rates deteriorate and GPU VRAM suffers cache thrashing.

RadixGate operates as an L7 reverse proxy that inspects incoming prompt token streams, tracks worker KV-cache residency via a compressed radix trie, and routes requests to maximize cache reuse while balancing in-flight queue depth.

```
                    Client Request
                          │
                          ▼
             ┌───────────────────────────┐
             │         RadixGate         │
             │   (Reverse Proxy / L7)    │
             ├───────────────────────────┤
             │ • Chunk Hasher (XXH3-64)  │
             │ • Concurrent Radix Trie   │
             │ • Shadow LRU State Mirror │
             │ • Concurrency-Aware Score │
             └─────────────┬─────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Worker 01   │    │  Worker 02   │    │  Worker 03   │
│ (LCP Match)  │    │  (Partial)   │    │   (Cold)     │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Routing Objective Function

The router optimizes a joint objective balancing Longest Common Prefix (LCP) reuse against replica in-flight queue saturation:

$$\text{Score}(W_i) = \alpha \cdot \frac{\text{LCP}(P, W_i)}{|P|} - \beta \cdot \frac{\text{ActiveRequests}(W_i)}{\text{MaxConcurrency}_i}$$

Where:
- $P$ is the sequence of token block hashes for the incoming prompt.
- $\text{LCP}(P, W_i)$ is the length of the longest matching prefix cached on worker $W_i$.
- $|P|$ is the total number of blocks in the prompt.
- $\alpha \in [0, 1]$ controls cache affinity weighting (default: `0.75`).
- $\beta \in [0, 1]$ controls concurrency penalty weighting (default: `0.25`).

---

## Benchmark Results

Evaluation was conducted using a synthetic trace of 20 concurrent agent sessions (shared system definitions and tool schemas, interleaved multi-turn queries) across 4 simulated inference workers under constrained KV cache capacity (150 blocks per worker):

| Routing Policy | Total Requests | Cache Hit Rate | Mean TTFT | P95 TTFT | P99 TTFT | Total Prefill Compute | Speedup vs RR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Round-Robin (Prefix-Blind) | 1,000 | 68.7% | 11.65 ms | 32.40 ms | 46.80 ms | 11.65 s | 1.00x |
| Least-Loaded (Prefix-Blind) | 1,000 | 68.9% | 11.58 ms | 32.40 ms | 46.80 ms | 11.58 s | 1.01x |
| **RadixGate (Prefix-Aware)** | **1,000** | **86.6%** | **5.64 ms** | **20.28 ms** | **38.40 ms** | **5.64 s** | **2.06x** |

Observed performance characteristics:
- 51.6% reduction in aggregate GPU prefill compute cycles.
- 2.06x mean TTFT speedup under identical concurrency constraints.
- Effective suppression of cache thrashing without hot-spotting individual replicas.

---

## Features

- **Compressed Radix Tree**: Token block sequences are indexed using an adaptive radix trie with $\mathcal{O}(L)$ prefix lookup complexity, where $L$ is the prompt block depth.
- **Zero-Allocation Token Chunking**: Fast rolling XXH3 64-bit hashing across configurable block windows (default: 64 tokens, configurable for Anthropic 1,024-token breakpoints).
- **Shadow LRU Cache Tracking**: Tracks GPU memory capacity limits per worker and simulates LRU eviction to prevent routing against stale prefixes.
- **Power-of-Two-Choices with Prefix Bias (P2C-Prefix)**: Mitigates stampedes and herd-flapping by sampling two candidate replicas and picking the best joint score under bounded loads.
- **Dual API Compatibility**: Native endpoint support for both Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`) schemas.
- **Zero-Copy Streaming**: Pass-through Server-Sent Events (`text/event-stream`) streaming with injected telemetry headers (`X-Radix-Worker`, `X-Radix-Matched-Blocks`, `X-Radix-Hit-Ratio`).
- **Telemetry & OpenMetrics Exporter**: Native Prometheus OpenMetrics text format exporter (`/metrics` with auto-negotiation for `text/plain` vs `application/json`) and health checks (`/health`).
- **Live Topology Visualizer**: Real-time ASCII prefix tree rendering via HTTP (`/api/v1/tree`) or CLI (`radix-gate tree`).
- **Interactive Terminal Dashboard (TUI)**: 60-FPS full-terminal cluster dashboard (`radix-gate top`) displaying live cache hit gauges, throughput counters, tree depth, and GPU fleet memory usage.
- **Proactive Hotspot Replication**: Automatically detects viral prompt prefix branches and pre-warms secondary worker replicas to prevent single-GPU cache saturation.
- **Prefill-Decode Disaggregation (PD Separation)**: Differentiates compute-dense `Prefill` nodes (FLOPs-bound) from memory-dense `Decode` nodes (KV-bandwidth bound), routing cold massive prompts to prefillers while pinning warm hits to decoders.
- **Distributed Multi-Gateway Mesh**: Peer-to-peer UDP gossip protocol maintaining cluster-wide cache coherence across multiple proxy replicas in $< 1\text{ms}$ without a centralized database.
- **High-Concurrency Socket Load Generator**: Built-in async benchmark engine (`radix-gate load-test`) testing real socket connection pools, P50-P99 latencies, and cache hit ratios.
- **Declarative YAML Engine & Zero-Downtime Hot-Reload**: Declarative `radixgate.yaml` configuration engine with atomic dynamic policy reloading via HTTP `POST /-/reload` or Unix `SIGHUP`.
- **Kubernetes Service Discovery & Helm Chart**: Automated reconciliation with upstream headless GPU services (`vllm-headless`), auto-registering scaled pods and evicting scaled-down nodes with production Helm v3 packaging.

---

## Installation & Build

Requires Rust 1.80+.

```bash
git clone https://github.com/SamGanYX/radix-gate.git
cd radix-gate
cargo build --release
```

Run test suite:
```bash
cargo test
```

---

## Usage

### 1. Run Benchmark Suite

```bash
cargo run --release -- bench --workers 4 --requests 1000 --kv-blocks 150 --agents 20
```

Available flags:
- `--workers <COUNT>`: Number of simulated worker replicas (default: `4`).
- `--requests <COUNT>`: Number of requests in trace (default: `800`).
- `--kv-blocks <COUNT>`: Worker KV memory capacity in blocks (default: `500`).
- `--agents <COUNT>`: Number of distinct template personas (default: `6`).
- `--block-size <TOKENS>`: Token chunk granularity (default: `64`).

### 2. Run Proxy Server

Start with mock workers (standalone verification):
```bash
cargo run --release -- serve --port 8080 --mock-workers 4
```

Start with live upstream replicas:
```bash
cargo run --release -- serve --port 8080 --upstreams "http://10.0.0.1:8000,http://10.0.0.2:8000"
```

### 3. Send Requests

#### Anthropic Messages API
```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are a database optimization engine.",
    "messages": [
      {"role": "user", "content": "Explain index selectivity."}
    ]
  }'
```

#### OpenAI Chat Completions API
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a code reviewer."},
      {"role": "user", "content": "Review this patch."}
    ]
  }'
```

#### Telemetry & Metrics (Prometheus / OpenMetrics)

Prometheus scrape target (default `text/plain`):
```bash
curl http://localhost:8080/metrics
```

Output:
```text
# HELP radixgate_requests_total Total number of routed inference requests
# TYPE radixgate_requests_total counter
radixgate_requests_total 1000

# HELP radixgate_cache_hit_ratio Global cluster KV-cache prefix hit ratio
# TYPE radixgate_cache_hit_ratio gauge
radixgate_cache_hit_ratio 0.8660

# HELP radixgate_matched_blocks_total Total token blocks matched across cache
# TYPE radixgate_matched_blocks_total counter
radixgate_matched_blocks_total 15420

# HELP radixgate_prompt_blocks_total Total token blocks received in prompts
# TYPE radixgate_prompt_blocks_total counter
radixgate_prompt_blocks_total 17800

# HELP radixgate_worker_load Active requests in-flight per inference worker
# TYPE radixgate_worker_load gauge
radixgate_worker_load{worker="worker-01"} 2
radixgate_worker_load{worker="worker-02"} 1
```

JSON telemetry format is available via `Accept: application/json`:
```bash
curl -H "Accept: application/json" http://localhost:8080/metrics
```

### 4. Live Prefix Tree Inspection

Display the active compressed Radix Tree topology, showing cached workers per branch, prefix block ranges, in-flight leases, and hit counts:

```bash
cargo run --release -- tree --url http://localhost:8080
```

Or query the HTTP endpoint directly:
```bash
curl http://localhost:8080/api/v1/tree
```

Sample output:
```text
ROOT [block_size: 64 tokens]
├── [0x8f2a1b9c..0x3e110a72] (4 blk) [Cached: worker-01, worker-03] (Hits: 284)
│   ├── [0x09a4128f] [Cached: worker-01] [InFlight: 1] (Hits: 92)
│   └── [0x7b5e4120..0x91d302a4] (2 blk) [Cached: worker-03] (Hits: 14)
└── [0x41f8021c] [Cached: worker-02] (Hits: 110)
```

### 5. Interactive Terminal Dashboard (`radix-gate top`)

Launch the real-time 60-FPS terminal UI dashboard (built with `ratatui` + `crossterm`) to inspect live cluster health, GPU worker memory utilization, cache hit percentages, and active tree branches:

```bash
cargo run --release -- top --url http://127.0.0.1:8080
```

Keybindings:
- `q` or `Esc`: Cleanly exit dashboard.
- `r`: Force immediate telemetry poll.

```text
╭─ RADIX-GATE :: Distributed KV-Cache Gateway & Topology Inspector [http://127.0.0.1:8080] Status: ● CONNECTED ─╮
│                                                                                                              │
│ ┌ Global KV Hit Ratio ┐  ┌ Throughput ──────────┐  ┌ Block Cache Match ────┐  ┌ Radix Topology ────────────┐ │
│ │ [██████████] 86.7%  │  │ 1,240 reqs           │  │ 18,420 / 21,240       │  │ 42 active branches         │ │
│ └─────────────────────┘  └──────────────────────┘  └───────────────────────┘  └────────────────────────────┘ │
│                                                                                                              │
│ ┌ Live Radix Tree Topology ──────────────────────────┐  ┌ Inference Worker Fleet (KV-Cache & In-Flight Load) ┐│
│ │ ROOT [block_size: 64 tokens]                       │  │ ● OK  worker-01  InFlight:  2/32 ( 6%)  KV: 68.4%   ││
│ │ ├── [0x8f2a1b9c..0x3e110a72] (4 blk)               │  │    [████████████████        ]                      ││
│ │ │   [Cached: worker-01, worker-02] (Hits: 412)     │  │ ● OK  worker-02  InFlight:  1/32 ( 3%)  KV: 54.2%   ││
│ │ │   ├── [0x09a4128f] [Cached: worker-01] (Hits: 92)│  │    [█████████████           ]                      ││
│ └────────────────────────────────────────────────────┘  └────────────────────────────────────────────────────┘│
╰─────────────────────────────── [q] Quit  |  [r] Force Refresh  |  RadixGate v0.1.0 ───────────────────────────╯
```

### 6. High-Concurrency Async Socket Load Generator

Benchmark real socket throughput, TCP connection reuse, P50–P99 latencies, and cache efficiency under multi-turn agent workloads:

```bash
cargo run --release -- load-test \
  --url http://127.0.0.1:8080 \
  --concurrency 50 \
  --requests 1000 \
  --endpoint messages \
  --shared-prefixes 4 \
  --turns 3
```

Available flags:
- `--url <URL>`: Gateway base URL (default: `http://127.0.0.1:8080`).
- `--concurrency <N>`: Concurrent async Tokio worker tasks (default: `20`).
- `--requests <N>`: Total requests to execute (default: `200`).
- `--endpoint <ENDPOINT>`: `messages` (`/v1/messages`) or `chat` (`/v1/chat/completions`).
- `--shared-prefixes <N>`: Number of distinct shared system prompt archetypes (default: `4`).
- `--turns <N>`: Sequential conversation turns per session before rotating prefix (default: `3`).

### 7. Docker Compose & Multi-Gateway Mesh

Launch a clustered dual-gateway deployment with automatic UDP peer gossip mesh synchronization enabled:

```bash
docker compose up --build
```

- Gateway Node 01: `http://localhost:8080` (Gossip UDP `9090`)
- Gateway Node 02: `http://localhost:8081` (Gossip UDP `9091`)

Requests sent to Gateway Node 01 are immediately synchronized to Gateway Node 02's local trie in sub-millisecond time.

### 8. vLLM / SGLang Upstream Eviction Sidecar

When physical GPU workers evict blocks under memory pressure, use the Python connector to synchronize state directly:

```python
from vllm.entrypoints.openai.api_server import app
from connectors.vllm_sidecar import RadixGateMiddleware, RadixGateSyncClient

# Attach sidecar middleware to inject X-Evicted-Blocks trailers and emit POST /v1/cache/sync
app.add_middleware(
    RadixGateMiddleware,
    sync_client=RadixGateSyncClient("http://radix-gate:8080", "worker-01")
)
```

### 9. True Bidirectional Server-Sent Events (SSE) Streaming

RadixGate features native zero-copy Server-Sent Events passthrough (`text/event-stream; charset=utf-8`) with active concurrency tracking:

```bash
# OpenAI SSE streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain KV-cache streaming."}],
    "stream": true
  }'

# Anthropic SSE streaming
curl -N http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Explain KV-cache streaming."}],
    "stream": true
  }'
```

- **Zero-Buffering**: Chunks are forwarded immediately with `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no` for minimal TTFT.
- **RAII `InFlightGuard`**: Concurrency counters on worker replicas stay accurately tracked until the final stream token is emitted or client disconnect occurs, preventing load balancer estimation drift.

### 10. Multi-Arch Container Images & GitHub Actions CI/CD

Pre-built multi-platform container images are automatically published via GitHub Actions to the GitHub Container Registry (GHCR):

```bash
# Pull and run multi-arch image (linux/amd64, linux/arm64)
docker pull ghcr.io/samganyx/radix-gate:latest
docker run -p 8080:8080 ghcr.io/samganyx/radix-gate:latest serve --mock-workers 4
```

### 11. Declarative Configuration & Zero-Downtime Hot-Reload

RadixGate supports enterprise YAML configuration (`config/radixgate.example.yaml`) managing chunking parameters, load balancing policies, mesh peering, and discovery:

```bash
# Start gateway with declarative YAML configuration
cargo run --release -- --config config/radixgate.example.yaml
```

Update balancing parameters, timeouts, or worker weights at runtime with **zero downtime**:
- **HTTP Reload Endpoint**: `curl -X POST http://localhost:8080/-/reload`
- **POSIX Signal Listener**: `kill -HUP <radixgate_pid>` (triggers atomic thread-safe `Arc<RwLock<RoutingPolicy>>` swap)

### 12. Kubernetes Service Discovery & Production Helm Chart

Deploy RadixGate natively to Kubernetes with automated Horizontal Pod Autoscaling (HPA) reconciliation and headless service discovery:

```bash
# Deploy using the production Helm v3 chart
helm install radix-gate ./charts/radix-gate \
  --set radixConfig.discovery.enabled=true \
  --set radixConfig.discovery.service_address="vllm-headless.default.svc.cluster.local:8000"
```

**Key Kubernetes Capabilities:**
- **Headless DNS Reconciler**: Periodically resolves upstream headless services (e.g., `vllm-headless:8000`) via standard non-blocking DNS queries—requiring **zero RBAC cluster permissions**.
- **Scale-Up Detection**: Newly spawned GPU pods are automatically registered with full shadow LRU tracking without restarting the gateway.
- **Scale-Down & Eviction Safety**: Terminated or scaled-down pods are immediately removed from the active worker pool and pruned from the Radix Tree, preventing cache hallucination.
- **Native Health & Observability**: Integrated `/health` liveness/readiness probes and optional Prometheus Operator `PodMonitor` resource.

---

## Distributed Systems Design & Edge-Case Handling

### 1. Thundering Herd Suppression via In-Flight Prefill Leases
In bursty multi-agent workflows, multiple concurrent requests sharing a previously unseen prefix frequently arrive within the same millisecond window. A naive prefix router registers prefixes only *after* request completion, which causes all subsequent concurrent requests to miss and scatter across independent workers—triggering $N$ redundant cold prefills.

RadixGate solves this through **Optimistic In-Flight Prefill Leases**:
- When Request 1 is dispatched to worker $W_i$, an in-flight reservation lease is recorded along the trie branch.
- When Requests 2 through $K$ arrive while Request 1 is still computing, the router scores $W_i$'s in-flight warming prefix at 90% weighting of a confirmed hit.
- The router coalesces the burst by queueing subsequent requests behind $W_i$. When Request 1 finishes, Requests 2 through $K$ execute against a 100% warm KV cache.

### 2. Active Eviction Synchronization vs. Shadow Drift
Physical GPU workers dynamically evict prefix blocks when output token generation surges or memory is preempted. A purely passive gateway shadow cache inevitably drifts from physical VRAM reality, leading to **cache hallucination** (routing requests expecting hits that are cold on the GPU).

RadixGate provides bi-directional eviction synchronization:
- **Upstream Response Trailers**: Workers emit `X-Evicted-Blocks: <hash_csv>` on completions when blocks are dropped.
- **Dedicated Sync Ingress**: Upstream engines broadcast asynchronous eviction receipts directly to `POST /v1/cache/sync`.
- The gateway immediately prunes evicted block hashes from the trie, ensuring state fidelity.

### 3. Exact BPE Token Boundary Alignment
Word-level or naive character chunking fails in production because Byte-Pair Encoding (BPE) is non-associative across arbitrary string boundaries. If the gateway hashes text approximations rather than exact BPE token sequences, the generated 64-bit block hashes will not match the upstream engine's PagedAttention block keys.

RadixGate integrates `tiktoken-rs` (`cl100k_base`), chunking directly over canonical u32 token IDs on exact token boundaries (e.g. 64-token PagedAttention blocks or 1,024-token Anthropic cache breakpoints).

### 4. Partitioned Concurrency (Sharded Radix Trie)
Under high concurrency, acquiring a global write lock on a central trie for every request completion creates severe CPU contention. RadixGate provides `ShardedRadixTree`, which partitions the trie into $N$ independent shards indexed by a 64-bit mixer hash of the prompt's root token block. Requests with distinct prefixes operate in parallel across independent `RwLock` shards without lock contention.

### 5. Stampede & Herd-Flapping Prevention via P2C-Prefix
Greedy prefix matching (always picking the replica with the highest $\text{LCP}$) causes pathological stampedes when thousands of concurrent clients query the same popular prefix. The single replica holding that prefix becomes saturated, causing queued request latencies to far exceed the time required to simply re-prefill the prompt on an idle replica.

RadixGate implements **Power-of-Two-Choices with Prefix Bias (P2C-Prefix)** (`PowerOfTwoChoicesPolicy`):
- Randomly samples two candidate replicas $W_a$ and $W_b$ from the cluster.
- Evaluates composite scores balancing normalized prefix length against queued in-flight requests.
- Applies a hard load ceiling ($c \cdot \bar{L}$, where $\bar{L}$ is average cluster load). If a candidate exceeds the bounded load ceiling, the router spills traffic to an underloaded alternative, bounding maximum queue delay to $\mathcal{O}(\log \log N)$ while maintaining high prefix hit rates.

### 6. Circuit Breaking, Automatic Failover, and Active Health Probing
If an upstream GPU worker panics or its host machine crashes, naive routers black-hole client traffic because the dead worker still appears to hold cached prefixes.

RadixGate integrates automated fault tolerance:
- **Immediate Failover Retry**: When an upstream request fails (socket drop, connection timeout, 5xx), the gateway catches the failure, increments the worker's failure counter, and immediately fails over to the next healthy candidate in the cluster rather than failing the client request.
- **Circuit Breaker Tripping**: After 3 consecutive upstream failures, the worker's circuit breaker trips (`is_healthy = false`). The router instantly purges the worker's leaf nodes from the Radix Tree so subsequent queries are not misrouted.
- **Strict 502 Handling**: If all healthy upstreams are exhausted, the gateway returns a standard `HTTP 502 Bad Gateway` JSON error—never a synthetic mock string.
- **Background Health Recovery Loop**: A background task periodically probes `{worker}/health`. When an offline worker restarts and returns `200 OK`, it is automatically marked healthy and admitted back into the load-balancing pool.

### 7. Early Prefix Cutoff (Gateway CPU Protection)
Full BPE tokenization over massive prompts (50k+ tokens) on the gateway hot path introduces multi-millisecond CPU delays. RadixGate enforces an early prefix cutoff (`max_prefix_tokens: 4096`), slicing long prompt strings at clean whitespace boundaries and tokenizing only the prefix depth necessary for cluster routing decisions. Unchunked trailing tokens remain untouched for pass-through to the upstream model, preventing gateway CPU saturation at high QPS.

### 8. Proactive Hotspot Replication (Auto-Duplicating Viral Prefixes)
In multi-agent clusters, a single popular system prompt (e.g. standard developer instructions or coding schemas) can experience severe traffic bursts. Greedy routing saturates the sole worker holding the prefix, while simple fallback spills excess queries to cold replicas—imposing a 100% cold prefill latency penalty.

RadixGate implements **Proactive Hotspot Replication**:
- Branch nodes in the Radix Tree track access frequency and access velocity.
- When a prefix branch's hit frequency crosses a hotspot threshold and its primary worker exceeds bounded load ($L \ge 0.50$), the router flags the decision with `replicate_to_worker`.
- The gateway registers the prefix with an idle secondary replica and pre-warms its shadow LRU cache.
- Subsequent queries sharing the viral prefix benefit from two warm replicas, allowing P2C-Prefix to distribute load across GPUs with **100% cache hits**.
- When traffic cools, secondary replicas naturally evict the prefix via O(1) LRU capacity limits.

### 9. Prefill-Decode Disaggregation (PD-Aware Scheduling)
Modern frontier inference architectures (Mooncake, Splitwise, vLLM P/D) disaggregate clusters into:
- **Prefill Nodes (`WorkerRole::Prefill`)**: Compute-dense hardware (e.g. SXM5 H100s) optimized for high-FLOP matrix multiplication during the initial prompt ingestion phase (TTFT).
- **Decode Nodes (`WorkerRole::Decode`)**: Memory-bandwidth-dense hardware with large KV pools optimized for autoregressive token generation (TBT).
- **Unified Nodes (`WorkerRole::Unified`)**: Standard monolithic replicas capable of serving both phases.

RadixGate actively inspects prefix affinity before dispatching to heterogeneous worker roles:
- **Cold Prompts ($\text{Hit Ratio} < 35\%$)**: Routed to `Prefill` nodes to execute heavy matrix multiplication without degrading the memory bandwidth and decoding latency of in-flight requests.
- **Warm Prompts ($\text{Hit Ratio} \ge 70\%$)**: Routed directly to `Decode` nodes holding the cached KV prefix, completely bypassing prefill queues.

### 10. Distributed Multi-Gateway Mesh (Cluster-Wide Cache Coherence)
In large deployments where multiple RadixGate reverse proxy replicas sit behind an AWS ALB or Cloudflare, single-node in-memory caches fragment. If Client A routes a prefix through Gateway 1 to Worker 3, Gateway 2 normally has no awareness of this mapping and may route Client B's follow-up to Worker 1, destroying cache reuse.

RadixGate implements an autonomous **Peer Gossip Mesh Protocol**:
- Each gateway binds a UDP gossip listener (`--gossip-bind 0.0.0.0:9090 --peers <peer_ips>`).
- When any gateway commits a prefix to an inference worker, it emits a compact binary UDP gossip packet: `[Magic: 'RGAT', NodeID, WorkerID, BlockHashes, Seq, Timestamp]`.
- Peer gateways ingest the packet in sub-millisecond background loops, deduplicating via `(node_id, seq)` sequence windows.
- Remote prefixes are synchronized directly into peer local Radix Trees without any centralized Redis/database bottleneck or single point of failure.

---

## Architecture Details

- `src/trie/`: Radix tree implementation with branch splitting, atomic in-flight commit, ASCII visualizer, and sharded partitioning (`ShardedRadixTree`).
- `src/tokenizer.rs`: BPE token chunker (`tiktoken-rs`) with early prefix cutoff and sub-block residual hashing.
- `src/balancer/`: Multi-objective routing policies (`PrefixAware`, `PowerOfTwoChoices`, `RoundRobin`, `LeastLoaded`) with proactive hotspot replication and role-biased disaggregation.
- `src/mesh/`: Distributed peer gossip protocol for cluster-wide cache coherence across gateway instances.
- `src/shadow_cache/`: Worker capacity tracker, circuit breaker state, and O(1) doubly-linked list LRU state mirror.
- `src/proxy/`: Axum-based HTTP proxy, Prometheus OpenMetrics exporter, automatic failover retries, background health prober, and cache sync ingress.
- `src/tui.rs`: Real-time 60-FPS terminal dashboard (`radix-gate top`) built with `ratatui` and `crossterm`.
- `src/load_test.rs`: High-concurrency async socket load generator and percentile latency analyzer.
- `src/simulator/`: Discrete event benchmark harness with prefill/decode latency model.

---

## References

1. Anthropic. (2024). *Prompt Caching with Claude*. https://www.anthropic.com/news/prompt-caching
2. Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. https://arxiv.org/abs/2309.06180
3. Zheng, L., et al. (2023). *SGLang: Efficient Execution of Structured Language Models with RadixAttention*. https://arxiv.org/abs/2312.07104

---

## License

Apache-2.0
