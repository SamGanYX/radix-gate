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
- **Dual API Compatibility**: Native endpoint support for both Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`) schemas.
- **Zero-Copy Streaming**: Pass-through Server-Sent Events (`text/event-stream`) streaming with injected telemetry headers (`X-Radix-Worker`, `X-Radix-Matched-Blocks`, `X-Radix-Hit-Ratio`).
- **Telemetry & Metrics**: Built-in Prometheus-compatible runtime metrics (`/metrics`) and health checks (`/health`).

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

#### Telemetry
```bash
curl http://localhost:8080/metrics
```

Response includes trie node count, cache hit ratio, active worker distribution, and per-worker utilization:
```json
{
  "performance": {
    "cluster_cache_hit_ratio": "86.60%",
    "total_matched_blocks": 15420,
    "total_prompt_blocks": 17800,
    "total_requests": 1000
  },
  "radix_tree": {
    "total_nodes": 42,
    "unique_workers": 4
  },
  "status": "healthy"
}
```

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

---

## Architecture Details

- `src/trie/`: Radix tree implementation with branch splitting, in-flight leases, and sharded partitioning (`ShardedRadixTree`).
- `src/tokenizer.rs`: Exact BPE token chunker (`tiktoken-rs`) and zero-allocation streaming fallback.
- `src/balancer/`: Multi-objective routing policies (`PrefixAware`, `RoundRobin`, `LeastLoaded`).
- `src/shadow_cache/`: Worker capacity tracker and O(1) doubly-linked list LRU state mirror.
- `src/proxy/`: Axum-based HTTP proxy, request parser, cache sync handler, and telemetry decorator.
- `src/simulator/`: Discrete event benchmark harness with prefill/decode latency model.

---

## References

1. Anthropic. (2024). *Prompt Caching with Claude*. https://www.anthropic.com/news/prompt-caching
2. Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. https://arxiv.org/abs/2309.06180
3. Zheng, L., et al. (2023). *SGLang: Efficient Execution of Structured Language Models with RadixAttention*. https://arxiv.org/abs/2312.07104

---

## License

Apache-2.0
