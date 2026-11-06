# ⚡ RadixGate: Prefix-Aware KV-Cache Gateway & Radix Router

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Target](https://img.shields.io/badge/target-Inference%20Infra%20%7C%20Anthropic%20%7C%20vLLM-purple.svg)]()

> **A high-throughput, deterministic reverse proxy and Radix Router that balances distributed LLM inference clusters by tracking GPU KV-cache state and maximizing prompt prefix reusability.**

---

## 🎯 The Motivation: Why Prefix Routing Matters

Modern LLM inference engines (vLLM, SGLang, and Anthropic Claude) feature **prompt caching** (PagedAttention / RadixAttention). Reusing cached key-value (KV) states reduces Time-To-First-Token (TTFT) by up to **85%** and cuts token serving compute costs by up to **90%**.

### The Failure Mode of Distributed LLM Serving
In multi-worker production clusters ($N > 1$), standard L7 load balancers (Round-Robin, Random, or Least-Connections) are **prefix-blind**:
- Turn 1 of an agentic workflow (with an 8,000-token system prompt and tool schema) is routed to **Worker 1**.
- Turn 2 is routed to **Worker 2** $\to$ Worker 2 has a cold cache and must recompute all 8,000 tokens from scratch.
- Turn 3 is routed to **Worker 3** $\to$ Cold cache again.

This destroys KV-cache locality, causes massive VRAM cache thrashing, and reduces cluster cache hit rates to near zero under concurrent load.

**RadixGate solves this at the gateway layer.** By maintaining a shadow Radix Tree of what block prefixes are hot across each worker replica, incoming requests are routed deterministically to the worker that possesses the Longest Common Prefix (LCP), while penalizing queue depth to prevent hot-spotting.

---

## 🏛️ System Architecture

```
                             [ Client / Agent Request ]
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │              RadixGate               │
                      │   (High-Performance Reverse Proxy)   │
                      ├──────────────────────────────────────┤
                      │  1. Fast Token Chunker (XXH3-64)     │
                      │  2. Radix Tree Prefix Matcher        │
                      │  3. Affinity + Queue Score Engine    │
                      │  4. Worker Shadow LRU Tracker        │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
      │     Worker 01       │ │     Worker 02       │ │     Worker 03       │
      │  (vLLM / SGLang)    │ │  (vLLM / SGLang)    │ │  (vLLM / SGLang)    │
      │ Hot: [SysPrompt A]  │ │ Hot: [SysPrompt B]  │ │ Hot: [SysPrompt C]  │
      └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

### Routing Objective Function
Routing decisions optimize a multi-objective trade-off between cache affinity and active worker concurrency:

$$\text{Score}(W_i) = \alpha \cdot \frac{\text{LCP}(P, W_i)}{|P|} - \beta \cdot \frac{\text{ActiveRequests}(W_i)}{\text{MaxConcurrency}_i}$$

Where:
- $\text{LCP}(P, W_i)$ is the Longest Common Prefix (in token blocks) between request $P$ and worker $W_i$.
- $\alpha = 0.75$ (affinity weight), $\beta = 0.25$ (load penalty).

---

## 📊 Empirical Benchmark Results

Running against 20 concurrent agentic personas across 4 worker replicas with constrained KV memory (150 blocks/worker) under interleaved traffic:

```
==========================================================================================
               RADIX-GATE: DISTRIBUTED KV-CACHE ROUTER BENCHMARK RESULTS                  
==========================================================================================
╭─────────────────────────────┬──────────┬───────────────────┬───────────────┬───────────────┬───────────────┬─────────────────────┬───────────────╮
│ Routing Policy              ┆ Requests ┆ KV Cache Hit Rate ┆ Avg TTFT (ms) ┆ P95 TTFT (ms) ┆ P99 TTFT (ms) ┆ Prefill Compute (s) ┆ Speedup vs RR │
╞═════════════════════════════╪══════════╪═══════════════════╪═══════════════╪═══════════════╪═══════════════╪═════════════════════╪═══════════════╡
│ Round-Robin (Prefix-Blind)  ┆ 1000     ┆ 68.7%             ┆ 11.65 ms      ┆ 32.40 ms      ┆ 46.80 ms      ┆ 11.65 s             ┆ 1.00x         │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ Least-Loaded (Prefix-Blind) ┆ 1000     ┆ 68.9%             ┆ 11.58 ms      ┆ 32.40 ms      ┆ 46.80 ms      ┆ 11.58 s             ┆ 1.01x         │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ RadixGate (Prefix-Aware)    ┆ 1000     ┆ 86.6%             ┆ 5.64 ms       ┆ 20.28 ms      ┆ 38.40 ms      ┆ 5.64 s              ┆ 2.06x         │
╰─────────────────────────────┴──────────┴───────────────────┴───────────────┴───────────────┴───────────────┴─────────────────────┴───────────────╯
```

**Key Takeaways**:
- **51.6% reduction in GPU prefill compute cycles** (from 11.65s down to 5.64s).
- **2.06x speedup** on Time-To-First-Token (TTFT).
- **Cluster KV Hit Rate improved from 68.7% to 86.6%** without sacrificing worker load balance.

---

## 🚀 Quickstart

### 1. Build & Run Tests
```bash
cargo test
```

### 2. Run the Benchmark Simulator
```bash
cargo run --release -- bench --workers 4 --requests 1000 --kv-blocks 150 --agents 20
```

### 3. Start the Gateway Proxy
```bash
cargo run --release -- serve --host 127.0.0.1 --port 8080 --mock-workers 4
```

### 4. Query with Anthropic or OpenAI API Schemas

**Anthropic Messages API:**
```bash
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are a senior inference infrastructure engineer.",
    "messages": [
      {"role": "user", "content": "How does Radix routing optimize KV cache hits?"}
    ]
  }'
```

**Inspect Live Cluster & Radix Tree Metrics:**
```bash
curl http://127.0.0.1:8080/metrics
```

---

## 🛠️ Codebase Structure

- `src/trie/`: Compact Radix Tree data structure for 64-bit token block hash prefix matching.
- `src/tokenizer.rs`: Deterministic XXH3 rolling chunker (64-token / 1024-token windows).
- `src/balancer/`: Multi-objective affinity + queue penalty routing algorithms.
- `src/shadow_cache/`: Worker replica KV capacity tracking and shadow LRU eviction.
- `src/proxy/`: Axum async streaming reverse proxy with OpenAI & Anthropic compatibility.
- `src/simulator/`: Realistic distributed inference simulation harness with PagedAttention hardware physics.

---

## 📚 References & Background Reading
- [Anthropic: Prompt Caching with Claude](https://www.anthropic.com/news/prompt-caching)
- [Anthropic Research: Effective Context Engineering for AI Agents](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents)
- [SGLang: Efficient Execution of Structured Language Models (RadixAttention)](https://arxiv.org/abs/2312.07104)
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

---

## License
Apache-2.0
