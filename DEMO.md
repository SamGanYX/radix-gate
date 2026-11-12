# RadixGate Showcase & Verification Runbook

This document is an engineer-to-engineer walkthrough designed for systems interviewers, inference platform architects, and evaluators looking to inspect and verify the mechanisms powering **RadixGate**.

---

## 1. Quickstart Automated Verification

To run the automated end-to-end verification pipeline (release compilation, test suite, comparative multi-strategy benchmark, live daemon startup, socket load generation, tree topology extraction, and OpenMetrics export):

### On Linux / macOS (POSIX):
```bash
chmod +x scripts/demo.sh
./scripts/demo.sh
```

### On Windows (PowerShell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\demo.ps1
```

---

## 2. Step-by-Step Verification Scenarios

### Scenario A: Dual-Gateway Gossip Mesh & Prefill-Decode Disaggregation

This scenario proves:
1. Two independent gateways peering over UDP without centralized Redis or Consul coordination.
2. Cold prompts (>2 blocks) automatically routing to **Prefill** nodes.
3. Peer-to-peer UDP gossip synchronizing cache state so the second gateway immediately achieves a 100% warm hit without re-executing prefill.

**Run command:**
```bash
cargo test --test e2e_showcase test_e2e_dual_gateway_mesh_and_disaggregation -- --nocapture
```

**Expected output:**
```text
running 1 test
test test_e2e_dual_gateway_mesh_and_disaggregation ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.81s
```

---

### Scenario B: Empirical Comparative Benchmark (Round-Robin vs Least-Loaded vs RadixGate)

Simulates a 4-worker vLLM cluster subjected to synthetic multi-agent multi-turn conversational traces (e.g. shared agent system instructions, multi-turn tool traces, shared few-shot examples):

**Run command:**
```bash
cargo run --release -- bench --requests 1000 --workers 4 --kv-blocks 600
```

**What to look for in output:**
- **Cache Hit Rate**: RadixGate typically delivers **60%–85%** KV cache reuse, whereas Round-Robin achieves **15%–25%** and Least-Loaded achieves **20%–30%**.
- **P99 Latency Reduction**: Demonstrates up to **4x lower tail latency** by avoiding redundant attention prefill passes.
- **Worker Load Distribution**: P2C balancer avoids skewing requests while maintaining prefix affinity.

---

### Scenario C: High-Concurrency Async Socket Load Test Against Live Daemon

Launches the gateway in daemon mode with simulated upstream inference replicas, then executes an asynchronous load test hammering `/v1/messages` (Anthropic format) with 20 concurrent tasks generating multi-turn agent conversations:

**Step 1 — Start the Gateway:**
```bash
cargo run --release -- serve --port 8080 --mock-workers 4
```

**Step 2 — Execute the Load Generator (in another terminal):**
```bash
cargo run --release -- load-test --url http://127.0.0.1:8080 --concurrency 20 --requests 200 --turns 4
```

**Step 3 — Inspect Active Prefix Trie Topology:**
```bash
cargo run --release -- tree --url http://127.0.0.1:8080
```
*Visualizes active radix branches, block hashes, token lengths, and worker affinity mappings.*

**Step 4 — Scrape Prometheus Metrics:**
```bash
curl -s http://127.0.0.1:8080/metrics | grep radixgate_
```

---

### Scenario D: Real-Time 60-FPS Terminal Observability Dashboard (`top`)

RadixGate features a built-in terminal dashboard powered by **Ratatui** for operators monitoring GPU KV-cache pressure, worker load, and cluster cache hit ratio:

**Launch command:**
```bash
cargo run --release -- top --url http://127.0.0.1:8080 --interval 250
```

**Interactive Controls:**
- `q` / `Esc`: Graceful exit.
- `r`: Force immediate metric refresh.

---

### Scenario E: Full Docker Compose Inference Cluster with vLLM Sidecar

For end-to-end containerized deployments, RadixGate ships with a multi-worker `docker-compose.yml` including Prometheus and the Python vLLM dynamic eviction sidecar adapter:

```bash
docker-compose up --build -d
```

**Exposed Services:**
- `http://localhost:8080`: RadixGate reverse proxy (`/v1/chat/completions`, `/v1/messages`).
- `http://localhost:9090`: Prometheus dashboard.
- `http://localhost:8001` – `8003`: Upstream vLLM inference worker nodes with eviction sidecars.

---

## 3. Architecture Deep-Dive for Interviewers

```
                Client Request (OpenAI / Anthropic API)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   RadixGate Router    │
                     └───────────┬───────────┘
                                 │
                 Tokenize & Chunk (xxh3_64)
                                 │
                                 ▼
                     ┌───────────────────────┐  P2P UDP Gossip   ┌───────────────────────┐
                     │ 64-Way Sharded Radix  │◄─────────────────►│ Peer RadixGate Router │
                     │      Prefix Tree      │                   │      (Gateway 2)      │
                     └───────────┬───────────┘                   └───────────────────────┘
                                 │
                 P2C Score = α(Affinity) - β(Load) + RoleBias
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                                         ▼
   Prefill Worker (Compute)                  Decode Worker (Memory)
   - Cold Prompts (Affinity < 0.40)          - Warm Hits (Affinity ≥ 0.70)
   - Long Context Ingestion                  - Auto-Regressive Generation
```

### Key Technical Distinctions:
1. **Zero-Lock Contention on Hot Paths**: 64-way independent shard locks (`parking_lot::RwLock`) allow concurrent lookups across thousands of worker threads.
2. **In-Flight Reservation & Coalescing**: Thundering-herd requests sharing the exact same cold prefix are optimistically bound to the same worker node before prefill completes, avoiding redundant parallel computation.
3. **Proactive Hotspot Replication**: Automatically identifies viral prefix nodes whose request frequency exceeds cluster mean by $>2.5\times$ and replicates prefixes to secondary idle replicas before saturation occurs.
4. **Disaggregated Prefill vs Decode Awareness**: Balancer understands heterogeneous node capabilities, directing heavy prefill computation away from memory-constrained decode nodes.
5. **Decentralized UDP Gossip**: Mesh updates are broadcast as compact 32-byte header JSON datagrams with sequence-based deduplication and loopback suppression.
