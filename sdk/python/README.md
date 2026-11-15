# RadixGate Python SDK & Sidecar

Official Python SDK and inference engine sidecar for **RadixGate** — Distributed, Prefix-Cache Aware Reverse Proxy for LLM inference clusters.

## Installation

```bash
pip install radixgate
```

Or install in editable mode from the repository:

```bash
pip install -e sdk/python
```

---

## 1. Client Usage

### Query Gateway & Inspect Routing Metadata

```python
from radixgate import RadixGateClient

client = RadixGateClient(base_url="http://localhost:8080")

# 1. Cluster Health & Metrics
health = client.health()
print(f"Cluster Status: {health.status} ({health.healthy_workers}/{health.total_workers} healthy)")

metrics = client.metrics()
print(f"Global KV Hit Ratio: {metrics.cache_hit_ratio * 100:.1f}%")

# 2. Dispatch Chat Request & Inspect Prefix Cache Hits
response, meta = client.chat_completion({
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a database query optimizer."},
        {"role": "user", "content": "How do B-tree indexes work?"}
    ]
})

print(f"Routed Worker: {meta.worker_id}")
print(f"Matched Blocks: {meta.matched_blocks} / {meta.prompt_blocks}")
print(f"Instantaneous Hit Ratio: {meta.cache_hit_ratio * 100:.1f}%")
```

### Async Client (`AsyncRadixGateClient`)

```python
import asyncio
from radixgate import AsyncRadixGateClient

async def main():
    async with AsyncRadixGateClient("http://localhost:8080") as client:
        resp, meta = await client.messages({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Explain KV-cache reuse."}]
        })
        print(f"Served by {meta.worker_id} with hit ratio {meta.cache_hit_ratio:.2f}")

asyncio.run(main())
```

---

## 2. Upstream Engine Integration (vLLM / SGLang Sidecar)

### Attach to vLLM Server

```python
from vllm.entrypoints.openai.api_server import app
from radixgate.vllm import attach_vllm_app, hook_block_space_manager

# Attach ASGI middleware to inject X-Evicted-Blocks headers
middleware = attach_vllm_app(
    app,
    gateway_url="http://radix-gate:8080",
    worker_id="vllm-worker-01"
)

# Optional: Hook block manager to capture frees automatically
# hook_block_space_manager(llm_engine.block_manager, middleware)
```

### Attach to SGLang Server

```python
from radixgate.sglang import attach_sglang_app, hook_radix_cache

middleware = attach_sglang_app(
    sglang_server.app,
    gateway_url="http://radix-gate:8080",
    worker_id="sglang-worker-01"
)
```

---

## Features

- **Pydantic V2 Models**: Type-safe telemetry, health, and eviction models.
- **Prefix Header Parsing**: Extracts `X-Radix-Worker`, `X-Radix-Matched-Blocks`, `X-Radix-Hit-Ratio` into structured objects.
- **Bi-Directional Eviction Sync**: Automatically informs RadixGate when GPU VRAM evicts prompt blocks, preventing cache hallucination.
- **Sync & Async**: First-class async support built on `httpx`.
