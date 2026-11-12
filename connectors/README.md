# RadixGate vLLM / SGLang Sidecar Connector

This directory contains integration tools and middleware for bridging physical inference engines (vLLM, SGLang, TensorRT-LLM) directly with RadixGate's L7 routing plane.

## Overview

When physical GPU workers evict prompt prefix blocks under memory pressure, RadixGate must be notified immediately to eliminate **cache hallucination** (routing requests expecting cache hits on evicted workers).

The sidecar provides two integration paths:

### 1. Zero-Code Response Header Decorator (`RadixGateMiddleware`)

Wrap any ASGI/FastAPI application (such as `vllm.entrypoints.openai.api_server`):

```python
from vllm.entrypoints.openai.api_server import app
from connectors.vllm_sidecar import RadixGateMiddleware, RadixGateSyncClient

client = RadixGateSyncClient(
    gateway_url="http://radix-gate:8080",
    worker_id="worker-01"
)

# Injects 'X-Evicted-Blocks' trailer headers into LLM responses
app.add_middleware(RadixGateMiddleware, sync_client=client)
```

### 2. Standalone Asynchronous Eviction Daemon

Run as a companion container alongside your vLLM pod:

```bash
python connectors/vllm_sidecar.py
```

Environment Variables:
- `RADIXGATE_URL`: Base URL of the RadixGate proxy (default: `http://127.0.0.1:8080`).
- `WORKER_ID`: Unique ID of the inference replica (default: `worker-01`).
