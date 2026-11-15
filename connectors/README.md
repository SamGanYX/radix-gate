# RadixGate vLLM / SGLang Sidecar Connector & Python SDK

This directory and the official `radixgate` Python package (`sdk/python`) provide integration tools and middleware for bridging physical inference engines (vLLM, SGLang, TensorRT-LLM) directly with RadixGate's L7 routing plane.

## Installation

```bash
pip install radixgate
```
Or install directly from this repository:
```bash
pip install -e sdk/python
```

## Overview

When physical GPU workers evict prompt prefix blocks under memory pressure, RadixGate must be notified immediately to eliminate **cache hallucination** (routing requests expecting cache hits on evicted workers).

The connector provides two integration paths:

### 1. ASGI Middleware Hook (`RadixGateMiddleware`)

Wrap any ASGI/FastAPI application (such as `vllm.entrypoints.openai.api_server`):

```python
from vllm.entrypoints.openai.api_server import app
from radixgate.vllm import attach_vllm_app

# Attaches ASGI middleware to inject 'X-Evicted-Blocks' headers and notify gateway
attach_vllm_app(app, gateway_url="http://radix-gate:8080", worker_id="worker-01")
```

For SGLang servers:
```python
from radixgate.sglang import attach_sglang_app

attach_sglang_app(sglang_server.app, gateway_url="http://radix-gate:8080", worker_id="worker-01")
```

### 2. Standalone Asynchronous Eviction Daemon

Run as a companion container alongside your vLLM pod:

```bash
python connectors/vllm_sidecar.py
```

Environment Variables:
- `RADIXGATE_URL`: Base URL of the RadixGate proxy (default: `http://127.0.0.1:8080`).
- `WORKER_ID`: Unique ID of the inference replica (default: `worker-01`).
