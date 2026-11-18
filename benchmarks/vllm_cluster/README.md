# Real vLLM Cluster Benchmark: NGINX Round-Robin vs. RadixGate

This directory contains a turnkey benchmark harness to measure **real GPU Time-To-First-Token (TTFT)** and **actual vLLM PagedAttention prefix cache hit rates** comparing standard NGINX Round-Robin against `radix-gate`.

---

## Hardware Requirements

You can run 4 independent vLLM replicas on either:
- **Option A (Lowest Cost, ~$0.40/hr)**: 1x 24GB+ GPU (RTX 3090 / RTX 4090 / RTX PRO 4500 / A10G / L4) running 4 `vLLM` processes with `--gpu-memory-utilization 0.20` and `Qwen/Qwen2.5-1.5B-Instruct`.
- **Option B (Multi-GPU)**: 4x GPUs with `MULTI_GPU=1 GPU_MEM_UTIL=0.85 MODEL=meta-llama/Llama-3.1-8B-Instruct`.

---

## Quick Start (Fresh Ubuntu GPU Container / RunPod)

```bash
git clone https://github.com/SamGanYX/radix-gate.git && cd radix-gate

# 1. System packages + Rust toolchain
apt-get update && apt-get install -y build-essential nginx curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Install vLLM + httpx (--ignore-installed PyJWT avoids Debian apt PyJWT conflict)
pip install --ignore-installed PyJWT vllm httpx

# 3. Run the 4-worker vLLM cluster benchmark (ports 8101-8104, NGINX :8095, RadixGate :8085)
chmod +x benchmarks/vllm_cluster/run_cluster.sh
./benchmarks/vllm_cluster/run_cluster.sh
```

For a multi-GPU node with an 8B model:
```bash
MULTI_GPU=1 GPU_MEM_UTIL=0.85 MODEL="meta-llama/Llama-3.1-8B-Instruct" ./benchmarks/vllm_cluster/run_cluster.sh
```

Once complete, copy the printed `### Real vLLM Cluster Results` Markdown table into the root `README.md`. Raw JSON metrics are saved to `vllm_benchmark_results.json`.
