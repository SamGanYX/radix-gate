# Real vLLM Cluster Benchmark: NGINX Round-Robin vs. RadixGate

This directory contains a turnkey benchmark harness to measure **real GPU Time-To-First-Token (TTFT)** and **actual vLLM PagedAttention prefix cache hit rates** comparing standard NGINX Round-Robin against `radix-gate`.

---

## Hardware Requirements

You can run 4 independent vLLM replicas on either:
- **Option A (Lowest Cost, ~$0.40/hr)**: 1x 24GB+ GPU (RTX 3090 / RTX 4090 / A10G / L4) running 4 `vLLM` processes with `--gpu-memory-utilization 0.20` and `Qwen/Qwen2.5-1.5B-Instruct`.
- **Option B (Multi-GPU)**: 4x GPUs with `MULTI_GPU=1 GPU_MEM_UTIL=0.85 MODEL=meta-llama/Llama-3.1-8B-Instruct`.

---

## Quick Start (Ubuntu Linux GPU Instance)

1. Install dependencies:
   ```bash
   pip install vllm httpx
   sudo apt-get update && sudo apt-get install -y nginx curl
   ```

2. Run the full cluster benchmark (starts 4 vLLM servers with `--enable-prefix-caching`, NGINX on `:8090`, RadixGate on `:8080`, and runs the streaming multi-turn workload):
   ```bash
   chmod +x benchmarks/vllm_cluster/run_cluster.sh
   ./benchmarks/vllm_cluster/run_cluster.sh
   ```

   For a multi-GPU node with an 8B model:
   ```bash
   MULTI_GPU=1 GPU_MEM_UTIL=0.85 MODEL="meta-llama/Llama-3.1-8B-Instruct" ./benchmarks/vllm_cluster/run_cluster.sh
   ```

3. Paste the printed Markdown table (`### Real vLLM Cluster Results`) directly into the root `README.md`. Raw JSON metrics are saved to `vllm_benchmark_results.json`.
