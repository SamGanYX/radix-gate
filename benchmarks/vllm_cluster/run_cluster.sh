#!/usr/bin/env bash
set -euo pipefail

# Turnkey script to launch 4 real vLLM workers (--enable-prefix-caching),
# an NGINX Round-Robin reverse proxy (:8090), and RadixGate (:8080),
# then execute the multi-turn TTFT benchmark.

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.20}" # 0.20 * 4 = 0.80 on a single 24GB+ GPU; set 0.85 if using 1 GPU per worker
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MULTI_GPU="${MULTI_GPU:-0}"          # Set to 1 if you have >= NUM_WORKERS physical GPUs

PIDS=()
cleanup() {
  echo "[cleanup] Stopping cluster processes..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "[1/4] Building RadixGate release binary..."
cargo build --release

WORKER_URLS=()
for i in $(seq 1 "$NUM_WORKERS"); do
  PORT=$((8000 + i))
  GPU_ID=0
  if [ "$MULTI_GPU" -eq 1 ]; then
    GPU_ID=$((i - 1))
  fi
  echo "  Starting vLLM worker-$i on port $PORT (CUDA_VISIBLE_DEVICES=$GPU_ID, gpu-memory-utilization=$GPU_MEM_UTIL)..."
  TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-log-requests \
    > "/tmp/vllm_${PORT}.log" 2>&1 &
  PIDS+=($!)
  WORKER_URLS+=("http://127.0.0.1:${PORT}")
done

echo "[2/4] Waiting for all $NUM_WORKERS vLLM workers to become healthy..."
for idx in "${!WORKER_URLS[@]}"; do
  url="${WORKER_URLS[$idx]}"
  pid="${PIDS[$idx]}"
  port=$((8001 + idx))
  until curl -sf "${url}/health" > /dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] vLLM worker on port $port exited unexpectedly! Last 30 lines of /tmp/vllm_${port}.log:"
      tail -n 30 "/tmp/vllm_${port}.log"
      exit 1
    fi
    sleep 2
  done
  echo "  Healthy: $url"
done

echo "[3/4] Starting NGINX Round-Robin (:8090) and RadixGate (:8080)..."
cat > /tmp/radix_nginx.conf <<EOF
worker_processes auto;
pid /tmp/radix_nginx.pid;
events { worker_connections 1024; }
http {
  upstream vllm_rr {
$(for i in $(seq 1 "$NUM_WORKERS"); do echo "    server 127.0.0.1:$((8000 + i));"; done)
  }
  server {
    listen 8090;
    location / {
      proxy_pass http://vllm_rr;
      proxy_buffering off;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
    }
  }
}
EOF
nginx -c /tmp/radix_nginx.conf
PIDS+=($(cat /tmp/radix_nginx.pid))

UPSTREAMS_CSV=$(IFS=,; echo "${WORKER_URLS[*]}")
./target/release/radix-gate serve --host 127.0.0.1 --port 8080 --upstreams "$UPSTREAMS_CSV" \
  > /tmp/radixgate.log 2>&1 &
PIDS+=($!)

until curl -sf "http://127.0.0.1:8080/health" > /dev/null; do
  sleep 1
done

echo "[4/4] Running empirical multi-turn TTFT benchmark..."
python3 benchmarks/vllm_cluster/benchmark_ttft.py \
  --nginx-url "http://127.0.0.1:8090" \
  --radix-url "http://127.0.0.1:8080" \
  --workers "$UPSTREAMS_CSV" \
  --model "$MODEL"
