#!/usr/bin/env bash
set -euo pipefail

# Turnkey script to launch 4 real vLLM workers (--enable-prefix-caching),
# an NGINX Round-Robin reverse proxy, and RadixGate,
# then execute the multi-turn TTFT benchmark.

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.23}" # 0.23 * 4 = 0.92 on a 24GB-32GB GPU; set 0.85 if using 1 GPU per worker
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MULTI_GPU="${MULTI_GPU:-0}"          # Set to 1 if you have >= NUM_WORKERS physical GPUs
BASE_PORT="${BASE_PORT:-8100}"       # Workers bind BASE_PORT+1 .. BASE_PORT+NUM_WORKERS (avoids :8000/:8001 conflicts)
RADIX_PORT="${RADIX_PORT:-8085}"
NGINX_PORT="${NGINX_PORT:-8095}"

PIDS=()
cleanup() {
  echo "[cleanup] Stopping cluster processes..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "[1/5] Building RadixGate release binary..."
cargo build --release

echo "[2/5] Pre-caching model weights ($MODEL)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

WORKER_URLS=()
WORKER_PORTS=()
for i in $(seq 1 "$NUM_WORKERS"); do
  PORT=$((BASE_PORT + i))
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
    --max-num-seqs "$MAX_NUM_SEQS" \
    --disable-log-requests \
    > "/tmp/vllm_${PORT}.log" 2>&1 &
  PIDS+=($!)
  WORKER_URLS+=("http://127.0.0.1:${PORT}")
  WORKER_PORTS+=("$PORT")
done

echo "[3/5] Waiting for all $NUM_WORKERS vLLM workers to become healthy..."
for idx in "${!WORKER_URLS[@]}"; do
  url="${WORKER_URLS[$idx]}"
  pid="${PIDS[$idx]}"
  port="${WORKER_PORTS[$idx]}"
  waited=0
  until curl -sf "${url}/health" > /dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] vLLM worker on port $port exited unexpectedly! Last 35 lines of /tmp/vllm_${port}.log:"
      tail -n 35 "/tmp/vllm_${port}.log"
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
    if [ $((waited % 10)) -eq 0 ]; then
      echo "  Still loading vLLM worker on :$port (${waited}s elapsed)..."
    fi
  done
  echo "  Healthy: $url"
done

echo "[4/5] Starting NGINX Round-Robin (:$NGINX_PORT) and RadixGate (:$RADIX_PORT)..."
cat > /tmp/radix_nginx.conf <<EOF
worker_processes auto;
pid /tmp/radix_nginx.pid;
events { worker_connections 1024; }
http {
  upstream vllm_rr {
$(for port in "${WORKER_PORTS[@]}"; do echo "    server 127.0.0.1:${port};"; done)
  }
  server {
    listen ${NGINX_PORT};
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
./target/release/radix-gate serve --host 127.0.0.1 --port "$RADIX_PORT" --upstreams "$UPSTREAMS_CSV" \
  > /tmp/radixgate.log 2>&1 &
PIDS+=($!)

until curl -sf "http://127.0.0.1:${RADIX_PORT}/health" > /dev/null; do
  sleep 1
done

echo "[5/5] Running empirical multi-turn TTFT benchmark..."
python3 -u benchmarks/vllm_cluster/benchmark_ttft.py \
  --nginx-url "http://127.0.0.1:${NGINX_PORT}" \
  --radix-url "http://127.0.0.1:${RADIX_PORT}" \
  --workers "$UPSTREAMS_CSV" \
  --model "$MODEL"
