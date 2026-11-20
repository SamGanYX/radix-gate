#!/usr/bin/env bash
set -euo pipefail

# Turnkey script to launch real vLLM workers (--enable-prefix-caching),
# an NGINX Round-Robin reverse proxy, and RadixGate,
# then execute the multi-turn TTFT benchmark.

if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

# Auto-detect GPU topology if NUM_WORKERS / GPU_MEM_UTIL are not explicitly set
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 32000)

if [ -z "${NUM_WORKERS:-}" ]; then
  if [ "$GPU_COUNT" -ge 4 ]; then
    NUM_WORKERS=4
    MULTI_GPU="${MULTI_GPU:-1}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
  elif [ "$VRAM_MIB" -ge 45000 ]; then
    NUM_WORKERS=4
    MULTI_GPU="${MULTI_GPU:-0}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.22}"
  else
    NUM_WORKERS=2
    MULTI_GPU="${MULTI_GPU:-0}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
  fi
fi

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MULTI_GPU="${MULTI_GPU:-0}"
BASE_PORT="${BASE_PORT:-8100}"       # Workers bind BASE_PORT+1 .. BASE_PORT+NUM_WORKERS
RADIX_PORT="${RADIX_PORT:-8085}"
NGINX_PORT="${NGINX_PORT:-8095}"

echo "[0/5] Detected ${GPU_COUNT} GPU(s) (${VRAM_MIB} MiB VRAM). Using NUM_WORKERS=${NUM_WORKERS}, GPU_MEM_UTIL=${GPU_MEM_UTIL}, MULTI_GPU=${MULTI_GPU}."

# Guard against broken torchaudio ABI in container images when serving text models
if ! python3 -c "import transformers.audio_utils" >/dev/null 2>&1; then
  echo "[preflight] Removing incompatible torchaudio installation for text-only serving..."
  pip uninstall -y torchaudio >/dev/null 2>&1 || true
fi

# Guard against vllm-0.34 + torch-2.14 minimax_m3_msa_warmup inductor crash
python3 - <<'PY' || true
import importlib.util
spec = importlib.util.find_spec("vllm.model_executor.warmup.kernel_warmup")
if spec and spec.origin:
    s = open(spec.origin, "r", encoding="utf-8").read()
    target = "from vllm.model_executor.warmup.minimax_m3_msa_warmup import (\n            minimax_m3_msa_warmup,\n        )"
    if target in s:
        s = s.replace(target, "pass").replace("minimax_m3_msa_warmup(worker)", "pass")
        open(spec.origin, "w", encoding="utf-8").write(s)
        print("[preflight] Patched vllm kernel_warmup.py to skip unused minimax_m3_msa_warmup")
PY

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
