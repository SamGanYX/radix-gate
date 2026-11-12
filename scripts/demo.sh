#!/usr/bin/env bash
# RadixGate Showcase & End-to-End Verification Script (POSIX Bash)
# Usage: ./scripts/demo.sh

set -euo pipefail

CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m'

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}      RadixGate: Enterprise LLM Inference Router Showcase        ${NC}"
echo -e "${CYAN}=================================================================${NC}\n"

# 1. Build optimized release binary
echo -e "${YELLOW}[1/6] Building release binaries...${NC}"
cargo build --release
echo -e "${GREEN}Build complete.${NC}\n"

# 2. Run comprehensive test suite
echo -e "${YELLOW}[2/6] Executing unit, integration, and E2E cluster test suite...${NC}"
cargo test --all-targets
echo -e "${GREEN}All tests passed with zero errors.${NC}\n"

# 3. Run comparative routing benchmark
echo -e "${YELLOW}[3/6] Running empirical multi-strategy benchmark (Round-Robin vs Least-Loaded vs RadixGate)...${NC}"
./target/release/radix-gate bench --requests 600 --workers 4
echo ""

# 4. Launch RadixGate server in background
echo -e "${YELLOW}[4/6] Starting RadixGate Gateway daemon on 127.0.0.1:8080...${NC}"
./target/release/radix-gate serve --port 8080 --mock-workers 4 &
SERVER_PID=$!

cleanup() {
    echo -e "${GRAY}Shutting down gateway daemon (PID: ${SERVER_PID})...${NC}"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for gateway to become ready
READY=0
for i in {1..30}; do
    if curl -s http://127.0.0.1:8080/health | grep -q '"status":"ok"'; then
        READY=1
        break
    fi
    sleep 0.2
done

if [ "$READY" -ne 1 ]; then
    echo "Gateway server failed to start within timeout"
    exit 1
fi
echo -e "${GREEN}Gateway healthy and listening at http://127.0.0.1:8080${NC}\n"

# 5. Execute high-concurrency socket load generator
echo -e "${YELLOW}[5/6] Injecting synthetic multi-turn multi-agent traffic...${NC}"
./target/release/radix-gate load-test --url "http://127.0.0.1:8080" --concurrency 8 --requests 40 --turns 3
echo ""

# 6. Query prefix tree state and Prometheus metrics
echo -e "${YELLOW}[6/6] Inspecting live Radix Tree prefix topology and OpenMetrics...${NC}"
echo -e "${MAGENTA}--- Radix Tree Topology ---${NC}"
./target/release/radix-gate tree --url "http://127.0.0.1:8080"
echo ""

echo -e "${MAGENTA}--- OpenMetrics Prometheus Telemetry Sample ---${NC}"
curl -s http://127.0.0.1:8080/metrics | grep -E '^radixgate_' | head -n 12
echo ""

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}       Showcase completed successfully! All systems nominal.     ${NC}"
echo -e "${CYAN}  See DEMO.md for interviewer walkthrough and cluster recipes.   ${NC}"
echo -e "${CYAN}=================================================================${NC}"
