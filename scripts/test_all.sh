#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/4] Rust Formatting (cargo fmt --check) ==="
cargo fmt --check

echo ""
echo "=== [2/4] Rust Linter (cargo clippy --all-targets -- -D warnings) ==="
cargo clippy --all-targets -- -D warnings

echo ""
echo "=== [3/4] Rust Unit & E2E Tests (cargo test --all-targets) ==="
cargo test --all-targets

echo ""
echo "=== [4/4] Python SDK Unit Tests ==="
PYTHONPATH="${REPO_ROOT}/sdk/python" python3 -m unittest discover -s "${REPO_ROOT}/sdk/python/tests" -p "test_*.py" -v

echo ""
echo "ALL CHECKS PASSED!"
