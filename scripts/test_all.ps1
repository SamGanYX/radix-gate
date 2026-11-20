$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot

try {
    Write-Host "=== [1/4] Rust Formatting (cargo fmt --check) ===" -ForegroundColor Cyan
    cargo fmt --check

    Write-Host "`n=== [2/4] Rust Linter (cargo clippy --all-targets -- -D warnings) ===" -ForegroundColor Cyan
    cargo clippy --all-targets -- -D warnings

    Write-Host "`n=== [3/4] Rust Unit & E2E Tests (cargo test --all-targets) ===" -ForegroundColor Cyan
    cargo test --all-targets

    Write-Host "`n=== [4/4] Python SDK Unit Tests ===" -ForegroundColor Cyan
    $env:PYTHONPATH = Join-Path $RepoRoot "sdk\python"
    python -m unittest discover -s (Join-Path $RepoRoot "sdk\python\tests") -p "test_*.py" -v

    Write-Host "`nALL CHECKS PASSED!" -ForegroundColor Green
}
finally {
    Pop-Location
}
