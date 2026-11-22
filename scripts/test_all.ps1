$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nFAILED at: $Label (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot

try {
    Invoke-Checked "[1/4] Rust Formatting (cargo fmt --check)" { cargo fmt --check }
    Invoke-Checked "[2/4] Rust Linter (cargo clippy --all-targets -- -D warnings)" { cargo clippy --all-targets -- -D warnings }
    Invoke-Checked "[3/4] Rust Unit & E2E Tests (cargo test --all-targets)" { cargo test --all-targets }
    Invoke-Checked "[4/4] Python SDK Unit Tests" {
        $env:PYTHONPATH = Join-Path $RepoRoot "sdk\python"
        python -m unittest discover -s (Join-Path $RepoRoot "sdk\python\tests") -p "test_*.py" -v
    }

    Write-Host "`nALL CHECKS PASSED!" -ForegroundColor Green
}
finally {
    Pop-Location
}
