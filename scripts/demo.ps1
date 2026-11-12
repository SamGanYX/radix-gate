# RadixGate Showcase & End-to-End Verification Script (PowerShell)
# Usage: .\scripts\demo.ps1

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "      RadixGate: Enterprise LLM Inference Router Showcase        " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Build optimized release binary
Write-Host "[1/6] Building release binaries..." -ForegroundColor Yellow
cargo build --release
if ($LASTEXITCODE -ne 0) { throw "Cargo build failed" }
Write-Host "Build complete." -ForegroundColor Green
Write-Host ""

# 2. Run comprehensive test suite
Write-Host "[2/6] Executing unit, integration, and E2E cluster test suite..." -ForegroundColor Yellow
cargo test --all-targets
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
Write-Host "All tests passed with zero errors." -ForegroundColor Green
Write-Host ""

# 3. Run comparative routing benchmark
Write-Host "[3/6] Running empirical multi-strategy benchmark (Round-Robin vs Least-Loaded vs RadixGate)..." -ForegroundColor Yellow
& "target\release\radix-gate.exe" bench --requests 600 --workers 4
Write-Host ""

# 4. Launch RadixGate server in background
Write-Host "[4/6] Starting RadixGate Gateway daemon on 127.0.0.1:8080..." -ForegroundColor Yellow
$serverProcess = Start-Process -FilePath "target\release\radix-gate.exe" -ArgumentList "serve", "--port", "8080", "--mock-workers", "4" -PassThru -NoNewWindow

try {
    # Wait for gateway to become ready
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 200
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -TimeoutSec 1
            if ($resp.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {}
    }

    if (-not $ready) {
        throw "Gateway server failed to start within timeout"
    }
    Write-Host "Gateway healthy and listening at http://127.0.0.1:8080" -ForegroundColor Green
    Write-Host ""

    # 5. Execute high-concurrency socket load generator
    Write-Host "[5/6] Injecting synthetic multi-turn multi-agent traffic..." -ForegroundColor Yellow
    & "target\release\radix-gate.exe" load-test --url "http://127.0.0.1:8080" --concurrency 8 --requests 40 --turns 3
    Write-Host ""

    # 6. Query prefix tree state and Prometheus metrics
    Write-Host "[6/6] Inspecting live Radix Tree prefix topology and OpenMetrics..." -ForegroundColor Yellow
    Write-Host "--- Radix Tree Topology ---" -ForegroundColor Magenta
    & "target\release\radix-gate.exe" tree --url "http://127.0.0.1:8080"
    Write-Host ""

    Write-Host "--- OpenMetrics Prometheus Telemetry Sample ---" -ForegroundColor Magenta
    $metrics = (Invoke-RestMethod -Uri "http://127.0.0.1:8080/metrics" -TimeoutSec 2)
    $metricLines = $metrics -split "`n" | Where-Object { $_ -match "^radixgate_" } | Select-Object -First 12
    $metricLines | ForEach-Object { Write-Host $_ }
    Write-Host ""

} finally {
    Write-Host "Shutting down gateway daemon (PID: $($serverProcess.Id))..." -ForegroundColor Gray
    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "       Showcase completed successfully! All systems nominal.     " -ForegroundColor Cyan
Write-Host "  See DEMO.md for interviewer walkthrough and cluster recipes.   " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
