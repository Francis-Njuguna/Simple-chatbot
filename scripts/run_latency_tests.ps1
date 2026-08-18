# Both outstanding latency measurements, in the order that keeps them clean.
# Run:  .\scripts\run_latency_tests.ps1
#
#   TEST 1  one chat request against the running server, then the [timing:chat]
#           line it produced. This is the measurement RERANK_LATENCY.md calls
#           decisive: every fast reading of the reranker so far was taken
#           in-process, every slow one inside uvicorn, and nothing has yet
#           measured the same work in the server on a quiet box.
#   TEST 2  time-to-first-visible-token for the current gateway vs the two free
#           candidates.
#
# Sequential on purpose. Run them together and the probe's three concurrent HTTP
# clients contend with the server for four cores, which is exactly the confound
# being investigated.
#
# ASCII ONLY in this file. Windows PowerShell 5.1 reads a BOM-less script as
# Windows-1252, where the last byte of a UTF-8 em dash decodes to a right double
# quote -- a character the parser treats as a string delimiter. One em dash in a
# comment is enough to break the whole script.
param([switch]$SkipProbe)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$log = ".\logs\app.log"
$question = "How do I download and install SMOWL for my exam?"

# ---------------------------------------------------------------- TEST 1
# The port is discovered rather than assumed: 8001 is also CHROMA_PORT in .env,
# and a stale server from an earlier run may still be holding 8000.
#
# /health is deliberately NOT under api_prefix (main.py:420 mounts it at the
# root, while the chat router gets /api/v1). Probing /api/v1/health 404s, and a
# 404 makes Invoke-WebRequest throw, which reads here as "server down" when the
# server is in fact up. That misdiagnosis already happened once.
$port = $null
foreach ($p in 8001, 8000) {
    try {
        $h = Invoke-WebRequest -Uri "http://127.0.0.1:$p/health" -TimeoutSec 5 -UseBasicParsing
        Write-Host "port $p  health $($h.StatusCode)"
        if ($null -eq $port) { $port = $p }
    } catch {
        Write-Host "port $p  no answer"
    }
}

if ($null -eq $port) {
    Write-Host ""
    Write-Host "no server answered /health on 8001 or 8000 - skipping TEST 1"
    Write-Host ""
} else {
    # Byte offset, not line count: the marker for "lines this request produced"
    # has to survive the log growing between here and the read below.
    $before = 0
    if (Test-Path $log) { $before = (Get-Item $log).Length }

    Write-Host ""
    Write-Host "TEST 1  POST /api/v1/chat on port $port (up to ~50s if the gateway retries)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $status = "error"
    try {
        $body = @{ message = $question } | ConvertTo-Json -Compress
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/v1/chat" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 180 -UseBasicParsing
        $status = $r.StatusCode
    } catch {
        Write-Host "  request failed: $($_.Exception.Message)"
    }
    $sw.Stop()
    Write-Host "  http=$status wall=$([math]::Round($sw.Elapsed.TotalSeconds, 2))s"

    # The number that matters is the server's own stage attribution, not the wall
    # clock: wall includes the client round trip, [timing:chat] does not.
    $grew = $false
    if (Test-Path $log) {
        if ((Get-Item $log).Length -gt $before) { $grew = $true }
    }
    if ($grew) {
        $fresh = Get-Content $log -Tail 400 | Select-String -Pattern "timing:chat|timing:retrieval" | Select-Object -Last 4
        if ($null -eq $fresh) {
            Write-Host "  the log grew, but not with timing lines"
        } else {
            Write-Host ""
            foreach ($line in $fresh) { Write-Host "  $($line.Line)" }
        }
    } else {
        Write-Host "  $log did not grow - this server logs elsewhere (check its LOG_FILE / console)"
    }
    Write-Host ""
    Write-Host "  read it as: retrieval ~800ms  => uvicorn was never the problem"
    Write-Host "              retrieval seconds => the uvicorn context is the cause"
    Write-Host ""
}

# ---------------------------------------------------------------- TEST 2
# Skippable because TEST 1 is the one that needs re-running after a server
# restart, and there is no reason to spend nine more API calls to learn the same
# thing twice: .\scripts\run_latency_tests.ps1 -SkipProbe
if ($SkipProbe) {
    Write-Host "TEST 2 skipped (-SkipProbe)"
    return
}

# The NVIDIA NIM key sits in a commented-out block in .env, so pydantic never
# loads it. Reading it here puts it in this process's environment only: off the
# command line, out of the log, .env untouched, server's provider unchanged.
$hit = Select-String -Path .env -Pattern 'nvapi-[A-Za-z0-9_-]+' | Select-Object -First 1
if ($null -eq $hit) {
    Write-Host "no nvapi- key in .env - NIM will report 'not built' and the others still run"
} else {
    $env:OPENAI_API_KEY = $hit.Matches[0].Value
    $env:OPENAI_API_BASE = "https://integrate.api.nvidia.com/v1"
    Write-Host "NIM key loaded (length $($env:OPENAI_API_KEY.Length)) -> $($env:OPENAI_API_BASE)"
}

# Round-robin, 3 trials each; a provider whose key is dead is skipped with its
# error shown rather than aborting the run. Baseline to beat: opus-5 at 6.96s to
# first visible token, 14.45s full.
Write-Host ""
Write-Host "TEST 2  provider probe (several minutes - opus-5 alone is ~15s per trial)"
Write-Host ""
& .\.venv\Scripts\python.exe -u scripts\probe_llm_latency.py --trials 3 --models claude-opus-5 gemini:gemini-2.0-flash openai:meta/llama-3.1-8b-instruct | Tee-Object -FilePath logs\probe_providers.log
