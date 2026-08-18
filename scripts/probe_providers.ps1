# Head-to-head time-to-first-visible-token: the current gateway vs the two free
# candidates. Run from anywhere:  .\scripts\probe_providers.ps1
#
# Why a wrapper instead of just calling the probe: the NVIDIA NIM key lives in a
# commented-out block in .env, so pydantic never loads it. Reading it here puts
# it in the environment for this process only — off the command line, out of the
# log, and with .env unmodified, so the running server keeps its own provider.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$hit = Select-String -Path .env -Pattern 'nvapi-[A-Za-z0-9_-]+' | Select-Object -First 1
if ($null -eq $hit) {
    Write-Host "no nvapi- key in .env — NIM will report 'not built' and the others still run"
} else {
    $env:OPENAI_API_KEY = $hit.Matches[0].Value
    $env:OPENAI_API_BASE = "https://integrate.api.nvidia.com/v1"
    Write-Host "NIM key loaded (length $($env:OPENAI_API_KEY.Length)) -> $($env:OPENAI_API_BASE)"
}

# Round-robin, 3 trials each. A provider whose key is dead is skipped with its
# error printed rather than aborting the run — the point is to compare whatever
# is actually reachable. Baseline to beat: opus-5 at 6.96s first token / 14.45s full.
& .\.venv\Scripts\python.exe -u scripts\probe_llm_latency.py --trials 3 --models claude-opus-5 gemini:gemini-2.0-flash openai:meta/llama-3.1-8b-instruct | Tee-Object -FilePath logs\probe_providers.log
