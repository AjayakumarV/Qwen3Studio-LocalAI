param(
    [ValidateSet(4, 5, 6, 8)]
    [int]$ActiveExperts = 4,
    [switch]$SpeculativeDraft,
    [int]$CpuMoeLayers = 41,
    [int]$ContextSize = 8192,
    [int]$Port = 8080,
    [int]$Threads = 8,
    [int]$BatchThreads = 16,
    [string]$KvCacheType = "q8_0"
)

$ErrorActionPreference = "Stop"

$BaseDir   = if ($env:LOCALAI_HOME) { $env:LOCALAI_HOME } else { Join-Path $HOME "LocalAI" }
$ServerExe = Join-Path $BaseDir "bin\llama-server.exe"
$ModelPath = Join-Path $BaseDir "models\Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
$DraftPath = Join-Path $BaseDir "models\Qwen3-0.6B-UD-Q4_K_XL.gguf"

if (-not (Test-Path $ServerExe)) {
    Write-Error "llama-server.exe not found at $ServerExe. Run 'python setup_localai.py' first."
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found at $ModelPath. Run 'python setup_localai.py' first."
}

$ArgsList = @(
    "-m", $ModelPath,
    "--alias", "Qwen3-30B-A3B",
    "--load-mode", "none",
    "-ngl", "99",
    "-ncmoe", "$CpuMoeLayers",
    "-fa", "on",
    "-ctk", $KvCacheType,
    "-ctv", $KvCacheType,
    "-np", "1",
    "--keep", "-1",
    "--cache-reuse", "256",
    "--no-webui",
    "-c", "$ContextSize",
    "-ub", "1024",
    "-t", "$Threads",
    "-tb", "$BatchThreads",
    "-Cr", "0-15",
    "--prio", "2",
    "--prio-batch", "2",
    "--poll", "100",
    "--override-kv", "qwen3moe.expert_used_count=int:$ActiveExperts",
    "--temp", "0.6",
    "--top-p", "0.95",
    "--top-k", "20",
    "--min-p", "0.06",
    "--repeat-penalty", "1.1",
    "--presence-penalty", "0.4",
    "--host", "127.0.0.1",
    "--port", "$Port"
)

if (($SpeculativeDraft -or $ActiveExperts -ge 6) -and (Test-Path $DraftPath)) {
    Write-Host "[Speculative Decoding Enabled] Pairing Qwen3-0.6B Draft Model (n-max=3, p-min=0.80)..." -ForegroundColor Cyan
    $ArgsList += @(
        "-md", $DraftPath,
        "-ngld", "99",
        "--spec-type", "draft-simple",
        "--spec-draft-n-max", "3",
        "--spec-draft-p-min", "0.80"
    )
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host " Starting Qwen3-30B-A3B Selective-Weight Turbo Engine (~39 tok/s)" -ForegroundColor Green
Write-Host " - Active Weights: Top-$ActiveExperts Experts/Token selected dynamically by Router" -ForegroundColor Cyan
Write-Host " - GPU VRAM:       100% Attention + Routers + Output + $(48 - $CpuMoeLayers) MoE Layers" -ForegroundColor Gray
Write-Host " - System RAM:     --load-mode none (Aligned SIMD buffers, zero disk paging)" -ForegroundColor Gray
Write-Host " - CPU Affinity:   Locked to Performance Cores (0-15) at High Priority" -ForegroundColor Gray
Write-Host "================================================================" -ForegroundColor Green

& $ServerExe @ArgsList
