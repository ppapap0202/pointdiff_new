param(
    [int]$CurrentPid = 0,
    [string]$Python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe",
    [string]$RepoRoot = "C:\pycharm\pointdiff_new"
)

$ErrorActionPreference = "Stop"

$tag = "train_t50_conf_interaction_from_epoch0130_to0230_bs16"
$config = "config\train_density_only_prior_t50_conf_interaction_from_epoch0130_to0230_bs16.yaml"
$readyCkpt = "D:\output\density_only_prior_t50_conf_interaction_from_epoch0100_to0130_bs16_detachgate_ddim5_nonms\last_epoch0130.pth"
$stdout = Join-Path $RepoRoot "logs\${tag}_stdout.log"
$stderr = Join-Path $RepoRoot "logs\${tag}_stderr.log"
$pidFile = Join-Path $RepoRoot "logs\${tag}.pid"
$watchLog = Join-Path $RepoRoot "logs\${tag}_watcher.log"

function Write-WatchLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $watchLog -Value "$stamp $Message"
}

Write-WatchLog "watcher started current_pid=$CurrentPid ready_ckpt=$readyCkpt"

if ($CurrentPid -gt 0) {
    $proc = Get-Process -Id $CurrentPid -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-WatchLog "waiting for pid=$CurrentPid to exit"
        Wait-Process -Id $CurrentPid
    }
}

for ($i = 0; $i -lt 60; $i++) {
    if (Test-Path -LiteralPath $readyCkpt) {
        break
    }
    Start-Sleep -Seconds 30
}

if (-not (Test-Path -LiteralPath $readyCkpt)) {
    Write-WatchLog "missing ready checkpoint; resume not started"
    exit 1
}

$argsList = @("main.py", "--config", $config)
$p = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Set-Content -Path $pidFile -Value $p.Id
Write-WatchLog "resume started pid=$($p.Id) config=$config"
