param(
    [Parameter(Mandatory = $true)]
    [int]$Phase1Pid,
    [string]$RepoRoot = "C:\pycharm\pointdiff_new",
    [string]$Python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe",
    [string]$Phase1Checkpoint = "D:\output\density_only_prior_t50_conf_active_refine_from_epoch0100_to0120_bs48_gate05_nodetach\last_epoch0120.pth",
    [string]$Phase2Config = "C:\pycharm\pointdiff_new\config\train_density_only_prior_t50_conf_point_refine_from_epoch0120_to0150_bs48.yaml",
    [string]$Phase2Tag = "train_t50_conf_point_refine_from_epoch0120_to0150_bs48"
)

$ErrorActionPreference = "Stop"
$logDir = Join-Path $RepoRoot "logs"
$status = Join-Path $logDir "$($Phase2Tag)_watcher_status.txt"
$stdout = Join-Path $logDir "$($Phase2Tag)_stdout.log"
$stderr = Join-Path $logDir "$($Phase2Tag)_stderr.log"
$pidFile = Join-Path $logDir "$($Phase2Tag).pid"

function Write-Status([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $status -Value "$stamp $Message"
}

Write-Status "watcher_start phase1_pid=$Phase1Pid phase2_config=$Phase2Config"

try {
    $phase1 = [System.Diagnostics.Process]::GetProcessById($Phase1Pid)
    $phase1.WaitForExit()
    $exitCode = $phase1.ExitCode
    $exitCodeText = if ($null -eq $exitCode) { "unknown" } else { "$exitCode" }
    Write-Status "phase1_exit code=$exitCodeText"
} catch {
    Write-Status "phase1_wait_failed error=$($_.Exception.Message)"
    exit 1
}

if ($null -ne $exitCode -and $exitCode -ne 0) {
    Write-Status "not_starting_phase2 reason=phase1_nonzero_exit"
    exit $exitCode
}

if (-not (Test-Path -LiteralPath $Phase1Checkpoint)) {
    Write-Status "not_starting_phase2 reason=missing_checkpoint checkpoint=$Phase1Checkpoint"
    exit 2
}

$argsList = @("-u", "main.py", "--config", $Phase2Config)
$p = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Set-Content -LiteralPath $pidFile -Value $p.Id
Write-Status "phase2_started pid=$($p.Id) stdout=$stdout stderr=$stderr"
