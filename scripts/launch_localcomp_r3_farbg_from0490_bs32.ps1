param(
    [string]$RepoRoot = "C:\pycharm\pointdiff_new",
    [string]$Python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe"
)

$ErrorActionPreference = "Stop"

$tag = "train_localcomp_r3_farbg_v1_from0490_bs32"
$config = Join-Path $RepoRoot "config\train_density_only_prior_t50_conf_add64_localcomp_r3_farbg_v1_from_epoch0490_to0590_bs32.yaml"
$pidFile = Join-Path $RepoRoot "logs\$($tag).pid"
$status = Join-Path $RepoRoot "logs\$($tag)_status.txt"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $RepoRoot "logs\$($tag)_$($ts)_stdout.log"
$stderr = Join-Path $RepoRoot "logs\$($tag)_$($ts)_stderr.log"
$outDir = "D:\output\density_only_prior_t50_conf_add64_localcomp_r3_farbg_v1_from_epoch0490_to0590_bs32_ddim5_nonms"

if (!(Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}
if (!(Test-Path -LiteralPath $config)) {
    throw "Config not found: $config"
}

$argsList = @("-u", "main.py", "--config", $config)
$p = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Set-Content -LiteralPath $pidFile -Value $p.Id
@(
    "state=running",
    "started_at=$((Get-Date).ToString('o'))",
    "python_pid=$($p.Id)",
    "python=$Python",
    "config=$config",
    "stdout=$stdout",
    "stderr=$stderr",
    "out_dir=$outDir",
    "ckpt_path=D:\output\density_only_prior_t50_conf_add64_localcomp_r3_v1_from_epoch0440_to0490_bs32_ddim5_nonms\last_epoch0490.pth",
    "target_epochs=0491-0590"
) | Set-Content -LiteralPath $status

Write-Output "started pid=$($p.Id)"
Write-Output "config=$config"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"
Write-Output "status=$status"
