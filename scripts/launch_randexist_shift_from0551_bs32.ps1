param(
    [string]$RepoRoot = "C:\pycharm\pointdiff_new",
    [string]$Python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe"
)

$ErrorActionPreference = "Stop"

$tag = "train_randexist_shift_v1_from0551_bs32"
$config = Join-Path $RepoRoot "config\train_density_only_prior_t50_conf_add64_localcomp_r3_randexist_shift_v1_from_epoch0551_to0651_bs32.yaml"
$pidFile = Join-Path $RepoRoot "logs\$($tag).pid"
$status = Join-Path $RepoRoot "logs\$($tag)_status.txt"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $RepoRoot "logs\$($tag)_$($ts)_stdout.log"
$stderr = Join-Path $RepoRoot "logs\$($tag)_$($ts)_stderr.log"
$outDir = "D:\output\density_only_prior_t50_conf_add64_localcomp_r3_randexist_shift_v1_from_epoch0551_to0651_bs32_ddim5_nonms"
$ckpt = "D:\output\density_only_prior_t50_conf_add64_localcomp_r3_groupassign_pos32_v1_from_epoch0550_to0650_bs32_ddim5_nonms\last_epoch0551.pth"

if (!(Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}
if (!(Test-Path -LiteralPath $config)) {
    throw "Config not found: $config"
}
if (!(Test-Path -LiteralPath $ckpt)) {
    throw "Resume checkpoint not found: $ckpt"
}
if (Test-Path -LiteralPath $pidFile) {
    $oldPid = (Get-Content -LiteralPath $pidFile | Select-Object -First 1)
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        throw "A run for tag '$tag' is already alive (pid=$oldPid). Stop it first."
    }
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
    "ckpt_path=$ckpt",
    "target_epochs=0552-0651",
    "experiment=EXP-B: lambda_exist 40->10, lambda_rand_exist 8->24 (weight moved to inference-distribution branch); exist_duplicate_weight back to 1.0",
    "baseline=groupassign_pos32 from same 0551 start: recall 0.4393 mae 105.4 valMAE 95.34; EXP-A peaked 0.4557 then decayed to 0.4347",
    "watch=recall should rise without the EXP-A decay; dup@6 and precision are the risks"
) | Set-Content -LiteralPath $status

Write-Output "started pid=$($p.Id)"
Write-Output "config=$config"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"
Write-Output "status=$status"
