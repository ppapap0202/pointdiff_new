param(
    [int]$Stage1Pid = 33256,
    [string]$RepoRoot = "C:\pycharm\pointdiff_new",
    [string]$Python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe",
    [string]$Stage1OutDir = "D:\output\density_only_prior_from_0197_stage1",
    [string]$Stage1Stderr = "C:\pycharm\pointdiff_new\logs\train_density_only_prior_from0197_stage1_20260625_031242_stderr.log",
    [string]$Stage1Config = "C:\pycharm\pointdiff_new\config\train_density_only_prior_from_0197_stage1.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

$tag = "train_density_only_prior_stage2_from_best_candidate"
$watchLog = Join-Path $RepoRoot "logs\$($tag)_watcher.log"
$status = Join-Path $RepoRoot "logs\$($tag)_status.txt"
$pidFile = Join-Path $RepoRoot "logs\$($tag).pid"
$configOut = Join-Path $RepoRoot "config\train_density_only_prior_stage2_from_best_candidate.yaml"
$stage2OutDir = "D:\output\density_only_prior_stage2_from_best_candidate"
$stage2SaveDir = Join-Path $RepoRoot "vis_results\density_only_prior_stage2_from_best_candidate"

function Write-WatchLog([string]$Message) {
    $stamp = (Get-Date).ToString("o")
    Add-Content -LiteralPath $watchLog -Value "$stamp $Message"
}

Write-WatchLog "watching stage1 pid=$Stage1Pid"
while (Get-Process -Id $Stage1Pid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}
Write-WatchLog "stage1 process exited"

if (!(Test-Path -LiteralPath $Stage1Stderr)) {
    throw "Stage1 stderr log not found: $Stage1Stderr"
}

$bestEpoch = $null
$bestCover = -1.0
$pattern = "\[Epoch\s+(\d+)\].*val_ddim_candidate_cover@6=([0-9.]+)"
foreach ($line in Get-Content -LiteralPath $Stage1Stderr) {
    $m = [regex]::Match($line, $pattern)
    if ($m.Success) {
        $epoch = [int]$m.Groups[1].Value
        $cover = [double]$m.Groups[2].Value
        if ($cover -gt $bestCover) {
            $bestCover = $cover
            $bestEpoch = $epoch
        }
    }
}

if ($null -eq $bestEpoch) {
    throw "Could not find val_ddim_candidate_cover@6 in $Stage1Stderr"
}

$ckpt = Join-Path $Stage1OutDir ("last_epoch{0:D4}.pth" -f $bestEpoch)
if (!(Test-Path -LiteralPath $ckpt)) {
    throw "Best epoch checkpoint not found: $ckpt"
}

Write-WatchLog ("selected epoch={0:D4} val_ddim_candidate_cover@6={1:F4} ckpt={2}" -f $bestEpoch, $bestCover, $ckpt)

$cfg = Get-Content -LiteralPath $Stage1Config
$cfg = $cfg -replace "^batch_size:.*", "batch_size: 8"
$cfg = $cfg -replace "^epochs:.*", "epochs: 40"
$cfg = $cfg -replace "^out_dir:.*", "out_dir: '$stage2OutDir'"
$cfg = $cfg -replace "^freeze_base_for_prior:.*", "freeze_base_for_prior: False"
$cfg = $cfg -replace "^lambda_prior_occupancy:.*", "lambda_prior_occupancy: 0.0"
$cfg = $cfg -replace "^lambda_prior_density:.*", "lambda_prior_density: 1.0"
$cfg = $cfg -replace "^lambda_prior_count:.*", "lambda_prior_count: 0.3"
$cfg = $cfg -replace "^lr:.*", "lr: 0.000002"
$cfg = $cfg -replace "^lr_backbone:.*", "lr_backbone: 0.0000002"
$cfg = $cfg -replace "^rand_cover_t_max:.*", "rand_cover_t_max: 999"
$cfg = $cfg -replace "^resume_training:.*", "resume_training: False"
$cfg = $cfg -replace "^reset_optimizer:.*", "reset_optimizer: True"
$cfg = $cfg -replace "^ckpt_path:.*", "ckpt_path: '$ckpt'"
$cfg = $cfg -replace "^save_dir:.*", "save_dir: '$stage2SaveDir'"
$cfg | Set-Content -LiteralPath $configOut
Write-WatchLog "wrote config=$configOut"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $RepoRoot "logs\$($tag)_$($ts)_stdout.log"
$stderr = Join-Path $RepoRoot "logs\$($tag)_$($ts)_stderr.log"
$argsList = @("-u", "main.py", "--config", $configOut)
$p = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $pidFile -Value $p.Id
@(
    "state=running",
    "started_at=$((Get-Date).ToString('o'))",
    "python_pid=$($p.Id)",
    "python=$Python",
    "stdout=$stdout",
    "stderr=$stderr",
    "config=$configOut",
    "out_dir=$stage2OutDir",
    ("selected_stage1_epoch={0:D4}" -f $bestEpoch),
    ("selected_stage1_candidate_cover={0:F4}" -f $bestCover),
    "ckpt_path=$ckpt"
) | Set-Content -LiteralPath $status
Write-WatchLog "started stage2 pid=$($p.Id)"
