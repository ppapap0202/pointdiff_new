param(
    [int]$TrainPid = 15828,
    [string]$Repo = "C:\pycharm\pointdiff_new",
    [string]$Python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe",
    [string]$Config = "config\train_density_only_prior_cover_antidup_from_epoch0005_resume_to100_bs8_t500.yaml",
    [string]$OutDir = "D:\output\density_only_prior_cover_antidup_from_epoch0006_100ep_bs8_t500",
    [string]$RunName = "epoch0100_prior_sweep",
    [int[]]$StartTs = @(300, 100, 500, 50),
    [int[]]$StepsList = @(50, 30, 20),
    [int]$NumRealizations = 1,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 4,
    [int]$PollSeconds = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LogDir = Join-Path $Repo "logs"
$SweepRoot = Join-Path $Repo ("vis_results\" + $RunName)
$MonitorLog = Join-Path $LogDir ($RunName + "_monitor.log")
$DoneJson = Join-Path $SweepRoot "status.json"
$TargetCkpt = Join-Path $OutDir "last_epoch0100.pth"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null

function Write-MonitorLog {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $Message" | Tee-Object -FilePath $MonitorLog -Append | Out-Null
}

function Get-LatestCheckpointName {
    $latest = Get-ChildItem -Path $OutDir -Filter "last_epoch*.pth" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        return "<none>"
    }
    return $latest.Name
}

Write-MonitorLog "Waiting for epoch100 checkpoint: $TargetCkpt"
Write-MonitorLog "Sweep settings: start_ts=$($StartTs -join ',') steps=$($StepsList -join ',') realizations=$NumRealizations batch_size=$BatchSize num_workers=$NumWorkers"

while ($true) {
    $proc = Get-Process -Id $TrainPid -ErrorAction SilentlyContinue
    $hasTarget = Test-Path -LiteralPath $TargetCkpt

    if ($hasTarget -and $null -eq $proc) {
        Write-MonitorLog "Training process ended and epoch100 checkpoint exists. Starting diagnostics."
        break
    }

    if (-not $hasTarget -and $null -eq $proc) {
        $latestName = Get-LatestCheckpointName
        Write-MonitorLog "Training process ended before epoch100 checkpoint existed. Latest checkpoint: $latestName"
        [pscustomobject]@{
            status = "blocked"
            reason = "training_stopped_before_epoch100"
            latest_checkpoint = $latestName
            target_checkpoint = $TargetCkpt
            finished_at = (Get-Date).ToString("s")
        } | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 -Path $DoneJson
        exit 2
    }

    $latest = Get-LatestCheckpointName
    Write-MonitorLog "Still waiting. pid_alive=$($null -ne $proc) target_exists=$hasTarget latest=$latest"
    Start-Sleep -Seconds $PollSeconds
}

$rows = @()

foreach ($startT in $StartTs) {
    foreach ($steps in $StepsList) {
        $caseName = "t{0}_steps{1}_r{2}" -f $startT, $steps, $NumRealizations
        $saveDir = Join-Path $SweepRoot $caseName
        $stdoutLog = Join-Path $LogDir ("validate_" + $RunName + "_" + $caseName + ".log")
        $stderrLog = Join-Path $LogDir ("validate_" + $RunName + "_" + $caseName + "_stderr.log")

        New-Item -ItemType Directory -Force -Path $saveDir | Out-Null
        Write-MonitorLog "RUN $caseName"

        $validateArgs = @(
            "validate_diagnostics.py",
            "--config", $Config,
            "--ckpt_path", $TargetCkpt,
            "--proposal_prior_start_t", [string]$startT,
            "--ddim_steps", [string]$steps,
            "--num_realizations", [string]$NumRealizations,
            "--batch_size", [string]$BatchSize,
            "--num_workers", [string]$NumWorkers,
            "--save_dir", $saveDir
        )

        $procRun = Start-Process `
            -FilePath $Python `
            -ArgumentList $validateArgs `
            -WorkingDirectory $Repo `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -Wait `
            -PassThru `
            -NoNewWindow
        $exitCode = $procRun.ExitCode

        if ($exitCode -ne 0) {
            Write-MonitorLog "FAILED $caseName exit_code=$exitCode stdout=$stdoutLog stderr=$stderrLog"
            [pscustomobject]@{
                status = "failed"
                failed_case = $caseName
                exit_code = $exitCode
                stdout_log = $stdoutLog
                stderr_log = $stderrLog
                finished_at = (Get-Date).ToString("s")
            } | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 -Path $DoneJson
            exit $exitCode
        }

        $summaryPath = Join-Path $saveDir "summary.json"
        if (-not (Test-Path -LiteralPath $summaryPath)) {
            Write-MonitorLog "FAILED $caseName missing summary.json"
            [pscustomobject]@{
                status = "failed"
                failed_case = $caseName
                reason = "missing_summary_json"
                stdout_log = $stdoutLog
                stderr_log = $stderrLog
                finished_at = (Get-Date).ToString("s")
            } | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 -Path $DoneJson
            exit 3
        }

        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $row = [pscustomobject]@{
            start_t = $startT
            ddim_steps = $steps
            num_realizations = $NumRealizations
            mae = [double]$summary.mae
            rmse = [double]$summary.rmse
            proposal_cover = [double]$summary.proposal_cover_ratio_mean
            candidate_cover = [double]$summary.candidate_cover_ratio_mean
            final_cover = [double]$summary.final_cover_ratio_mean
            proposal_dup_per_gt = [double]$summary.proposal_dup_per_gt_mean
            candidate_dup_per_gt = [double]$summary.candidate_dup_per_gt_mean
            final_dup_per_gt = [double]$summary.final_dup_per_gt_mean
            mean_candidate_count_pre_nms = [double]$summary.mean_candidate_count_pre_nms
            mean_pred_count = [double]$summary.mean_pred_count
            save_dir = $saveDir
            stdout_log = $stdoutLog
            stderr_log = $stderrLog
        }
        $rows += $row

        $rows |
            Sort-Object -Property candidate_cover -Descending |
            Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $SweepRoot "sweep_summary.csv")

        Write-MonitorLog ("DONE {0} candidate_cover={1:N4} raw_cover={2:N4} mae={3:N2} cand_dup={4:N3}" -f `
            $caseName, $row.candidate_cover, $row.proposal_cover, $row.mae, $row.candidate_dup_per_gt)
    }
}

$best = $rows | Sort-Object -Property candidate_cover -Descending | Select-Object -First 1
[pscustomobject]@{
    status = "complete"
    target_checkpoint = $TargetCkpt
    run_name = $RunName
    sweep_root = $SweepRoot
    summary_csv = (Join-Path $SweepRoot "sweep_summary.csv")
    best_start_t = $best.start_t
    best_ddim_steps = $best.ddim_steps
    best_candidate_cover = $best.candidate_cover
    best_proposal_cover = $best.proposal_cover
    best_mae = $best.mae
    best_candidate_dup_per_gt = $best.candidate_dup_per_gt
    finished_at = (Get-Date).ToString("s")
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $DoneJson

Write-MonitorLog ("SWEEP COMPLETE best=t{0}/steps{1} candidate_cover={2:N4} raw_cover={3:N4} mae={4:N2} cand_dup={5:N3}" -f `
    $best.start_t, $best.ddim_steps, $best.candidate_cover, $best.proposal_cover, $best.mae, $best.candidate_dup_per_gt)
