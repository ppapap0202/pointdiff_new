param(
    [Parameter(Mandatory=$true)][string]$LogPath,
    [Parameter(Mandatory=$true)][int]$PidToStop,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$StatusPath,
    [Parameter(Mandatory=$true)][double]$BaselineRecall,
    [Parameter(Mandatory=$true)][int]$BaselineEpoch,
    [int]$Patience = 2,
    [double]$MinDelta = 0.000001,
    [int]$PollSeconds = 30,
    [int]$CheckpointTimeoutSeconds = 3600,
    [int]$CheckpointStableSeconds = 8
)

function Write-Status {
    param([string]$Message)
    $dir = Split-Path -Parent $StatusPath
    if ($dir -and !(Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -LiteralPath $StatusPath -Value "$timestamp $Message" -Encoding UTF8
}

function Wait-StableCheckpoint {
    param([int]$Epoch)
    $ckpt = Join-Path $OutDir ("last_epoch{0:D4}.pth" -f $Epoch)
    $deadline = (Get-Date).AddSeconds($CheckpointTimeoutSeconds)
    $lastSize = -1
    $stableSince = $null
    Write-Status "waiting for checkpoint $ckpt"
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $ckpt) {
            $size = (Get-Item -LiteralPath $ckpt).Length
            if ($size -gt 0 -and $size -eq $lastSize) {
                if ($null -eq $stableSince) {
                    $stableSince = Get-Date
                } elseif (((Get-Date) - $stableSince).TotalSeconds -ge $CheckpointStableSeconds) {
                    Write-Status "checkpoint stable: $ckpt size=$size"
                    return
                }
            } else {
                $lastSize = $size
                $stableSince = $null
            }
        }
        Start-Sleep -Seconds 2
    }
    Write-Status "checkpoint wait timed out: $ckpt"
}

$bestRecall = $BaselineRecall
$bestEpoch = $BaselineEpoch
$badCount = 0
$seen = @{}
$pattern = '\[Epoch\s+(\d+)\].*?val_conf_no_nms_recall@6=([0-9.]+)'

Write-Status ("monitor started pid={0} baseline={1:F4}@{2} patience={3}" -f $PidToStop, $bestRecall, $bestEpoch, $Patience)

while ($true) {
    $proc = Get-Process -Id $PidToStop -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        Write-Status "training process exited pid=$PidToStop"
        exit 0
    }

    if (Test-Path -LiteralPath $LogPath) {
        $text = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
        if ($null -ne $text) {
            foreach ($match in [regex]::Matches($text, $pattern)) {
                $epoch = [int]$match.Groups[1].Value
                if ($seen.ContainsKey($epoch)) {
                    continue
                }
                $seen[$epoch] = $true
                $recall = [double]$match.Groups[2].Value

                if ($recall -gt ($bestRecall + $MinDelta)) {
                    $bestRecall = $recall
                    $bestEpoch = $epoch
                    $badCount = 0
                    Write-Status ("epoch={0:D4} recall={1:F4} improved best" -f $epoch, $recall)
                } else {
                    $badCount += 1
                    Write-Status (
                        "epoch={0:D4} recall={1:F4} no improvement ({2}/{3}); best={4:F4}@{5:D4}" -f
                        $epoch, $recall, $badCount, $Patience, $bestRecall, $bestEpoch
                    )
                }

                if ($badCount -ge $Patience) {
                    Wait-StableCheckpoint -Epoch $epoch
                    Write-Status "stopping pid=$PidToStop; recall plateau after $badCount validation checks"
                    Stop-Process -Id $PidToStop -Force -ErrorAction SilentlyContinue
                    exit 0
                }
            }
        }
    }

    Start-Sleep -Seconds $PollSeconds
}
