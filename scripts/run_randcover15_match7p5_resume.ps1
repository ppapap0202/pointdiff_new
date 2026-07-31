$ErrorActionPreference = "Stop"

$root = "C:\pycharm\pointdiff_new"
$python = "C:\Users\crf\AppData\Local\anaconda3\envs\pointdiff\python.exe"
$config = Join-Path $root "config\train_randcover15_match7p5_from_0073.yaml"
$logs = Join-Path $root "logs"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$statusPath = Join-Path $logs "randcover15_match7p5_resume_status.txt"
$stdoutPath = Join-Path $logs "randcover15_match7p5_resume_${stamp}.stdout.log"
$stderrPath = Join-Path $logs "randcover15_match7p5_resume_${stamp}.stderr.log"

New-Item -ItemType Directory -Force -Path $logs | Out-Null
Set-Location $root

function Write-RunStatus {
    param([string[]] $Lines)

    $Lines | Set-Content -Path $statusPath -Encoding UTF8
}

Write-RunStatus @(
    "state=starting",
    "started_at=$(Get-Date -Format o)",
    "wrapper_pid=$PID",
    "python_pid=",
    "exit_code=",
    "stdout=$stdoutPath",
    "stderr=$stderrPath",
    "config=$config"
)

$stdoutStream = $null
$stderrStream = $null
$proc = $null

try {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $python
    $psi.Arguments = ".\main.py --config .\config\train_randcover15_match7p5_from_0073.yaml"
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void] $proc.Start()

    $stdoutStream = [System.IO.File]::Create($stdoutPath)
    $stderrStream = [System.IO.File]::Create($stderrPath)
    $stdoutCopy = $proc.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
    $stderrCopy = $proc.StandardError.BaseStream.CopyToAsync($stderrStream)

    Write-RunStatus @(
        "state=running",
        "started_at=$(Get-Date -Format o)",
        "wrapper_pid=$PID",
        "python_pid=$($proc.Id)",
        "exit_code=",
        "stdout=$stdoutPath",
        "stderr=$stderrPath",
        "config=$config"
    )

    $proc.WaitForExit()
    $stdoutCopy.Wait()
    $stderrCopy.Wait()

    Write-RunStatus @(
        "state=exited",
        "started_at=",
        "ended_at=$(Get-Date -Format o)",
        "wrapper_pid=$PID",
        "python_pid=$($proc.Id)",
        "exit_code=$($proc.ExitCode)",
        "stdout=$stdoutPath",
        "stderr=$stderrPath",
        "config=$config"
    )
} catch {
    Write-RunStatus @(
        "state=failed",
        "ended_at=$(Get-Date -Format o)",
        "wrapper_pid=$PID",
        "python_pid=$(if ($proc) { $proc.Id } else { '' })",
        "exit_code=",
        "error=$($_.Exception.Message)",
        "stdout=$stdoutPath",
        "stderr=$stderrPath",
        "config=$config"
    )
    throw
} finally {
    if ($stdoutStream) { $stdoutStream.Dispose() }
    if ($stderrStream) { $stderrStream.Dispose() }
}
