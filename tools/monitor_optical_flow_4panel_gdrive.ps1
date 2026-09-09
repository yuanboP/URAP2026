param([int]$IntervalSeconds = 5)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $repoRoot 'artifacts\detached_optical_flow_4panel_gdrive'
$metadataPath = Join-Path $runDir 'run.json'
if (-not (Test-Path -LiteralPath $metadataPath)) { Write-Output 'NOT RUNNING: no upload metadata'; return }
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json

function Read-State {
    if (-not (Test-Path -LiteralPath $metadata.progress)) { return $null }
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { return (Get-Content -LiteralPath $metadata.progress -Raw | ConvertFrom-Json) }
        catch {
            if ($attempt -eq 19) { throw }
            Start-Sleep -Milliseconds (10 * ($attempt + 1))
        }
    }
}

function Snapshot {
    $state = Read-State
    $files = @(Get-ChildItem -LiteralPath $metadata.destination_root -Filter 'comparison.mp4' -Recurse -File -ErrorAction SilentlyContinue)
    $bytes = [long](($files | Measure-Object Length -Sum).Sum)
    return [ordered]@{ state = $state; count = $files.Count; bytes = $bytes; signature = "$($state.done)|$($state.updated_at)|$($files.Count)|$bytes" }
}

$first = Snapshot
Start-Sleep -Seconds ([Math]::Max(1, $IntervalSeconds))
$second = Snapshot
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($metadata.pid)" -ErrorAction SilentlyContinue
$matched = $process -and $process.CommandLine -and $process.CommandLine.Contains('upload_optical_flow_4panel_gdrive.ps1')
$advanced = $first.signature -ne $second.signature
if ($matched -and $advanced) { $status = 'RUNNING; PID MATCHED; PROGRESS ADVANCED' }
elseif ($matched) { $status = 'RUNNING; PID MATCHED; NO PROGRESS ADVANCE OBSERVED' }
elseif ($second.state.phase -eq 'completed') { $status = 'COMPLETED; NOT RUNNING' }
else { $status = 'NOT RUNNING' }

Write-Output "observed=$(Get-Date -Format o) status=$status"
Write-Output "done/total=$($second.state.done)/$($second.state.total) phase=$($second.state.phase) destination_count=$($second.count) destination_bytes=$($second.bytes)"
Write-Output "pid=$($metadata.pid) start=$($metadata.started_at) command=$($process.CommandLine)"
Write-Output "last_completed=$($second.state.last_completed_unit) last_output_timestamp=$($second.state.updated_at) current=$($second.state.current)"
Write-Output "copied_this_run=$($second.state.copied_this_run) skipped_existing=$($second.state.skipped_existing) failures=$(@($second.state.failures).Count)"
$driveProcesses = @(Get-Process GoogleDriveFS -ErrorAction SilentlyContinue)
$latestDriveLog = Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\Google\DriveFS\Logs" -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq 'drive_fs.txt' -or $_.Name -match '^structured_log_(global|[0-9]+)$' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
Write-Output "drivefs_pids=$(($driveProcesses.Id -join ',')) drivefs_latest_log=$($latestDriveLog.FullName) drivefs_log_updated=$($latestDriveLog.LastWriteTime.ToString('o'))"
Write-Output "destination=$($metadata.destination_root)"
Write-Output "stdout=$($metadata.stdout)"
Write-Output "stderr=$($metadata.stderr)"
if (Test-Path -LiteralPath $metadata.stdout) { Get-Content -LiteralPath $metadata.stdout -Tail 5 }
if ((Test-Path -LiteralPath $metadata.stderr) -and (Get-Item -LiteralPath $metadata.stderr).Length -gt 0) { Get-Content -LiteralPath $metadata.stderr -Tail 10 }
