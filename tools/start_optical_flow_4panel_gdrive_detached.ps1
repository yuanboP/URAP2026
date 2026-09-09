param(
    [string]$SourceRoot = 'H:\URAP_OpticalFlow_20260903',
    [string]$DestinationRoot = 'G:\My Drive\URAP_OpticalFlow_20260903_4Panel',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$destination = [IO.Path]::GetFullPath($DestinationRoot)
$worker = Join-Path $repoRoot 'tools\upload_optical_flow_4panel_gdrive.ps1'
$runDir = Join-Path $repoRoot 'artifacts\detached_optical_flow_4panel_gdrive'
$pidPath = Join-Path $runDir 'upload.pid'
$metadataPath = Join-Path $runDir 'run.json'
$progressPath = Join-Path $runDir 'progress.json'
$stopPath = Join-Path $runDir 'STOP_REQUESTED'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if (-not (Test-Path -LiteralPath 'G:\My Drive')) { throw 'Google Drive mount G:\My Drive is unavailable.' }
if (-not (Test-Path -LiteralPath $worker)) { throw "Worker script missing: $worker" }
$sourceFiles = @(Get-ChildItem -LiteralPath $source -Filter 'comparison.mp4' -Recurse -File)
if ($sourceFiles.Count -eq 0) { throw "No comparison.mp4 files found under $source" }
$requiredBytes = [long](($sourceFiles | Measure-Object Length -Sum).Sum)
$drive = Get-PSDrive -Name G
if ($drive.Free -lt $requiredBytes) { throw "Insufficient Google Drive free space: need=$requiredBytes free=$($drive.Free)" }

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -and $old.CommandLine.Contains('upload_optical_flow_4panel_gdrive.ps1')) {
        throw "Upload is already running: PID=$oldPid"
    }
}
if ((Test-Path -LiteralPath $progressPath) -and -not $Resume) {
    $previous = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
    Write-Output "NOT RUNNING previous_phase=$($previous.phase) done=$($previous.done)/$($previous.total)"
    throw 'Existing upload state requires explicit -Resume.'
}
if (Test-Path -LiteralPath $stopPath) {
    if (-not $Resume) { throw 'STOP_REQUESTED exists; use -Resume to clear it for an explicit restart.' }
    Remove-Item -LiteralPath $stopPath -Force
}

$launchId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdout = Join-Path $runDir ("upload_$launchId.stdout.log")
$stderr = Join-Path $runDir ("upload_$launchId.stderr.log")
$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $worker + '"'),
    '-SourceRoot', ('"' + $source + '"'),
    '-DestinationRoot', ('"' + $destination + '"'),
    '-RunDir', ('"' + $runDir + '"')
)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
    resume = [bool]$Resume
    source_root = $source
    destination_root = $destination
    source_count = $sourceFiles.Count
    source_bytes = $requiredBytes
    progress = $progressPath
    stdout = $stdout
    stderr = $stderr
    command = ('powershell.exe ' + ($arguments -join ' '))
}
$metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
$metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir ("run_$launchId.json")) -Encoding UTF8
Write-Output "LAUNCHED pid=$($process.Id) start=$($metadata.started_at) done/total=0/$($sourceFiles.Count)"
Write-Output "destination=$destination"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"
Write-Output 'Launch is not proof of upload progress; use monitor_optical_flow_4panel_gdrive.ps1.'
