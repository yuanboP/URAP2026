param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$DestinationRoot,
    [Parameter(Mandatory = $true)][string]$RunDir
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceRoot).Path.TrimEnd('\')
$destination = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\')
$runRoot = [IO.Path]::GetFullPath($RunDir)
$driveRoot = 'G:\My Drive'
if (-not ($destination.Equals($driveRoot, [StringComparison]::OrdinalIgnoreCase) -or
          $destination.StartsWith($driveRoot + '\', [StringComparison]::OrdinalIgnoreCase))) {
    throw "Destination must stay under $driveRoot"
}

New-Item -ItemType Directory -Force -Path $destination, $runRoot | Out-Null
$statusPath = Join-Path $runRoot 'progress.json'
$stopPath = Join-Path $runRoot 'STOP_REQUESTED'
$startedAt = (Get-Date).ToString('o')
$files = @(Get-ChildItem -LiteralPath $source -Filter 'comparison.mp4' -Recurse -File | Sort-Object FullName)
if ($files.Count -eq 0) { throw "No comparison.mp4 files found under $source" }
$totalBytes = [long](($files | Measure-Object Length -Sum).Sum)
$done = 0
$doneBytes = [long]0
$copiedThisRun = 0
$skippedExisting = 0
$failures = @()
$lastCompleted = $null

function Write-ProgressState([string]$Phase, [string]$Current = $null) {
    $destinationFiles = @(Get-ChildItem -LiteralPath $destination -Filter 'comparison.mp4' -Recurse -File -ErrorAction SilentlyContinue)
    $destinationBytes = [long](($destinationFiles | Measure-Object Length -Sum).Sum)
    $payload = [ordered]@{
        phase = $Phase
        pid = $PID
        started_at = $startedAt
        updated_at = (Get-Date).ToString('o')
        source_root = $source
        destination_root = $destination
        done = $done
        total = $files.Count
        done_bytes = $doneBytes
        total_bytes = $totalBytes
        copied_this_run = $copiedThisRun
        skipped_existing = $skippedExisting
        current = $Current
        last_completed_unit = $lastCompleted
        destination_count = $destinationFiles.Count
        destination_bytes = $destinationBytes
        failures = $failures
    }
    $temporary = $statusPath + '.tmp'
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

Write-ProgressState -Phase 'copying'
foreach ($file in $files) {
    if (Test-Path -LiteralPath $stopPath) {
        Write-ProgressState -Phase 'stopped' -Current $file.FullName
        Write-Output "STOPPED done=$done/$($files.Count) last=$lastCompleted"
        exit 3
    }
    $relative = $file.FullName.Substring($source.Length).TrimStart('\')
    $target = Join-Path $destination $relative
    $targetDirectory = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
    try {
        $existing = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
        if ($existing -and $existing.Length -eq $file.Length) {
            $skippedExisting += 1
        } else {
            $partial = $target + '.uploading'
            [IO.File]::Copy($file.FullName, $partial, $true)
            $partialItem = Get-Item -LiteralPath $partial
            if ($partialItem.Length -ne $file.Length) {
                throw "Copied size mismatch for $relative"
            }
            Move-Item -LiteralPath $partial -Destination $target -Force
            $copiedThisRun += 1
        }
        $done += 1
        $doneBytes += $file.Length
        $lastCompleted = $relative
        Write-ProgressState -Phase 'copying'
        Write-Output "$(Get-Date -Format o) done=$done/$($files.Count) file=$relative bytes=$($file.Length)"
    } catch {
        $failures += [ordered]@{ file = $relative; error = $_.Exception.Message; observed_at = (Get-Date).ToString('o') }
        Write-ProgressState -Phase 'copying_with_failures' -Current $relative
        Write-Error "FAILED file=$relative error=$($_.Exception.Message)"
    }
}

foreach ($name in @('summary.json', 'comparison_report.md', 'manifest.json')) {
    $supportSource = Join-Path $source $name
    if (Test-Path -LiteralPath $supportSource) {
        [IO.File]::Copy($supportSource, (Join-Path $destination $name), $true)
    }
}

$finalPhase = if ($failures.Count -eq 0) { 'completed' } else { 'completed_with_failures' }
Write-ProgressState -Phase $finalPhase
Write-Output "$(Get-Date -Format o) phase=$finalPhase done=$done/$($files.Count) failures=$($failures.Count) destination=$destination"
if ($failures.Count -gt 0) { exit 1 }
