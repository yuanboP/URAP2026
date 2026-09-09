$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $repoRoot 'artifacts\detached_optical_flow_4panel_gdrive'
if (-not (Test-Path -LiteralPath $runDir)) { Write-Output 'NOT RUNNING: no upload run directory'; return }
$stopPath = Join-Path $runDir 'STOP_REQUESTED'
Get-Date -Format o | Set-Content -LiteralPath $stopPath -Encoding ASCII
Write-Output "Stop requested: $stopPath"
Write-Output 'This is not confirmation of termination; run monitor_optical_flow_4panel_gdrive.ps1.'
