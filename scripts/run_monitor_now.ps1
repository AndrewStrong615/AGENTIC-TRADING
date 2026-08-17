# Run the monitor once and save output to artifacts with timestamp
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Resolve-Path
$RepoRoot = (Get-Item $RepoRoot).Parent.FullName
$Artifacts = Join-Path $RepoRoot "artifacts"
if (-not (Test-Path $Artifacts)) { New-Item -ItemType Directory -Path $Artifacts | Out-Null }
$ts = (Get-Date).ToString('yyyyMMddTHHmmssZ')
$outdir = Join-Path $Artifacts "monitor-run-$ts"
New-Item -ItemType Directory -Path $outdir | Out-Null

$python = $env:PYTHON_PATH
if (-not $python) { $python = "python" }
$monitor = Join-Path $RepoRoot "scripts\monitor.py"

Write-Output "Running monitor via: $python $monitor"
try {
    & $python $monitor 2>&1 | Tee-Object -FilePath (Join-Path $outdir "monitor_output.txt")
    $exit = $LASTEXITCODE
} catch {
    $_ | Out-File (Join-Path $outdir "monitor_error.txt")
    $exit = 2
}

if ($exit -ne 0) { exit $exit }
