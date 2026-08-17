<#
Registers a Windows scheduled task that runs scripts\monitor.py every N minutes.
Usage (PowerShell):
  .\register_monitor_task.ps1 -IntervalMinutes 30 -TaskName "TradingBotMonitor" -PythonPath "C:\\Python39\\python.exe"

If PythonPath is omitted, uses 'python' from PATH. Run as the user who should own the scheduled task.
#>
param(
    [int]$IntervalMinutes = 30,
    [string]$TaskName = "TradingBotMonitor",
    [string]$PythonPath = "python"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path | Resolve-Path
$repoRoot = (Get-Item (Join-Path $scriptDir "..") ).FullName
$monitorScript = Join-Path $repoRoot "scripts\monitor.py"

if (-not (Test-Path $monitorScript)) {
    Write-Error "Could not find monitor.py at $monitorScript"
    exit 2
}

$python = $PythonPath
$action = "`"$python`" `"$monitorScript`""
Write-Output "Creating scheduled task '$TaskName' to run every $IntervalMinutes minutes: $action"

# Schtasks command - run as the current user, highest run level
$cmd = "schtasks /Create /SC MINUTE /MO $IntervalMinutes /TN $TaskName /TR `"$action`" /F"
Write-Output $cmd

try {
    $p = Start-Process -FilePath schtasks -ArgumentList "/Create", "/SC", "MINUTE", "/MO", $IntervalMinutes, "/TN", $TaskName, "/TR", $action, "/F" -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -eq 0) { Write-Output "Scheduled task created." } else { Write-Warning "schtasks returned exit code $($p.ExitCode)" }
} catch {
    Write-Error "Failed to create scheduled task: $_"
}
