# Commit and push automation scaffolding and fixes to origin/main
# Usage: run this from the repository root in PowerShell
# Ensure you have git installed and authenticated (gh auth login or credential helper configured)

param(
    [string]$Remote = "origin",
    [string]$Branch = "main",
    [string]$CommitMessage = "Add autonomous monitor, reproduce/triage, propose-fix, and conditional auto-merge pipeline; make backtest robust to missing Alpaca keys`n`nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
)

Write-Output "Committing and pushing automation changes..."

# Ensure we're in the repo root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Resolve-Path
Set-Location $repoRoot

# Check for git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git not found in PATH. Install git and retry: https://git-scm.com/downloads"
    exit 2
}

# Initialize repo if needed
if (-not (Test-Path ".git")) {
    git init
}

# Configure a reasonable default user if none set
$usr = git config user.name
if ([string]::IsNullOrEmpty($usr)) { git config user.name "autofix-agent" }
$mail = git config user.email
if ([string]::IsNullOrEmpty($mail)) { git config user.email "autofix@local" }

# Stage files we added/modified by the agent
git add scripts config .github backtest.py bot.py

# Commit
$commitRes = git commit -m $CommitMessage
if ($LASTEXITCODE -ne 0) {
    Write-Warning "git commit returned exit code $LASTEXITCODE. Output: $commitRes"
} else {
    Write-Output "Committed changes."
}

# Add remote if missing
$remoteExists = git remote | Select-String -Pattern "^$Remote$"
if (-not $remoteExists) {
    Write-Output "Remote '$Remote' not found. Please run: git remote add $Remote <your-repo-url>"
    exit 1
}

# Ensure branch name set and push
git branch -M $Branch
git push -u $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed. Check authentication (gh auth login) or use GitHub Desktop to push."
    exit 1
}

Write-Output "Pushed to $Remote/$Branch. The GitHub Actions workflow should trigger shortly."