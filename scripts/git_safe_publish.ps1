param(
    [string]$CommitMessage = "",
    [string]$Remote = "origin",
    [string]$Branch = "",
    [switch]$StageAll,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $display = "git " + ($Args -join " ")
    Write-Host ">> $display"
    $mutating = @("add", "commit", "fetch", "pull", "push")
    if ($DryRun -and $mutating -contains $Args[0]) {
        return ""
    }

    $output = & git @Args 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ($output -join [Environment]::NewLine)
    }
    return ($output -join [Environment]::NewLine).Trim()
}

function Get-CurrentBranch {
    $name = Invoke-Git -Args @("rev-parse", "--abbrev-ref", "HEAD")
    if (-not $name) {
        throw "Unable to determine current branch."
    }
    return $name
}

function Ensure-CleanIndexLock {
    $gitDir = Invoke-Git -Args @("rev-parse", "--git-dir")
    $lockPath = Join-Path $gitDir "index.lock"
    if (Test-Path -LiteralPath $lockPath) {
        throw "Refusing to continue because '$lockPath' exists. Finish the other git command or remove the stale lock intentionally."
    }
}

function Get-AheadBehind {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteBranch
    )

    $counts = Invoke-Git -Args @("rev-list", "--left-right", "--count", "HEAD...$RemoteBranch")
    $parts = $counts -split "\s+"
    if ($parts.Count -lt 2) {
        throw "Unexpected rev-list output: '$counts'"
    }
    return @{
        Ahead = [int]$parts[0]
        Behind = [int]$parts[1]
    }
}

function Has-StagedChanges {
    if ($DryRun) {
        return $false
    }
    & git diff --cached --quiet
    return ($LASTEXITCODE -ne 0)
}

function Has-UncommittedChanges {
    if ($DryRun) {
        return $false
    }
    $status = & git status --porcelain
    return [bool]$status
}

Ensure-CleanIndexLock

$currentBranch = if ($Branch) { $Branch } else { Get-CurrentBranch }

if ($StageAll) {
    Invoke-Git -Args @("add", "-A")
}

if ($CommitMessage) {
    if (-not (Has-StagedChanges)) {
        throw "Commit message was provided, but there are no staged changes to commit."
    }
    Invoke-Git -Args @("commit", "-m", $CommitMessage)
}

if (Has-UncommittedChanges) {
    throw "Working tree is not clean. Commit or stash changes before syncing and pushing."
}

Invoke-Git -Args @("fetch", $Remote, $currentBranch)

$remoteBranch = "$Remote/$currentBranch"
$aheadBehind = Get-AheadBehind -RemoteBranch $remoteBranch

if ($aheadBehind.Behind -gt 0) {
    Write-Host "Local branch is behind $remoteBranch by $($aheadBehind.Behind) commit(s); rebasing first."
    Invoke-Git -Args @("pull", "--rebase", $Remote, $currentBranch)
    $aheadBehind = Get-AheadBehind -RemoteBranch $remoteBranch
}

if ($aheadBehind.Ahead -eq 0 -and $aheadBehind.Behind -eq 0) {
    Write-Host "Branch is already in sync with $remoteBranch."
    exit 0
}

Invoke-Git -Args @("push", $Remote, $currentBranch)
Write-Host "Publish completed successfully."
