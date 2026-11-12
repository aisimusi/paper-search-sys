[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'configure', 'init', 'doctor', 'collect', 'run', 'resume', 'demo', 'history', 'schedule', 'research', 'research-resume', 'research-status', 'mcp-serve', 'mcp-tools', 'mcp-call')]
    [string]$Command = 'doctor',
    [string]$ProjectRoot = '',
    [string]$Config = 'config.json',
    [string]$CondaExe = '',
    [string]$Output = 'deploy',
    [string]$Session = '',
    [string]$Goal = '',
    [string]$InputFile = '',
    [string]$TaskDir = '',
    [string]$Answer = '',
    [string]$Server = '',
    [string]$Tool = '',
    [string]$ArgumentsFile = '',
    [ValidateSet('stdio', 'streamable-http')]
    [string]$Transport = 'stdio',
    [int]$Port = 8765,
    [int]$MaxSteps = 20,
    [int]$MaxModelCalls = 30,
    [int]$TokenBudget = 250000,
    [int]$MaxAttempts = 2,
    [int]$MaxReviewRounds = 3,
    [ValidateSet('', 'codex-session', 'codex', 'harness', 'none')]
    [string]$Backend = '',
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'

function Find-Conda {
    if ($CondaExe) {
        if (-not (Test-Path -LiteralPath $CondaExe -PathType Leaf)) { throw "Conda not found: $CondaExe" }
        return (Resolve-Path -LiteralPath $CondaExe).Path
    }
    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) { return $env:CONDA_EXE }
    $available = Get-Command conda.exe, conda.bat -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($available) { return $available.Source }
    foreach ($base in @($env:USERPROFILE, $env:ProgramData, $env:LOCALAPPDATA)) {
        if (-not $base) { continue }
        $candidate = Join-Path $base 'anaconda3\Scripts\conda.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw 'Anaconda was not found. Run from Anaconda PowerShell Prompt or pass -CondaExe <path-to-conda.exe>.'
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Executable" }
}

$previousPath = $env:PATH
$previousUtf8 = $env:PYTHONUTF8
$previousEncoding = $env:PYTHONIOENCODING
$previousConsoleEncoding = [Console]::OutputEncoding
$pushed = $false
$resultCode = 1
try {
    if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
    $root = (Resolve-Path -LiteralPath $ProjectRoot).Path
    if (-not (Test-Path -LiteralPath (Join-Path $root 'pyproject.toml'))) { throw 'ProjectRoot must contain pyproject.toml.' }
    Push-Location -LiteralPath $root
    $pushed = $true
    $prefix = Join-Path $root '.conda'
    $python = Join-Path $prefix 'python.exe'
    $pdf = Join-Path $prefix 'Library\bin\pdftotext.exe'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    $OutputEncoding = [Console]::OutputEncoding
    if (-not (Test-Path -LiteralPath $python)) {
        $conda = Find-Conda
        Write-Host "Creating Anaconda environment: $prefix"
        Invoke-Checked $conda @('create', '--yes', '--prefix', $prefix, '--override-channels', '--channel', 'conda-forge', 'python=3.11', 'pip', 'poppler')
    } elseif (-not (Test-Path -LiteralPath (Join-Path $prefix 'conda-meta'))) {
        throw '.conda exists but is not a Conda environment. Choose a project directory with a valid environment.'
    }
    if (-not (Test-Path -LiteralPath $pdf)) {
        $conda = Find-Conda
        Invoke-Checked $conda @('install', '--yes', '--prefix', $prefix, '--override-channels', '--channel', 'conda-forge', 'poppler')
    }
    $env:PATH = "$prefix;$prefix\Library\mingw-w64\bin;$prefix\Library\usr\bin;$prefix\Library\bin;$prefix\Scripts;$previousPath"
    Invoke-Checked $python @('-c', 'import sys; assert sys.version_info >= (3, 10)')
    $configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $root $Config }
    $selectedBackend = if ($Backend) { $Backend } else { 'codex-session' }
    if (-not $Backend -and $Command -ne 'resume' -and (Test-Path -LiteralPath $configPath)) {
        $settings = Get-Content -Raw -Encoding UTF8 -LiteralPath $configPath | ConvertFrom-Json
        if ($settings.agent.backend) { $selectedBackend = $settings.agent.backend }
    }
    if ($Command -eq 'resume') { $selectedBackend = 'codex-session' }
    $researchCommand = $Command -in @('research', 'research-resume', 'research-status')
    if ($Command -in @('research-resume', 'research-status')) {
        if (-not $TaskDir) { throw 'research-resume/research-status requires -TaskDir.' }
        $taskSettings = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $TaskDir 'task.json') | ConvertFrom-Json
        $selectedBackend = $taskSettings.initial_state.config.agent.backend
    }
    $mcpCommand = $Command -in @('mcp-serve', 'mcp-tools', 'mcp-call')
    $mcpEnabled = $mcpCommand
    if ($researchCommand) {
        $mcpConfig = if ($Command -in @('research-resume', 'research-status')) { $taskSettings.initial_state.config } elseif (Test-Path -LiteralPath $configPath) { Get-Content -Raw -Encoding UTF8 -LiteralPath $configPath | ConvertFrom-Json } else { $null }
        if ($mcpConfig -and $mcpConfig.mcp -and $mcpConfig.mcp.servers -and $mcpConfig.mcp.servers.PSObject.Properties.Count) { $mcpEnabled = $true }
    }
    $marker = Join-Path $prefix '.scholar-daily-dependencies'
    $signature = (Get-FileHash -LiteralPath (Join-Path $root 'pyproject.toml') -Algorithm SHA256).Hash + ':' + $selectedBackend
    if ($researchCommand) { $signature += ':orchestrator' }
    if ($mcpEnabled) { $signature += ':mcp' }
    $installed = (Test-Path -LiteralPath $marker) -and ((Get-Content -Raw -LiteralPath $marker).Trim() -eq $signature)
    if (-not $installed -or $Command -eq 'setup') {
        $extras = @()
        if ($selectedBackend -eq 'harness') { $extras += 'harness' }
        if ($researchCommand) { $extras += 'orchestrator' }
        if ($mcpEnabled) { $extras += 'mcp' }
        $installTarget = if ($extras.Count) { $root + '[' + ($extras -join ',') + ']' } else { $root }
        Invoke-Checked $python @('-m', 'pip', 'install', '--editable', $installTarget)
        if ($selectedBackend -eq 'harness') {
            Invoke-Checked $python @('-c', 'import deepseek_harness; from deepseek_harness_runtime import bundled_runtime_path; print(bundled_runtime_path())')
        }
        Set-Content -LiteralPath $marker -Value $signature -Encoding ASCII
    }
    $created = $Command -notin @('resume', 'research-resume', 'research-status') -and -not (Test-Path -LiteralPath $configPath)
    if ($created) { Invoke-Checked $python @('-m', 'scholar_daily', 'init', '--config', $configPath) }
    if (($created -or $Command -eq 'setup') -and -not $NonInteractive -and $Command -notin @('init', 'configure')) {
        Invoke-Checked $python @('-m', 'scholar_daily', 'configure', '--config', $configPath)
    }
    if ($NonInteractive -and $Command -eq 'configure') { throw 'configure requires interactive input. Edit config.json directly for unattended use.' }
    $effectiveCommand = if ($Command -eq 'setup') { 'doctor' } else { $Command }
    $cliArgs = @('-m', 'scholar_daily', $effectiveCommand, '--config', $configPath)
    if ($effectiveCommand -eq 'mcp-serve') { $cliArgs += @('--transport', $Transport, '--port', $Port) }
    if ($effectiveCommand -in @('mcp-tools', 'mcp-call')) {
        if (-not $Server) { throw 'MCP client requires -Server.' }
        $cliArgs += @('--server', $Server)
    }
    if ($effectiveCommand -eq 'mcp-call') {
        if (-not $Tool -or -not $ArgumentsFile) { throw 'mcp-call requires -Tool and -ArgumentsFile.' }
        $cliArgs += @('--tool', $Tool, '--arguments', $ArgumentsFile)
    }
    if ($effectiveCommand -eq 'schedule') { $cliArgs += @('--output', $Output) }
    if ($effectiveCommand -eq 'resume') {
        if (-not $Session) { throw 'resume requires -Session <path-to-session.json>.' }
        $cliArgs += @('--session', $Session)
    }
    if ($effectiveCommand -eq 'research') {
        if (-not $Goal) { throw 'research requires -Goal.' }
        $cliArgs += @('--goal', $Goal, '--max-steps', $MaxSteps, '--max-model-calls', $MaxModelCalls, '--token-budget', $TokenBudget, '--max-attempts', $MaxAttempts, '--max-review-rounds', $MaxReviewRounds)
        if ($InputFile) { $cliArgs += @('--input', $InputFile) }
    }
    if ($effectiveCommand -in @('research-resume', 'research-status')) { $cliArgs += @('--task-dir', $TaskDir) }
    if ($effectiveCommand -eq 'research-resume' -and $Answer) { $cliArgs += @('--answer', $Answer) }
    if ($Backend -and $effectiveCommand -in @('run', 'doctor', 'research')) { $cliArgs += @('--backend', $Backend) }
    if ($Command -eq 'init' -and $created) {
        $resultCode = 0
    } else {
        & $python @cliArgs
        $resultCode = $LASTEXITCODE
    }
    if ($Command -eq 'setup' -or $created) {
        Write-Host 'Next: edit config.json or run scripts/scholar-daily.ps1 configure.'
        if ($selectedBackend -eq 'harness') { Write-Host 'Set DEEPSEEK_API_KEY in your local environment; never paste it into chat or a shared config.' }
        Write-Host 'Use doctor to check local readiness, demo for offline output, collect for search, run for model analysis.'
    }
} catch {
    Write-Error $_ -ErrorAction Continue
    $resultCode = 1
} finally {
    $env:PATH = $previousPath
    $env:PYTHONUTF8 = $previousUtf8
    $env:PYTHONIOENCODING = $previousEncoding
    [Console]::OutputEncoding = $previousConsoleEncoding
    if ($pushed) { Pop-Location }
}
exit $resultCode
