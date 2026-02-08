# Test connection to remote server
Write-Host "Testing connection to remote server..." -ForegroundColor Cyan
Write-Host ""

# Load environment variables from .env
if (Test-Path ".env") {
    Write-Host "Loading credentials from .env file..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match '(.+)=(.+)') {
            $name, $value = $matches[1], $matches[2]
            Set-Item -Path "env:$name" -Value $value
        }
    }
} else {
    Write-Host ".env file not found. Please create it first." -ForegroundColor Red
    exit 1
}

$REMOTE_HOST = $env:REMOTE_HOST
$REMOTE_USER = $env:REMOTE_USER
$REMOTE_PASSWORD = $env:REMOTE_PASSWORD

# Validate credentials are set
if (-not $REMOTE_HOST -or -not $REMOTE_USER -or -not $REMOTE_PASSWORD) {
    Write-Host "Error: REMOTE_HOST, REMOTE_USER, or REMOTE_PASSWORD not set" -ForegroundColor Red
    Write-Host "Please edit .env file with your credentials" -ForegroundColor Yellow
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Host: $REMOTE_HOST" -ForegroundColor Green
Write-Host "  User: $REMOTE_USER" -ForegroundColor Green
Write-Host ""

# Test connection using sshpass via WSL or Git Bash
Write-Host "Attempting connection..." -ForegroundColor Cyan
Write-Host ""

$bashPath = "C:\Program Files\Git\bin\bash.exe"

if (-not (Test-Path $bashPath)) {
    Write-Host "Git Bash not found. Installing or using WSL..." -ForegroundColor Yellow
    Write-Host "Please install Git for Windows from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Create temporary bash script
$tempBashScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.sh'
$bashScript = @"
#!/bin/bash
export SSHPASS='$REMOTE_PASSWORD'
sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $REMOTE_USER@$REMOTE_HOST 'whoami; uname -a; docker --version' 2>&1
exit `$?
"@

Set-Content -Path $tempBashScript -Value $bashScript -Encoding UTF8

try {
    Write-Host "Running SSH command..." -ForegroundColor Yellow
    Write-Host ""
    
    & $bashPath $tempBashScript
    $exitCode = $LASTEXITCODE
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "Connection successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Server information displayed above." -ForegroundColor Cyan
        Write-Host "Docker should be installed and running on your server." -ForegroundColor Green
    } else {
        Write-Host "Connection failed with exit code: $exitCode" -ForegroundColor Red
        Write-Host "Please verify your credentials in .env file" -ForegroundColor Yellow
        exit 1
    }
} finally {
    Remove-Item $tempBashScript -Force -ErrorAction SilentlyContinue
}

