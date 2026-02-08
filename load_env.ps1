# Load environment variables from .env file
# Usage: . .\load_env.ps1

$envFile = ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "Create .env file with:" -ForegroundColor Yellow
    Write-Host "  REMOTE_HOST=31.31.198.9" -ForegroundColor Gray
    Write-Host "  REMOTE_USER=root" -ForegroundColor Gray
    Write-Host "  REMOTE_PASSWORD=your_password" -ForegroundColor Gray
    return
}

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^([^=]+)=(.+)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        Set-Item -Path "env:$name" -Value $value
    }
}

Write-Host "✓ Environment variables loaded from .env" -ForegroundColor Green
Write-Host "  REMOTE_HOST: $env:REMOTE_HOST" -ForegroundColor Cyan
Write-Host "  REMOTE_USER: $env:REMOTE_USER" -ForegroundColor Cyan
Write-Host "  REMOTE_PASSWORD: ********" -ForegroundColor Cyan
