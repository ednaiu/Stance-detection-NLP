# Deploy Stance Detection as WSGI Application on Shared Hosting
# PowerShell version for Windows users

param(
    [string]$User = "u3089870",
    [string]$Host = "31.31.198.9"
)

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   STANCE DETECTION - WSGI DEPLOYMENT ON SHARED HOSTING         ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if .env file exists and load credentials
if (Test-Path ".env") {
    Write-Host "Loading credentials from .env..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match '(.+)=(.+)') {
            $name, $value = $matches[1], $matches[2]
            Set-Item -Path "env:$name" -Value $value
        }
    }
    $User = $env:REMOTE_USER
    $Host = $env:REMOTE_HOST
}

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  User: $User"
Write-Host "  Host: $Host"
Write-Host "  App Directory: /var/www/$User/data/stance-detection"
Write-Host ""

$AppDir = "/var/www/$User/data/stance-detection"

# Helper function to run SSH commands
function Invoke-SSH {
    param(
        [string]$Command,
        [string]$Description
    )
    Write-Host $Description -ForegroundColor Cyan
    ssh $User@$Host $Command
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Success" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
    Write-Host ""
}

# Step 1: Create directories
Invoke-SSH `
    "mkdir -p $AppDir && chmod 755 $AppDir" `
    "Step 1: Creating application directory..."

# Step 2: Clone/pull repository
Invoke-SSH `
    "cd $AppDir && if [ -d .git ]; then git pull origin main; else git clone https://github.com/ednaiu/Stance-detection.git .; fi && mkdir -p data/processed data/raw models" `
    "Step 2: Setting up application files..."

# Step 3: Setup virtual environment
Invoke-SSH `
    "cd $AppDir && if [ ! -d venv ]; then python3 -m venv venv; fi && source venv/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r StanceClassifier/requirements.txt" `
    "Step 3: Setting up Python virtual environment (this may take 5-10 minutes)..."

# Step 4: Create startup script
Write-Host "Step 4: Creating startup script..." -ForegroundColor Cyan
$RunScriptContent = @"
#!/bin/bash
cd $AppDir
source venv/bin/activate
export PYTHONUNBUFFERED=1
export MODEL_PATH=models/sentence_embedding_baseline
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 60 --access-logfile - wsgi:app
"@

ssh $User@$Host "cat > $AppDir/run.sh << 'RUNSCRIPT'
$RunScriptContent
RUNSCRIPT
chmod +x $AppDir/run.sh"

Write-Host "✓ Startup script created" -ForegroundColor Green
Write-Host ""

# Display next steps
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    DEPLOYMENT COMPLETE                          ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "📝 NEXT STEPS:" -ForegroundColor Yellow
Write-Host ""

Write-Host "1️⃣ Start the application on the server:" -ForegroundColor Cyan
Write-Host "   ssh $User@$Host" -ForegroundColor White
Write-Host "   cd $AppDir" -ForegroundColor White
Write-Host "   source venv/bin/activate" -ForegroundColor White
Write-Host "   ./run.sh" -ForegroundColor White
Write-Host ""

Write-Host "2️⃣ Test in browser:" -ForegroundColor Cyan
Write-Host "   http://$Host:5000/health" -ForegroundColor White
Write-Host ""

Write-Host "3️⃣ To run in background (using screen):" -ForegroundColor Cyan
Write-Host "   ssh $User@$Host" -ForegroundColor White
Write-Host "   screen -S stance" -ForegroundColor White
Write-Host "   cd $AppDir && source venv/bin/activate && ./run.sh" -ForegroundColor White
Write-Host "   # Press Ctrl+A then D to detach" -ForegroundColor White
Write-Host ""

Write-Host "4️⃣ Contact reg.ru support to:" -ForegroundColor Cyan
Write-Host "   - Set up a systemd service for auto-start" -ForegroundColor White
Write-Host "   - Or create a cron job for auto-restart" -ForegroundColor White
Write-Host ""

Write-Host "📚 API Endpoints:" -ForegroundColor Cyan
Write-Host "   GET  http://$Host:5000/health" -ForegroundColor White
Write-Host "   GET  http://$Host:5000/" -ForegroundColor White
Write-Host "   POST http://$Host:5000/classify" -ForegroundColor White
Write-Host "   POST http://$Host:5000/classify_batch" -ForegroundColor White
Write-Host ""

Write-Host "📖 Full documentation: WSGI_DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""
