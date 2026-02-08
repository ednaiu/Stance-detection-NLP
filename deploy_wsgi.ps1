# Deploy Stance Detection as WSGI Application on Shared Hosting
# PowerShell version for Windows users

Write-Host ""
Write-Host "STANCE DETECTION - WSGI DEPLOYMENT" -ForegroundColor Cyan
Write-Host ""

# Load .env if exists
if (Test-Path ".env") {
    Write-Host "Loading credentials..." -ForegroundColor Yellow
    Get-Content .env | Where-Object { $_ -match '=' } | ForEach-Object {
        $parts = $_.Split('=')
        if ($parts.Count -eq 2) {
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim())
        }
    }
}

$User = [Environment]::GetEnvironmentVariable('REMOTE_USER')
$RemoteHost = [Environment]::GetEnvironmentVariable('REMOTE_HOST')

if ([string]::IsNullOrEmpty($User)) { $User = "u3089870" }
if ([string]::IsNullOrEmpty($RemoteHost)) { $RemoteHost = "31.31.198.9" }

Write-Host "Deploying to: $User@$RemoteHost" -ForegroundColor Green
Write-Host ""

Write-Host "Step 1: Cloning repository..." -ForegroundColor Cyan
ssh $User@$RemoteHost "mkdir -p /var/www/$User/data/stance-detection"
ssh $User@$RemoteHost "cd /var/www/$User/data/stance-detection && git clone https://github.com/ednaiu/Stance-detection.git . 2>/dev/null || git pull origin main"
Write-Host "Done" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Installing Python dependencies..." -ForegroundColor Cyan
Write-Host "(This may take 5-10 minutes)" -ForegroundColor Yellow
ssh $User@$RemoteHost "cd /var/www/$User/data/stance-detection && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r StanceClassifier/requirements.txt"
Write-Host "Done" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Creating startup script..." -ForegroundColor Cyan
$ScriptPath = "/var/www/$User/data/stance-detection/run.sh"
ssh $User@$RemoteHost "cat > $ScriptPath << 'ENDOFSCRIPT'
#!/bin/bash
cd /var/www/$User/data/stance-detection
source venv/bin/activate
export PYTHONUNBUFFERED=1
export MODEL_PATH=models/sentence_embedding_baseline
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 60 --access-logfile - wsgi:app
ENDOFSCRIPT
chmod +x $ScriptPath"
Write-Host "Done" -ForegroundColor Green
Write-Host ""

Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Connect: ssh $User@$RemoteHost" -ForegroundColor White
Write-Host "2. Navigate: cd /var/www/$User/data/stance-detection" -ForegroundColor White
Write-Host "3. Activate: source venv/bin/activate" -ForegroundColor White
Write-Host "4. Run: ./run.sh" -ForegroundColor White
Write-Host ""
Write-Host "API: http://$RemoteHost:5000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host ""
