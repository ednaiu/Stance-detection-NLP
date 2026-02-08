# PowerShell deployment script for Windows (with password authentication)
# Usage: .\deploy.ps1
# Set environment variables: $env:REMOTE_USER="root" and $env:REMOTE_PASSWORD="your_password"

$ErrorActionPreference = "Stop"

$REMOTE_HOST = "31.31.198.9"
$REMOTE_USER = if ($env:REMOTE_USER) { $env:REMOTE_USER } else { "root" }
$REMOTE_PASSWORD = $env:REMOTE_PASSWORD
$IMAGE_NAME = "stance-classifier"
$CONTAINER_NAME = "stance-classifier"

# Prompt for password if not set
if (-not $REMOTE_PASSWORD) {
    $securePassword = Read-Host "Enter password for ${REMOTE_USER}@${REMOTE_HOST}" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    $REMOTE_PASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
}

Write-Host "=== Building Docker image ===" -ForegroundColor Green
Set-Location StanceClassifier
docker build -t "${IMAGE_NAME}:latest" -f docker/Dockerfile .

Write-Host "=== Saving Docker image ===" -ForegroundColor Green
docker save "${IMAGE_NAME}:latest" | gzip > "$env:TEMP\${IMAGE_NAME}.tar.gz"

Write-Host "=== Installing plink if needed ===" -ForegroundColor Green
# Check if we have plink (PuTTY) or use WSL with sshpass
$usePlink = $false
$useSshpass = $false

if (Get-Command plink -ErrorAction SilentlyContinue) {
    $usePlink = $true
    Write-Host "Using plink for deployment" -ForegroundColor Cyan
} elseif (Get-Command wsl -ErrorAction SilentlyContinue) {
    $useSshpass = $true
    Write-Host "Using WSL with sshpass for deployment" -ForegroundColor Cyan
} else {
    Write-Host "Installing plink (PuTTY)..." -ForegroundColor Yellow
    # Install plink via chocolatey or download directly
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install putty -y
        $usePlink = $true
    } else {
        Write-Host "ERROR: Please install PuTTY (plink) or WSL to use password authentication" -ForegroundColor Red
        Write-Host "Alternative: Use SSH key authentication (see SSH_SETUP_REGRU.md)" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "=== Copying image to remote host ${REMOTE_HOST} ===" -ForegroundColor Green
if ($usePlink) {
    # Using plink and pscp (PuTTY tools)
    echo y | plink -pw $REMOTE_PASSWORD "${REMOTE_USER}@${REMOTE_HOST}" "exit"
    pscp -pw $REMOTE_PASSWORD "$env:TEMP\${IMAGE_NAME}.tar.gz" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/"
} elseif ($useSshpass) {
    # Using WSL with sshpass
    wsl bash -c "sshpass -p '$REMOTE_PASSWORD' scp -o StrictHostKeyChecking=no $env:TEMP\${IMAGE_NAME}.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:/tmp/"
}

Write-Host "=== Deploying on remote host ===" -ForegroundColor Green
$deployScript = @"
docker load < /tmp/stance-classifier.tar.gz
docker stop stance-classifier 2>/dev/null || true
docker rm stance-classifier 2>/dev/null || true
docker run -d --name stance-classifier --restart unless-stopped -p 5000:5000 -v /data/stance-detection/data:/app/data -v /data/stance-detection/models:/app/models -e MODEL_PATH=models/sentence_embedding_baseline -e WORKERS=2 stance-classifier:latest
rm /tmp/stance-classifier.tar.gz
docker ps | grep stance-classifier
docker logs --tail 20 stance-classifier
"@

if ($usePlink) {
    echo $deployScript | plink -pw $REMOTE_PASSWORD "${REMOTE_USER}@${REMOTE_HOST}"
} elseif ($useSshpass) {
    wsl bash -c "sshpass -p '$REMOTE_PASSWORD' ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} '$deployScript'"
}

Write-Host "=== Cleaning up local files ===" -ForegroundColor Green
Remove-Item "$env:TEMP\${IMAGE_NAME}.tar.gz"

Write-Host "=== Deployment completed successfully! ===" -ForegroundColor Green
Write-Host "API available at: http://${REMOTE_HOST}:5000" -ForegroundColor Cyan
