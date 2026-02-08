# PowerShell deployment script for Windows
# Usage: .\deploy.ps1

$ErrorActionPreference = "Stop"

$REMOTE_HOST = "31.31.198.9"
$REMOTE_USER = if ($env:REMOTE_USER) { $env:REMOTE_USER } else { "root" }
$IMAGE_NAME = "stance-classifier"
$CONTAINER_NAME = "stance-classifier"

Write-Host "=== Building Docker image ===" -ForegroundColor Green
Set-Location StanceClassifier
docker build -t "${IMAGE_NAME}:latest" -f docker/Dockerfile .

Write-Host "=== Saving Docker image ===" -ForegroundColor Green
docker save "${IMAGE_NAME}:latest" | gzip > "$env:TEMP\${IMAGE_NAME}.tar.gz"

Write-Host "=== Copying image to remote host ${REMOTE_HOST} ===" -ForegroundColor Green
scp "$env:TEMP\${IMAGE_NAME}.tar.gz" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/"

Write-Host "=== Deploying on remote host ===" -ForegroundColor Green
$deployScript = @"
set -e

echo 'Loading Docker image...'
docker load < /tmp/stance-classifier.tar.gz

echo 'Stopping old container...'
docker stop stance-classifier 2>/dev/null || true
docker rm stance-classifier 2>/dev/null || true

echo 'Starting new container...'
docker run -d \
  --name stance-classifier \
  --restart unless-stopped \
  -p 5000:5000 \
  -v /data/stance-detection/data:/app/data \
  -v /data/stance-detection/models:/app/models \
  -e MODEL_PATH=models/sentence_embedding_baseline \
  -e WORKERS=2 \
  stance-classifier:latest

echo 'Cleaning up...'
rm /tmp/stance-classifier.tar.gz

echo 'Container status:'
docker ps | grep stance-classifier

echo 'Container logs:'
docker logs --tail 20 stance-classifier
"@

ssh "${REMOTE_USER}@${REMOTE_HOST}" $deployScript

Write-Host "=== Cleaning up local files ===" -ForegroundColor Green
Remove-Item "$env:TEMP\${IMAGE_NAME}.tar.gz"

Write-Host "=== Deployment completed successfully! ===" -ForegroundColor Green
Write-Host "API available at: http://${REMOTE_HOST}:5000" -ForegroundColor Cyan
