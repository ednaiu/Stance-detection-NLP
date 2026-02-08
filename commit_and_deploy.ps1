# Automated commit and deploy script
# Usage: .\commit_and_deploy.ps1 "Commit message"

param(
    [string]$Message = "Auto update"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Git Commit and Push ===" -ForegroundColor Green

# Check for changes
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No changes to commit" -ForegroundColor Yellow
} else {
    # Add all changes
    Write-Host "Adding changes..." -ForegroundColor Cyan
    git add -A
    
    # Commit
    Write-Host "Committing with message: $Message" -ForegroundColor Cyan
    git commit -m $Message
    
    # Push to GitHub
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    git push origin main
    
    Write-Host "Changes pushed successfully!" -ForegroundColor Green
}

# Deploy to remote host
Write-Host "`n=== Starting Deployment ===" -ForegroundColor Green
.\deploy.ps1

Write-Host "`n=== Pipeline completed! ===" -ForegroundColor Green
