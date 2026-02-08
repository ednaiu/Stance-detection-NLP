# Deployment Pipeline Documentation

## Overview

This project includes automated deployment pipeline that:
1. Commits and pushes changes to GitHub
2. Builds Docker image
3. Deploys to remote host (31.31.198.9)

## Quick Start

### Windows (PowerShell)

```powershell
# Full pipeline: commit + deploy
.\commit_and_deploy.ps1 "Your commit message"

# Deploy only
.\deploy.ps1
```

### Linux/Mac (Bash)

```bash
# Make script executable
chmod +x deploy.sh

# Deploy
./deploy.sh
```

## GitHub Actions (Automatic Deployment)

The project includes GitHub Actions workflow that automatically deploys when you push to `main` branch.

### Setup GitHub Secrets

Go to your repository settings → Secrets and variables → Actions, and add:

1. **REMOTE_HOST**: `31.31.198.9`
2. **REMOTE_USER**: SSH username (e.g., `root`)
3. **SSH_PRIVATE_KEY**: Your SSH private key for authentication

### Manual Trigger

You can also manually trigger deployment:
1. Go to GitHub → Actions tab
2. Select "Deploy to Remote Host" workflow
3. Click "Run workflow"

## Manual Deployment Steps

### 1. Commit Changes

```powershell
git add -A
git commit -m "Your changes"
git push origin main
```

### 2. Build Docker Image

```powershell
cd StanceClassifier
docker build -t stance-classifier:latest -f docker/Dockerfile .
```

### 3. Deploy to Remote Host

#### Option A: Using deployment script (recommended)
```powershell
.\deploy.ps1
```

#### Option B: Manual steps
```powershell
# Save image
docker save stance-classifier:latest | gzip > stance-classifier.tar.gz

# Copy to remote host
scp stance-classifier.tar.gz user@31.31.198.9:/tmp/

# SSH to remote host and deploy
ssh user@31.31.198.9
docker load < /tmp/stance-classifier.tar.gz
docker stop stance-classifier || true
docker rm stance-classifier || true
docker run -d --name stance-classifier --restart unless-stopped -p 5000:5000 stance-classifier:latest
```

## Remote Host Configuration

### Prerequisites on Remote Host (31.31.198.9)

1. **Docker installed**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **SSH access configured**
   - Ensure SSH key authentication is set up
   - User should have Docker permissions

3. **Data directories created**
   ```bash
   mkdir -p /data/stance-detection/{data,models}
   ```

### Port Configuration

- API port: `5000`
- Jupyter port: `8888` (if dev profile enabled)

### Firewall Rules

Ensure ports are open:
```bash
sudo ufw allow 5000/tcp
sudo ufw allow 8888/tcp  # For Jupyter (optional)
```

## Environment Variables

On remote host, you can customize:

```bash
docker run -d \
  --name stance-classifier \
  -p 5000:5000 \
  -v /data/stance-detection/data:/app/data \
  -v /data/stance-detection/models:/app/models \
  -e MODEL_PATH=models/sentence_embedding_baseline \
  -e WORKERS=2 \
  -e FLASK_ENV=production \
  stance-classifier:latest
```

## Monitoring

### Check deployment status
```bash
ssh user@31.31.198.9 "docker ps | grep stance-classifier"
```

### View logs
```bash
ssh user@31.31.198.9 "docker logs -f stance-classifier"
```

### Test API
```bash
curl http://31.31.198.9:5000/health
```

## Troubleshooting

### SSH Connection Issues
```powershell
# Test SSH connection
ssh user@31.31.198.9 "echo 'Connection successful'"

# Check SSH key
ssh-add -l
```

### Docker Build Issues
```powershell
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t stance-classifier:latest .
```

### Container Not Starting
```bash
# Check logs
docker logs stance-classifier

# Check if port is in use
netstat -tulpn | grep 5000

# Remove and recreate
docker rm -f stance-classifier
```

## Rollback

To rollback to previous version:

```bash
ssh user@31.31.198.9
docker stop stance-classifier
docker rm stance-classifier
docker run -d --name stance-classifier stance-classifier:previous
```

## Security Notes

1. **SSH Keys**: Keep SSH private keys secure, never commit to repository
2. **GitHub Secrets**: Use GitHub Secrets for sensitive data
3. **Firewall**: Only expose necessary ports
4. **Updates**: Regularly update Docker and system packages
5. **Access Control**: Limit SSH access to authorized users only

## Support

For issues or questions:
- Check logs: `docker logs stance-classifier`
- GitHub Issues: https://github.com/ednaiu/Stance-detection/issues
