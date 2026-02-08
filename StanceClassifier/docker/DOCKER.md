# Docker Deployment Guide

## Project Structure

```
docker/
├── Dockerfile              # Main Docker image configuration
├── Dockerfile.new          # Alternative modern Dockerfile
├── docker-compose.yml      # Docker Compose orchestration
├── .dockerignore           # Files to exclude from Docker build
├── docker-build.sh         # Quick build and run script
├── DOCKER.md               # This file
├── build.sh                # Original build script
├── docker-entrypoint.sh    # Container entry point
├── docker_classifier.py    # Flask API implementation
├── elg_stance.py           # ELG integration
└── cloud-application/      # Cloud deployment configuration
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
cd docker
docker-compose up -d

# With development services (includes Jupyter)
docker-compose --profile dev up -d

# View logs
docker-compose logs -f stance-classifier

# Stop services
docker-compose down
```

### Option 2: Using Docker CLI

```bash
cd docker

# Build image
docker build -t stance-classifier .

# Run container
docker run -d \
  --name stance-classifier \
  -p 5000:5000 \
  -p 8888:8888 \
  -v $(pwd)/../data:/app/data \
  -v $(pwd)/../models:/app/models \
  stance-classifier
```

### Option 3: Using Build Script

```bash
cd docker
chmod +x docker-build.sh
./docker-build.sh
```

## Access Services

- **Flask API**: http://localhost:5000
- **Jupyter Notebook**: http://localhost:8888 (if using dev profile)

## Available Commands

### Run the Stance Classifier on a single file
```bash
docker run --rm -v $(pwd)/examples:/app/examples stance-classifier \
  python -m StanceClassifier /app/examples/reply_new /app/examples/original_new
```

### Run Jupyter in container
```bash
docker run -it --rm \
  -p 8888:8888 \
  -v $(pwd):/app \
  stance-classifier \
  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

## Building with Custom Python Version

Edit the `FROM` line in Dockerfile:
- `python:3.9-slim` for Python 3.9
- `python:3.10-slim` for Python 3.10
- `python:3.11-slim` for Python 3.11 (default)
- `python:3.12-slim` for Python 3.12

## Troubleshooting

### Port already in use
```bash
# Use different ports
docker run -p 5001:5000 -p 8889:8888 stance-classifier
```

### Memory issues with transformers
Add memory limits:
```bash
docker run -m 4g stance-classifier
```

### GPU Support
If you have NVIDIA GPU and want to use it:

1. Install nvidia-docker: https://github.com/NVIDIA/nvidia-docker
2. Use GPU-enabled image in Dockerfile: `pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04`
3. Run with GPU:
```bash
docker run --gpus all stance-classifier
```

## Clean Up

```bash
# Remove container
docker rm stance-classifier

# Remove image
docker rmi stance-classifier

# Remove all unused Docker resources
docker system prune -a
```
