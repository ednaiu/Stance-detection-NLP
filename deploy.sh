#!/bin/bash
# Deployment script for stance detection project
# Usage: ./deploy.sh

set -e

REMOTE_HOST="31.31.198.9"
REMOTE_USER="${REMOTE_USER:-root}"
IMAGE_NAME="stance-classifier"
CONTAINER_NAME="stance-classifier"

echo "=== Building Docker image ==="
cd StanceClassifier
docker build -t ${IMAGE_NAME}:latest -f docker/Dockerfile .

echo "=== Saving Docker image ==="
docker save ${IMAGE_NAME}:latest | gzip > /tmp/${IMAGE_NAME}.tar.gz

echo "=== Copying image to remote host ${REMOTE_HOST} ==="
scp /tmp/${IMAGE_NAME}.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:/tmp/

echo "=== Deploying on remote host ==="
ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
  set -e
  
  echo "Loading Docker image..."
  docker load < /tmp/stance-classifier.tar.gz
  
  echo "Stopping old container..."
  docker stop stance-classifier 2>/dev/null || true
  docker rm stance-classifier 2>/dev/null || true
  
  echo "Starting new container..."
  docker run -d \
    --name stance-classifier \
    --restart unless-stopped \
    -p 5000:5000 \
    -v /data/stance-detection/data:/app/data \
    -v /data/stance-detection/models:/app/models \
    -e MODEL_PATH=models/sentence_embedding_baseline \
    -e WORKERS=2 \
    stance-classifier:latest
  
  echo "Cleaning up..."
  rm /tmp/stance-classifier.tar.gz
  
  echo "Container status:"
  docker ps | grep stance-classifier
  
  echo "Container logs:"
  docker logs --tail 20 stance-classifier
EOF

echo "=== Cleaning up local files ==="
rm /tmp/${IMAGE_NAME}.tar.gz

echo "=== Deployment completed successfully! ==="
echo "API available at: http://${REMOTE_HOST}:5000"
