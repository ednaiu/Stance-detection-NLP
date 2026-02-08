# Stance Detection Docker Deployment

This directory contains Docker configuration for deploying the Stance Detection classifier as a containerized Flask API service.

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
cd docker
docker-compose up -d
```

Access the API at: `http://localhost:5000`

### 2. With Development Services (Jupyter included)

```bash
docker-compose --profile dev up -d
```

- API: `http://localhost:5000`
- Jupyter: `http://localhost:8888`

### 3. Using Docker CLI Only

```bash
# Build image
docker build -t stance-classifier:latest .

# Run container
docker run -d \
  --name stance-classifier \
  -p 5000:5000 \
  -v $(pwd)/../data:/app/data \
  -v $(pwd)/../models:/app/models \
  -e MODEL_PATH=models/sentence_embedding_baseline \
  stance-classifier:latest
```

## API Endpoints

### GET `/health`
Health check endpoint.

Response: `{"status": "healthy", "classifier_loaded": true}`

### GET `/`
API documentation and available endpoints.

### POST `/classify`
Classify a single text sample.

Request:
```json
{
  "text": "Your text to classify",
  "target": "Optional target text for context"
}
```

Response:
```json
{
  "stance": "support",
  "scores": {
    "support": 0.85,
    "deny": 0.10,
    "query": 0.03,
    "comment": 0.02
  }
}
```

### POST `/classify_batch`
Classify multiple text samples.

Request:
```json
{
  "samples": [
    {"text": "Text 1", "target": "Optional target"},
    {"text": "Text 2"}
  ]
}
```

Response:
```json
{
  "results": [
    {
      "text": "Text 1",
      "stance": "support",
      "scores": {...}
    },
    {
      "text": "Text 2",
      "stance": "deny",
      "scores": {...}
    }
  ]
}
```

## Environment Variables

- `MODEL_PATH`: Path to the model directory (default: `models/sentence_embedding_baseline`)
- `WORKERS`: Number of Gunicorn workers (default: `1`)
- `PYTHONUNBUFFERED`: Set to 1 for unbuffered output (default: `1`)
- `FLASK_ENV`: Flask environment mode (default: `production`)

## Management Commands

```bash
# View logs
docker-compose logs -f stance-classifier

# Stop services
docker-compose down

# Remove all containers and volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache
```

## Files

- `Dockerfile` - Container image specification
- `docker-compose.yml` - Multi-container orchestration
- `docker-entrypoint.sh` - Container startup script
- `elg_stance.py` - Flask API application
- `docker_classifier.py` - Classifier initialization (backwards compatibility)
- `.dockerignore` - Files to exclude from build context
