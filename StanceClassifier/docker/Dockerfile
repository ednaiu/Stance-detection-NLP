FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      git \
      wget \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project structure
COPY StanceClassifier /app/StanceClassifier
COPY docker/elg_stance.py docker/docker_classifier.py /app/
COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh

# Create necessary directories (data and models will be mounted as volumes)
RUN mkdir -p /app/data/processed /app/data/raw /app/models /app/scripts /app/logs /app/setup.d && \
    chmod +x /app/docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports
EXPOSE 5000 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:5000/health || exit 1

# Entry point
ENTRYPOINT ["/app/docker-entrypoint.sh"]
