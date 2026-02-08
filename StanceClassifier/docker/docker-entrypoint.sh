#!/bin/sh

# Entry point for the stance detection container
# Starts the Flask API server

set -e

# Allow for setup shell scripts to be injected
for f in /app/setup.d/*.sh ; do
  if [ -r "$f" ]; then
    . "$f"
  fi
done

# Start Flask application with Gunicorn
exec python -m gunicorn \
  --bind 0.0.0.0:5000 \
  --workers "${WORKERS:-1}" \
  --worker-class sync \
  --timeout 300 \
  --access-logfile - \
  --error-logfile - \
  elg_stance:app
