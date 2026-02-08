#!/usr/bin/env python3
"""
WSGI entry point for Flask application
Compatible with gunicorn, uWSGI, and other WSGI servers
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'StanceClassifier'))

# Import Flask app from elg_stance.py
try:
    # Try to import from docker directory first
    sys.path.insert(0, os.path.join(project_root, 'StanceClassifier', 'docker'))
    from elg_stance import app, init_classifier
except ImportError:
    # Fallback to direct import
    from StanceClassifier.docker.elg_stance import app, init_classifier

# Initialize classifier when app starts
init_classifier()

# The 'app' object is what WSGI servers look for
if __name__ == '__main__':
    # For local testing only
    app.run(host='0.0.0.0', port=5000, debug=False)
