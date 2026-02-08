# This file is kept for backwards compatibility
# The classifier is now initialized directly in elg_stance.py

import os
from StanceClassifier.stance_classifier import StanceClassifier

# Get model path from environment or use default
model_path = os.environ.get('MODEL_PATH', 'models/sentence_embedding_baseline')

# Initialize classifier
try:
    classifier = StanceClassifier(model_path=model_path)
except Exception as e:
    print(f"Warning: Could not initialize classifier: {e}")
    classifier = None

print(f"Classifier initialized with model: {model_path}")

