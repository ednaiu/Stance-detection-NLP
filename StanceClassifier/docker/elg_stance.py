#!/usr/bin/env python3
"""
Stance Detection Flask API
Provides endpoints for stance classification of text samples
"""

from flask import Flask, request, jsonify
import os
import json
import logging
from typing import Dict, List, Any

# Try to import the stance classifier
try:
    from StanceClassifier.stance_classifier import StanceClassifier
except ImportError:
    try:
        from stance_classifier import StanceClassifier
    except ImportError:
        print("Warning: Could not import StanceClassifier")
        StanceClassifier = None

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize stance classifier
classifier = None
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/sentence_embedding_baseline')

def init_classifier():
    """Initialize the stance classifier"""
    global classifier
    if StanceClassifier is not None and classifier is None:
        try:
            classifier = StanceClassifier(model_path=MODEL_PATH)
            logger.info(f"Classifier initialized with model: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            classifier = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier is not None
    }), 200


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify stance of text samples.
    
    Request format:
    {
        "text": "text to classify",
        "target": "optional target text"  # If provided, uses context-aware classification
    }
    
    Response format:
    {
        "stance": "support/deny/query/comment",
        "scores": {
            "support": 0.8,
            "deny": 0.1,
            "query": 0.05,
            "comment": 0.05
        }
    }
    """
    if classifier is None:
        init_classifier()
    
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        target = data.get('target', None)
        
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Perform classification
        try:
            if target:
                prediction, scores = classifier.predict(text, target)
            else:
                prediction, scores = classifier.predict(text)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        return jsonify({
            'stance': prediction,
            'scores': {
                'support': float(scores[0]) if hasattr(scores[0], 'item') else float(scores[0]),
                'deny': float(scores[1]) if hasattr(scores[1], 'item') else float(scores[1]),
                'query': float(scores[2]) if hasattr(scores[2], 'item') else float(scores[2]),
                'comment': float(scores[3]) if hasattr(scores[3], 'item') else float(scores[3]),
            }
        }), 200
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple text samples in batch.
    
    Request format:
    {
        "samples": [
            {"text": "text1", "target": "optional"},
            {"text": "text2"}
        ]
    }
    """
    if classifier is None:
        init_classifier()
    
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'Missing required field: samples'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({'error': 'samples must be a non-empty list'}), 400
        
        results = []
        for idx, sample in enumerate(samples):
            if 'text' not in sample:
                results.append({'error': f'Sample {idx}: missing text field'})
                continue
            
            text = sample['text']
            target = sample.get('target', None)
            
            try:
                if target:
                    prediction, scores = classifier.predict(text, target)
                else:
                    prediction, scores = classifier.predict(text)
                
                results.append({
                    'text': text,
                    'stance': prediction,
                    'scores': {
                        'support': float(scores[0]) if hasattr(scores[0], 'item') else float(scores[0]),
                        'deny': float(scores[1]) if hasattr(scores[1], 'item') else float(scores[1]),
                        'query': float(scores[2]) if hasattr(scores[2], 'item') else float(scores[2]),
                        'comment': float(scores[3]) if hasattr(scores[3], 'item') else float(scores[3]),
                    }
                })
            except Exception as e:
                logger.error(f"Error classifying sample {idx}: {e}")
                results.append({'error': f'Classification failed: {str(e)}'})
        
        return jsonify({'results': results}), 200
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'name': 'Stance Detection API',
        'version': '1.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/classify': 'POST - Single text classification',
            '/classify_batch': 'POST - Batch text classification'
        },
        'example': {
            'classify': {
                'url': '/classify',
                'method': 'POST',
                'body': {'text': 'Your text here', 'target': 'Optional target text'}
            }
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    init_classifier()
    app.run(host='0.0.0.0', port=5000, debug=False)
