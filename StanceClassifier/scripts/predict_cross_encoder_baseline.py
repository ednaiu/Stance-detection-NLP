#!/usr/bin/env python3
"""
Predict stance using a trained Cross-Encoder + Logistic Regression model.

Loads a pre-trained cross-encoder and a trained LR classifier to make
stance predictions on new source-reply pairs.

Usage:
    python predict_cross_encoder_baseline.py \
        --model ./cross_encoder_baseline_model \
        --input-csv data.csv \
        --output-csv predictions.csv \
        --batch-size 32
"""

import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("[ERROR] Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict stance using Cross-Encoder baseline")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV with source_text and reply_text")
    parser.add_argument("--output-csv", required=True, help="Path to save predictions CSV")
    parser.add_argument("--source-col", default="source_text", help="Source text column name")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column name")
    parser.add_argument("--label-col", default=None, help="Label column name (if present, will compute metrics)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    return parser.parse_args()


def load_model(model_dir: str) -> Tuple[CrossEncoder, dict, dict]:
    """Load cross-encoder and LR classifier from model directory."""
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Load cross-encoder
    cross_encoder = CrossEncoder(config["cross_encoder_model"])

    # Load LR classifier
    with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
        lr_classifier = pickle.load(f)

    return cross_encoder, lr_classifier, config


def main():
    args = parse_args()

    # Load model
    print(f"[INFO] Loading model from {args.model}...")
    cross_encoder, lr_classifier, config = load_model(args.model)
    print(f"[INFO] Model loaded")
    print(f"[INFO] CrossEncoder: {config['cross_encoder_model']}")
    print(f"[INFO] Labels: {config['id2label']}")

    # Load input data
    print(f"\n[INFO] Loading input from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"[INFO] Loaded {len(df)} samples")

    sources = df[args.source_col].fillna("").astype(str).tolist()
    replies = df[args.reply_col].fillna("").astype(str).tolist()

    # Create pairs
    pairs = list(zip(sources, replies))

    # Get embeddings/scores for pairs
    print(f"\n[INFO] Getting cross-encoder scores for pairs...")
    scores = cross_encoder.predict(
        pairs,
        batch_size=args.batch_size,
    )
    # Reshape scores into feature format for LR
    embeddings = np.array(scores).reshape(-1, 1)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # Predict
    print(f"\n[INFO] Making predictions...")
    predictions = lr_classifier.predict(embeddings)
    probabilities = lr_classifier.predict_proba(embeddings)

    # Map predictions to label names
    id2label = config["id2label"]
    predicted_labels = [id2label[str(pred)] for pred in predictions]

    # Create output dataframe
    output_df = df.copy()
    output_df["predicted_label"] = predicted_labels
    output_df["predicted_id"] = predictions

    # Add probabilities for each class
    for class_id, class_name in id2label.items():
        output_df[f"prob_{class_name}"] = probabilities[:, int(class_id)]

    # Save predictions
    print(f"\n[INFO] Saving predictions to {args.output_csv}...")
    output_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Predictions saved")

    # Compute metrics if labels present
    if args.label_col and args.label_col in df.columns:
        print(f"\n[INFO] Computing metrics...")
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        # Normalize labels
        label2id = config["label2id"]
        lower_label2id = {k.lower(): v for k, v in label2id.items()}

        def normalize_label(value):
            if pd.isna(value):
                return None
            if isinstance(value, str):
                key = value.strip()
                if key in label2id:
                    return label2id[key]
                low = key.lower()
                if low in lower_label2id:
                    return lower_label2id[low]
            return int(value)

        true_labels = [normalize_label(v) for v in df[args.label_col].tolist()]
        true_labels = np.array([l for l in true_labels if l is not None], dtype=int)

        if len(true_labels) == len(predictions):
            acc = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
            print(f"\nAccuracy: {acc:.4f}")
            print(f"Macro-F1: {f1:.4f}")
            print(f"\n{classification_report(true_labels, predictions, target_names=list(id2label.values()), zero_division=0)}")
        else:
            print(f"[WARN] Label count mismatch: {len(true_labels)} vs {len(predictions)}")

    print(f"\n[INFO] Done")


if __name__ == "__main__":
    main()
