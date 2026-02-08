#!/usr/bin/env python3
"""
Make predictions using trained Sentence Embeddings + Logistic Regression model.

Reads a CSV file, encodes texts using pre-trained embeddings, and makes predictions.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make predictions using Sentence Embeddings + Logistic Regression model"
    )
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output-csv", default=None, help="Path to save predictions (if not specified, uses input_predictions.csv)")
    parser.add_argument("--source-col", default="source_text", help="Source text column name")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column name")
    parser.add_argument("--label-col", default=None, help="Label column name (optional, for computing metrics)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and metadata
    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "metadata.json"
    model_path = model_dir / "lr_model.pkl"

    if not metadata_path.exists() or not model_path.exists():
        print(f"[ERROR] Model files not found in {model_dir}")
        print(f"  Expected: {metadata_path} and {model_path}")
        exit(1)

    print(f"[INFO] Loading metadata from {metadata_path}...")
    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"[INFO] Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        lr_model = pickle.load(f)

    # Load embedding model
    embedding_model_name = metadata["embedding_model"]
    print(f"[INFO] Loading embedding model: {embedding_model_name}...")
    embedding_model = SentenceTransformer(embedding_model_name)

    # Load input data
    print(f"[INFO] Loading input data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"[INFO] Loaded {len(df)} samples")

    sources = df[args.source_col].fillna("").astype(str).tolist()
    replies = df[args.reply_col].fillna("").astype(str).tolist()

    # Prepare texts based on mode
    mode = metadata["mode"]
    if mode == "aware":
        texts = [f"{src} [SEP] {rep}" for src, rep in zip(sources, replies)]
    else:  # oblivious
        texts = replies

    # Encode texts
    print(f"[INFO] Encoding {len(texts)} texts (batch_size={args.batch_size})...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # Make predictions
    print(f"[INFO] Making predictions...")
    predictions = lr_model.predict(embeddings)
    probabilities = lr_model.predict_proba(embeddings)

    # Map IDs to labels
    id2label = metadata["id2label"]
    labels_list = metadata["labels"]
    pred_labels = [id2label[str(pred_id)] for pred_id in predictions]

    # Create output dataframe
    output_df = df.copy()
    output_df["pred_id"] = predictions
    output_df["pred_label"] = pred_labels

    # Add probabilities for each class
    for i, label in enumerate(labels_list):
        output_df[f"prob_{label}"] = probabilities[:, i]

    # Compute metrics if labels are provided
    if args.label_col and args.label_col in df.columns:
        print(f"\n[INFO] Computing metrics...")
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        label2id = metadata["label2id"]
        true_labels = []

        for raw_label in df[args.label_col]:
            if pd.isna(raw_label):
                print(f"[WARN] Skipping row with missing label")
                continue
            label_str = str(raw_label).strip().lower()
            for orig_label, idx in label2id.items():
                if orig_label.lower() == label_str:
                    true_labels.append(idx)
                    break
            else:
                print(f"[WARN] Unknown label: {raw_label}")
                true_labels.append(-1)

        if true_labels:
            valid_mask = np.array(true_labels) >= 0
            valid_true = np.array(true_labels)[valid_mask]
            valid_preds = predictions[valid_mask]

            accuracy = accuracy_score(valid_true, valid_preds)
            macro_f1 = f1_score(valid_true, valid_preds, average="macro", zero_division=0)

            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(valid_true, valid_preds, target_names=labels_list, zero_division=0))

    # Save predictions
    if args.output_csv is None:
        base_name = Path(args.input_csv).stem
        args.output_csv = f"{base_name}_predictions.csv"

    output_df.to_csv(args.output_csv, index=False)
    print(f"\n[INFO] ✅ Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
