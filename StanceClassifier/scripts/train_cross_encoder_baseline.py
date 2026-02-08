#!/usr/bin/env python3
"""
Train a Cross-Encoder baseline model for stance detection.

Cross-Encoders directly encode source-reply pairs jointly to better capture
the interaction and relationship between texts. This is more powerful than
separate sentence embeddings but slightly slower.

The model learns to directly score the relationship between pairs,
then uses logistic regression on top.

Expected CSV columns by default:
  - source_text
  - reply_text
  - label   (string labels or numeric IDs)
"""

import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("[ERROR] Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


DEFAULT_LABELS = ["support", "deny", "query", "comment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Cross-Encoder baseline for stance detection"
    )
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    parser.add_argument("--test-csv", default=None, help="Path to test CSV")
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/qnli-distilroberta-base",
        help="Pre-trained CrossEncoder model (default: cross-encoder/qnli-distilroberta-base)",
    )
    parser.add_argument("--output-dir", default="./cross_encoder_baseline_model", help="Output model directory")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label order (defines label IDs)",
    )
    parser.add_argument("--source-col", default="source_text", help="Source text column name")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LR")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", None],
        default=None,
        help="Use balanced class weights to handle imbalanced data",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for cross-encoder encoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _normalize_label(value, label2id, lower_label2id) -> int:
    if pd.isna(value):
        raise ValueError("Missing label (NaN)")

    if isinstance(value, str):
        key = value.strip()
        if not key:
            raise ValueError("Missing label (empty string)")
        if key in label2id:
            return label2id[key]
        low = key.lower()
        if low in lower_label2id:
            return lower_label2id[low]

        try:
            num = float(key)
            if num.is_integer():
                idx = int(num)
                if 0 <= idx < len(label2id):
                    return idx
        except ValueError:
            pass

        raise ValueError(f"Unknown label: {value}")
    return int(value)


def load_split(
    csv_path: str,
    source_col: str,
    reply_col: str,
    label_col: str,
    label2id,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Load a CSV split and return pairs + label IDs.
    
    Returns:
        pairs: List of (source, reply) tuples
        labels: Array of label IDs
    """
    df = pd.read_csv(csv_path)

    # Check for missing labels
    missing_mask = df[label_col].isna()
    if missing_mask.any():
        missing_count = missing_mask.sum()
        missing_rows = df[missing_mask].index.tolist()[:10]
        print(
            f"[WARN] {csv_path}: dropping {missing_count} rows with missing '{label_col}'. "
            f"Example CSV line numbers: {missing_rows}"
        )
        df = df.loc[~missing_mask].copy()
        if df.empty:
            raise ValueError(f"All rows in {csv_path} have missing '{label_col}'")

    sources = df[source_col].fillna("").astype(str).tolist()
    replies = df[reply_col].fillna("").astype(str).tolist()
    raw_labels = df[label_col].tolist()

    # Create pairs (CrossEncoder expects list of [text1, text2] pairs)
    pairs = list(zip(sources, replies))

    # Normalize labels
    lower_label2id = {k.lower(): v for k, v in label2id.items()}
    labels = np.array(
        [_normalize_label(raw_label, label2id, lower_label2id) for raw_label in raw_labels],
        dtype=int,
    )

    return pairs, labels


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Prepare labels
    labels_list = [l.strip() for l in args.labels.split(",")]
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"[INFO] Label mapping: {label2id}")
    print(f"[INFO] CrossEncoder model: {args.cross_encoder_model}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Regularization C: {args.C}")
    print(f"[INFO] Class weight: {args.class_weight or 'None'}")

    # Load cross-encoder model
    print(f"\n[INFO] Loading cross-encoder model: {args.cross_encoder_model}...")
    cross_encoder = CrossEncoder(args.cross_encoder_model)
    print(f"[INFO] Model loaded successfully")

    # Load training data
    print(f"\n[INFO] Loading training data from {args.train_csv}...")
    train_pairs, train_labels = load_split(
        args.train_csv,
        args.source_col,
        args.reply_col,
        args.label_col,
        label2id,
    )
    print(f"[INFO] Loaded {len(train_pairs)} training samples")
    print(f"[INFO] Label distribution: {np.bincount(train_labels)}")

    # Load validation data if provided
    val_pairs, val_labels = None, None
    if args.val_csv:
        print(f"\n[INFO] Loading validation data from {args.val_csv}...")
        val_pairs, val_labels = load_split(
            args.val_csv,
            args.source_col,
            args.reply_col,
            args.label_col,
            label2id,
        )
        print(f"[INFO] Loaded {len(val_pairs)} validation samples")
        print(f"[INFO] Label distribution: {np.bincount(val_labels)}")

    # Load test data if provided
    test_pairs, test_labels = None, None
    if args.test_csv:
        print(f"\n[INFO] Loading test data from {args.test_csv}...")
        test_pairs, test_labels = load_split(
            args.test_csv,
            args.source_col,
            args.reply_col,
            args.label_col,
            label2id,
        )
        print(f"[INFO] Loaded {len(test_pairs)} test samples")
        print(f"[INFO] Label distribution: {np.bincount(test_labels)}")

    # Get embeddings using cross-encoder's token embeddings
    # We'll use the cross-encoder to get embeddings for each pair
    print(f"\n[INFO] Getting embeddings for training pairs with cross-encoder...")
    
    # Method: Use the underlying transformer to get embeddings, then average them
    # Or use predict scores from the cross-encoder model as features
    # For simplicity, we'll use the cross-encoder's logits as features for LR
    train_scores = cross_encoder.predict(train_pairs, batch_size=args.batch_size)
    # Reshape scores into a format suitable for LR (treating logits as features)
    train_embeddings = np.array(train_scores).reshape(-1, 1)
    print(f"[INFO] Training embeddings shape: {train_embeddings.shape}")

    # Get scores for validation pairs if available
    val_embeddings = None
    if val_pairs:
        print(f"[INFO] Getting scores for validation pairs with cross-encoder...")
        val_scores = cross_encoder.predict(val_pairs, batch_size=args.batch_size)
        val_embeddings = np.array(val_scores).reshape(-1, 1)
        print(f"[INFO] Validation embeddings shape: {val_embeddings.shape}")

    # Get scores for test pairs if available
    test_embeddings = None
    if test_pairs:
        print(f"[INFO] Getting scores for test pairs with cross-encoder...")
        test_scores = cross_encoder.predict(test_pairs, batch_size=args.batch_size)
        test_embeddings = np.array(test_scores).reshape(-1, 1)
        print(f"[INFO] Test embeddings shape: {test_embeddings.shape}")

    # Train logistic regression on cross-encoder embeddings
    print(f"\n[INFO] Training Logistic Regression classifier...")
    lr_classifier = LogisticRegression(
        max_iter=1000,
        C=args.C,
        class_weight=args.class_weight,
        solver="lbfgs" if len(label2id) <= 4 else "sag",
        random_state=args.seed,
        multi_class="multinomial",
    )
    lr_classifier.fit(train_embeddings, train_labels)
    print(f"[INFO] Training complete")

    # Evaluate on training set
    print(f"\n[INFO] ========== TRAINING SET METRICS ==========")
    train_preds = lr_classifier.predict(train_embeddings)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
    print(f"Accuracy: {train_acc:.4f}")
    print(f"Macro-F1: {train_f1:.4f}")
    print(f"\n{classification_report(train_labels, train_preds, target_names=labels_list, zero_division=0)}")

    # Evaluate on validation set
    if val_embeddings is not None:
        print(f"\n[INFO] ========== VALIDATION SET METRICS ==========")
        val_preds = lr_classifier.predict(val_embeddings)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Macro-F1: {val_f1:.4f}")
        print(f"\n{classification_report(val_labels, val_preds, target_names=labels_list, zero_division=0)}")

    # Evaluate on test set
    if test_embeddings is not None:
        print(f"\n[INFO] ========== TEST SET METRICS ==========")
        test_preds = lr_classifier.predict(test_embeddings)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Macro-F1: {test_f1:.4f}")
        print(f"\n{classification_report(test_labels, test_preds, target_names=labels_list, zero_division=0)}")

    # Save model artifacts
    print(f"\n[INFO] Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save cross-encoder model name (so we know which one to load for inference)
    config = {
        "cross_encoder_model": args.cross_encoder_model,
        "label2id": label2id,
        "id2label": id2label,
        "embedding_dim": train_embeddings.shape[1],
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save LR classifier
    with open(os.path.join(args.output_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(lr_classifier, f)

    print(f"[INFO] Model saved to {args.output_dir}")
    print(f"  - config.json: CrossEncoder model name and label mappings")
    print(f"  - classifier.pkl: Trained Logistic Regression classifier")

    # Print final summary
    print(f"\n[INFO] ========== SUMMARY ==========")
    print(f"Training samples: {len(train_pairs)}")
    if val_pairs:
        print(f"Validation samples: {len(val_pairs)}")
    if test_pairs:
        print(f"Test samples: {len(test_pairs)}")
    print(f"Embedding dimension: {train_embeddings.shape[1]}")
    print(f"Number of classes: {len(label2id)}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
