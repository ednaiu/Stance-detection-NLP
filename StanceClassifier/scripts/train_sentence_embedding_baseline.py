#!/usr/bin/env python3
"""
Train a Sentence Embeddings + Logistic Regression baseline model for stance detection.

Uses pre-trained sentence transformers to encode text, then trains a simple
logistic regression classifier on top of the embeddings.

This combines semantic understanding with interpretability of LR.

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
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


DEFAULT_LABELS = ["support", "deny", "query", "comment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Sentence Embeddings + Logistic Regression baseline for stance detection"
    )
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    parser.add_argument("--test-csv", default=None, help="Path to test CSV")
    parser.add_argument(
        "--mode",
        choices=["aware", "oblivious"],
        default="aware",
        help="aware: concatenate source+reply; oblivious: reply only",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument("--output-dir", default="./sentence_embedding_baseline_model", help="Output model directory")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label order (defines label IDs)",
    )
    parser.add_argument("--source-col", default="source_text", help="Source text column name")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", None],
        default=None,
        help="Use balanced class weights to handle imbalanced data",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings")
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
    mode: str = "aware",
) -> Tuple[List[str], np.ndarray]:
    """Load a CSV split and return texts + label IDs"""

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

    # Prepare texts based on mode
    if mode == "aware":
        texts = [f"{src} [SEP] {rep}" for src, rep in zip(sources, replies)]
    else:  # oblivious
        texts = replies

    # Normalize labels
    lower_label2id = {k.lower(): v for k, v in label2id.items()}
    labels = np.array(
        [_normalize_label(raw_label, label2id, lower_label2id) for raw_label in raw_labels],
        dtype=int,
    )

    return texts, labels


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Prepare labels
    labels_list = [l.strip() for l in args.labels.split(",")]
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"[INFO] Label mapping: {label2id}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Embedding model: {args.embedding_model}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Regularization C: {args.C}")
    print(f"[INFO] Class weight: {args.class_weight or 'None'}")

    # Load embedding model
    print(f"\n[INFO] Loading embedding model: {args.embedding_model}...")
    embedding_model = SentenceTransformer(args.embedding_model)
    print(f"[INFO] Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

    # Load training data
    print(f"\n[INFO] Loading training data from {args.train_csv}...")
    train_texts, train_labels = load_split(
        args.train_csv,
        args.source_col,
        args.reply_col,
        args.label_col,
        label2id,
        args.mode,
    )
    print(f"[INFO] Loaded {len(train_texts)} training samples")
    print(f"[INFO] Label distribution: {np.bincount(train_labels)}")

    # Load validation data if provided
    val_texts, val_labels = None, None
    if args.val_csv:
        print(f"\n[INFO] Loading validation data from {args.val_csv}...")
        val_texts, val_labels = load_split(
            args.val_csv,
            args.source_col,
            args.reply_col,
            args.label_col,
            label2id,
            args.mode,
        )
        print(f"[INFO] Loaded {len(val_texts)} validation samples")
        print(f"[INFO] Label distribution: {np.bincount(val_labels)}")

    # Load test data if provided
    test_texts, test_labels = None, None
    if args.test_csv:
        print(f"\n[INFO] Loading test data from {args.test_csv}...")
        test_texts, test_labels = load_split(
            args.test_csv,
            args.source_col,
            args.reply_col,
            args.label_col,
            label2id,
            args.mode,
        )
        print(f"[INFO] Loaded {len(test_texts)} test samples")
        print(f"[INFO] Label distribution: {np.bincount(test_labels)}")

    # Encode training texts
    print(f"\n[INFO] Encoding training texts (batch_size={args.batch_size})...")
    train_embeddings = embedding_model.encode(
        train_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[INFO] Train embeddings shape: {train_embeddings.shape}")

    # Train Logistic Regression
    print(f"\n[INFO] Training Logistic Regression model...")
    class_weight = "balanced" if args.class_weight == "balanced" else None
    lr_model = LogisticRegression(
        C=args.C,
        max_iter=1000,
        random_state=args.seed,
        n_jobs=-1,
        solver="lbfgs",
        class_weight=class_weight,
    )
    lr_model.fit(train_embeddings, train_labels)
    print(f"[INFO] Model training complete")

    # Evaluate on training data
    print(f"\n[INFO] === TRAINING SET ===")
    train_preds = lr_model.predict(train_embeddings)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    print(f"Accuracy: {train_acc:.4f}")
    print(f"Macro F1: {train_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(train_labels, train_preds, target_names=labels_list))

    # Evaluate on validation data if provided
    if val_texts is not None:
        print(f"\n[INFO] === VALIDATION SET ===")
        val_embeddings = embedding_model.encode(
            val_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        val_preds = lr_model.predict(val_embeddings)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Macro F1: {val_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(val_labels, val_preds, target_names=labels_list))

    # Evaluate on test data if provided
    if test_texts is not None:
        print(f"\n[INFO] === TEST SET ===")
        test_embeddings = embedding_model.encode(
            test_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        test_preds = lr_model.predict(test_embeddings)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Macro F1: {test_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=labels_list))

    # Save model and metadata
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, "lr_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"\n[INFO] Saved Logistic Regression model to {model_path}")

    metadata = {
        "model_type": "sentence_embedding_baseline",
        "embedding_model": args.embedding_model,
        "embedding_dim": embedding_model.get_sentence_embedding_dimension(),
        "labels": labels_list,
        "label2id": label2id,
        "id2label": id2label,
        "mode": args.mode,
        "C": args.C,
        "class_weight": args.class_weight,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to {metadata_path}")

    print(f"\n[INFO] ✅ Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
