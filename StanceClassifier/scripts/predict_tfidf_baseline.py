#!/usr/bin/env python3
"""
Run predictions using TF-IDF + Logistic Regression baseline model.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Predict stance labels using TF-IDF baseline")
    parser.add_argument("--model-dir", required=True, help="Path to model directory (containing .pkl files)")
    parser.add_argument("--input-csv", required=True, help="Input CSV path")
    parser.add_argument("--output-csv", default="tfidf_predictions.csv", help="Output CSV path")
    parser.add_argument(
        "--mode",
        choices=["aware", "oblivious"],
        default="aware",
        help="aware: concatenate source+reply; oblivious: reply only",
    )
    parser.add_argument("--source-col", default="source_text", help="Source text column")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column")
    parser.add_argument("--label-col", default="label", help="Gold label column (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load metadata
    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    labels_list = metadata["labels"]
    id2label = {int(k): v for k, v in metadata["id2label"].items()}
    mode = metadata.get("mode", "aware")

    print(f"[INFO] Loaded model metadata")
    print(f"[INFO] Labels: {labels_list}")
    print(f"[INFO] Mode: {mode}")

    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(args.model_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    print(f"[INFO] Loaded TF-IDF vectorizer")

    # Load Logistic Regression model
    model_path = os.path.join(args.model_dir, "lr_model.pkl")
    with open(model_path, "rb") as f:
        lr_model = pickle.load(f)
    print(f"[INFO] Loaded Logistic Regression model")

    # Load input CSV
    df = pd.read_csv(args.input_csv)
    print(f"[INFO] Loaded {len(df)} samples from {args.input_csv}")

    for col in [args.reply_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.input_csv}")

    # Prepare texts
    sources = df[args.source_col].fillna("").astype(str).tolist() if args.source_col in df.columns else [""] * len(df)
    replies = df[args.reply_col].fillna("").astype(str).tolist()

    if mode == "aware":
        texts = [f"{src} [SEP] {rep}" for src, rep in zip(sources, replies)]
    else:
        texts = replies

    # Vectorize
    X = tfidf.transform(texts)
    print(f"[INFO] Vectorized {len(texts)} samples")

    # Predict
    predictions = lr_model.predict(X)
    probabilities = lr_model.predict_proba(X)
    
    # Convert to label names
    pred_labels = [id2label[pred] for pred in predictions]

    # Add predictions to dataframe
    df["pred_label"] = pred_labels
    df["pred_class"] = predictions

    # Add probabilities for each class
    for class_id, label in id2label.items():
        df[f"prob_{label}"] = probabilities[:, class_id]

    # Calculate metrics if gold labels are available
    if args.label_col in df.columns:
        gold_labels = df[args.label_col].tolist()
        # Try to convert to numeric if needed
        label2id = {v: k for k, v in id2label.items()}
        
        try:
            gold_ids = []
            for label in gold_labels:
                if isinstance(label, (int, np.integer)):
                    gold_ids.append(int(label))
                elif isinstance(label, str):
                    label_clean = label.strip().lower()
                    # Find matching label (case-insensitive)
                    found = False
                    for orig_label, lid in label2id.items():
                        if orig_label.lower() == label_clean:
                            gold_ids.append(lid)
                            found = True
                            break
                    if not found:
                        print(f"[WARN] Could not map label '{label}' to class ID")
                        gold_ids.append(-1)
                else:
                    gold_ids.append(-1)
            
            # Filter out unmappable labels
            valid_idx = [i for i, gid in enumerate(gold_ids) if gid >= 0]
            if valid_idx:
                valid_preds = [predictions[i] for i in valid_idx]
                valid_gold = [gold_ids[i] for i in valid_idx]
                
                accuracy = accuracy_score(valid_gold, valid_preds)
                f1 = f1_score(valid_gold, valid_preds, average="macro")
                
                print(f"\n[INFO] === METRICS ===")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Macro F1: {f1:.4f}")
                print("\nClassification Report:")
                print(classification_report(valid_gold, valid_preds, target_names=labels_list))
        except Exception as e:
            print(f"[WARN] Could not compute metrics: {e}")

    # Save predictions
    df.to_csv(args.output_csv, index=False)
    print(f"\n[INFO] Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
