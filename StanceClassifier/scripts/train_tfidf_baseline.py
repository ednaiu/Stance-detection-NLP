#!/usr/bin/env python3
"""
Train a TF-IDF + Logistic Regression baseline model for stance detection.

This is a lightweight, interpretable baseline that uses lexical features.

Expected CSV columns by default:
  - sourceText
  - replyText
  - label   (string labels or numeric IDs)
"""

import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)


DEFAULT_LABELS = ["support", "deny", "query", "comment"]


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression baseline for stance detection")
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    parser.add_argument("--test-csv", default=None, help="Path to test CSV")
    parser.add_argument(
        "--mode",
        choices=["aware", "oblivious"],
        default="aware",
        help="aware: concatenate source+reply; oblivious: reply only",
    )
    parser.add_argument("--output-dir", default="./tfidf_baseline_model", help="Output model directory")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label order (defines label IDs)",
    )
    parser.add_argument("--source-col", default="sourceText", help="Source text column name")
    parser.add_argument("--reply-col", default="replyText", help="Reply text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--max-features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--ngram-range", type=str, default="1,2", help="N-gram range (e.g., '1,2' for unigrams+bigrams)")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--class-weight", choices=["balanced", None], default=None, help="Use balanced class weights to handle imbalanced data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def normalizeLabel(value, label2Id, lowerLabel2Id) -> int:
    if pd.isna(value):
        raise ValueError("Missing label (NaN)")

    if isinstance(value, str):
        key = value.strip()
        if not key:
            raise ValueError("Missing label (empty string)")
        if key in label2Id:
            return label2Id[key]
        low = key.lower()
        if low in lowerLabel2Id:
            return lowerLabel2Id[low]

        try:
            num = float(key)
            if num.is_integer():
                idx = int(num)
                if 0 <= idx < len(label2Id):
                    return idx
        except ValueError:
            pass

        raise ValueError(
            f"Unknown label '{value}'. Expected one of {list(label2Id.keys())} or numeric IDs."
        )

    idx = int(value)
    if idx < 0 or idx >= len(label2Id):
        raise ValueError(f"Label ID {idx} out of range [0, {len(label2Id)-1}]")
    return idx


def loadSplit(
    csvPath: str,
    sourceCol: str,
    replyCol: str,
    labelCol: str,
    label2Id: dict,
    mode: str,
) -> Tuple[List[str], List[int]]:
    """Load data from CSV and prepare texts and labels."""
    df = pd.read_csv(csvPath)
    for col in [sourceCol, replyCol, labelCol]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csvPath}. Found: {list(df.columns)}")

    # Drop rows with missing labels
    missingMask = df[labelCol].isna() | (df[labelCol].astype(str).str.strip() == "")
    missingCount = int(missingMask.sum())
    if missingCount > 0:
        missingRows = (df.index[missingMask] + 2).tolist()[:10]
        print(
            f"[WARN] {csvPath}: dropping {missingCount} rows with missing '{labelCol}'. "
            f"Example CSV line numbers: {missingRows}"
        )
        df = df.loc[~missingMask].copy()
        if df.empty:
            raise ValueError(f"All rows in {csvPath} have missing '{labelCol}'")

    sources = df[sourceCol].fillna("").astype(str).tolist()
    replies = df[replyCol].fillna("").astype(str).tolist()
    rawLabels = df[labelCol].tolist()
    rowIds = df.index.tolist()

    # Prepare texts based on mode
    if mode == "aware":
        # Concatenate source and reply
        texts = [f"{src} [SEP] {rep}" for src, rep in zip(sources, replies)]
    else:  # oblivious
        texts = replies

    # Normalize labels
    lowerLabel2Id = {k.lower(): v for k, v in label2Id.items()}
    labels = []
    for rowIdx, val in zip(rowIds, rawLabels):
        try:
            labels.append(normalizeLabel(val, label2Id, lowerLabel2Id))
        except ValueError as e:
            raise ValueError(f"Row {rowIdx + 2}: {e}")

    return texts, labels


def main():
    args = parseArgs()
    np.random.seed(args.seed)

    # Prepare labels
    labelsList = [l.strip() for l in args.labels.split(",")]
    label2Id = {label: i for i, label in enumerate(labelsList)}
    id2Label = {i: label for label, i in label2Id.items()}

    print(f"[INFO] Label mapping: {label2Id}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Max features: {args.max_features}")
    print(f"[INFO] N-gram range: {args.ngram_range}")
    print(f"[INFO] Regularization C: {args.C}")
    print(f"[INFO] Class weight: {args.class_weight or 'None'}")

    # Load training data
    print(f"\n[INFO] Loading training data from {args.train_csv}...")
    trainTexts, trainLabels = loadSplit(
        args.train_csv,
        args.source_col,
        args.reply_col,
        args.label_col,
        label2Id,
        args.mode,
    )
    print(f"[INFO] Loaded {len(trainTexts)} training samples")
    print(f"[INFO] Label distribution: {np.bincount(trainLabels)}")

    # Load validation data if provided
    valTexts, valLabels = None, None
    if args.val_csv:
        print(f"\n[INFO] Loading validation data from {args.val_csv}...")
        valTexts, valLabels = loadSplit(
            args.val_csv,
            args.source_col,
            args.reply_col,
            args.label_col,
            label2Id,
            args.mode,
        )
        print(f"[INFO] Loaded {len(valTexts)} validation samples")
        print(f"[INFO] Label distribution: {np.bincount(valLabels)}")

    # Load test data if provided
    testTexts, testLabels = None, None
    if args.test_csv:
        print(f"\n[INFO] Loading test data from {args.test_csv}...")
        testTexts, testLabels = loadSplit(
            args.test_csv,
            args.source_col,
            args.reply_col,
            args.label_col,
            label2Id,
            args.mode,
        )
        print(f"[INFO] Loaded {len(testTexts)} test samples")
        print(f"[INFO] Label distribution: {np.bincount(testLabels)}")

    # Parse n-gram range
    ngramRange = tuple(map(int, args.ngram_range.split(",")))

    # Train TF-IDF vectorizer
    print(f"\n[INFO] Training TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=ngramRange,
        min_df=2,
        max_df=0.95,
        lowercase=True,
        stop_words="english",
    )
    XTrain = tfidf.fit_transform(trainTexts)
    print(f"[INFO] TF-IDF vectorizer trained. Feature shape: {XTrain.shape}")

    # Train Logistic Regression
    print(f"\n[INFO] Training Logistic Regression model...")
    classWeight = "balanced" if args.class_weight == "balanced" else None
    lrModel = LogisticRegression(
        C=args.C,
        max_iter=1000,
        random_state=args.seed,
        n_jobs=-1,
        solver="lbfgs",
        class_weight=classWeight,
    )
    lrModel.fit(XTrain, trainLabels)
    print(f"[INFO] Model training complete")

    # Evaluate on training data
    print(f"\n[INFO] === TRAINING SET ===")
    trainPreds = lrModel.predict(XTrain)
    trainAcc = accuracy_score(trainLabels, trainPreds)
    trainF1 = f1_score(trainLabels, trainPreds, average="macro")
    print(f"Accuracy: {trainAcc:.4f}")
    print(f"Macro F1: {trainF1:.4f}")
    print("\nClassification Report:")
    print(classification_report(trainLabels, trainPreds, target_names=labelsList))

    # Evaluate on validation data if provided
    if valTexts:
        print(f"\n[INFO] === VALIDATION SET ===")
        XVal = tfidf.transform(valTexts)
        valPreds = lrModel.predict(XVal)
        valAcc = accuracy_score(valLabels, valPreds)
        valF1 = f1_score(valLabels, valPreds, average="macro")
        print(f"Accuracy: {valAcc:.4f}")
        print(f"Macro F1: {valF1:.4f}")
        print("\nClassification Report:")
        print(classification_report(valLabels, valPreds, target_names=labelsList))

    # Evaluate on test data if provided
    if testTexts:
        print(f"\n[INFO] === TEST SET ===")
        XTest = tfidf.transform(testTexts)
        testPreds = lrModel.predict(XTest)
        testAcc = accuracy_score(testLabels, testPreds)
        testF1 = f1_score(testLabels, testPreds, average="macro")
        print(f"Accuracy: {testAcc:.4f}")
        print(f"Macro F1: {testF1:.4f}")
        print("\nClassification Report:")
        print(classification_report(testLabels, testPreds, target_names=labelsList))

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save TF-IDF vectorizer
    tfidfPath = os.path.join(args.output_dir, "tfidf_vectorizer.pkl")
    with open(tfidfPath, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"\n[INFO] Saved TF-IDF vectorizer to {tfidfPath}")

    # Save Logistic Regression model
    modelPath = os.path.join(args.output_dir, "lr_model.pkl")
    with open(modelPath, "wb") as f:
        pickle.dump(lrModel, f)
    print(f"[INFO] Saved Logistic Regression model to {modelPath}")

    # Save metadata
    metadata = {
        "mode": args.mode,
        "labels": labelsList,
        "label2id": label2Id,
        "id2label": id2Label,
        "maxFeatures": args.max_features,
        "ngramRange": ngramRange,
        "trainSamples": len(trainTexts),
        "trainAccuracy": float(trainAcc),
        "trainF1": float(trainF1),
    }
    if valTexts:
        metadata["valAccuracy"] = float(valAcc)
        metadata["valF1"] = float(valF1)
    if testTexts:
        metadata["testAccuracy"] = float(testAcc)
        metadata["testF1"] = float(testF1)

    metadataPath = os.path.join(args.output_dir, "metadata.json")
    with open(metadataPath, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to {metadataPath}")

    print(f"\n[INFO] Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
