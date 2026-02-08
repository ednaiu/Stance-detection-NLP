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
from scipy.sparse import hstack

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("[WARN] TextBlob not available. Install with: pip install textblob")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


DEFAULT_LABELS = ["support", "deny", "query", "comment"]

# Stance-specific keywords for enhanced features
SUPPORT_KEYWORDS = [
    'agree', 'support', 'correct', 'right', 'true', 'yes', 'absolutely', 
    'exactly', 'definitely', 'indeed', 'confirm', 'verify', 'valid', 'accurate',
    'approve', 'endorse', 'backing', 'concur', 'second', 'affirm', 'totally',
    'completely', 'precisely', 'spot on', 'well said', 'truth', 'fact', 'real',
    # Extended support keywords
    'agreed', 'supporting', 'confirmed', 'exactly right', 'so true', 'amen',
    'preach', '+1', 'this', 'same', 'ditto', 'likewise', 'same here', 'me too',
    'finally', 'glad', 'thank you', 'appreciate', 'kudos', 'bravo', 'well done',
    'perfect', 'brilliant', 'excellent', 'outstanding', 'accurate', 'valid point',
    'good point', 'fair point', 'makes sense', 'convinced', 'i believe'
]

DENY_KEYWORDS = [
    'disagree', 'wrong', 'false', 'no', 'not', 'never', 'against', 'oppose',
    'incorrect', 'untrue', 'fake', 'lie', 'misinformation', 'doubt', 'reject',
    'deny', 'refute', 'contradict', 'dispute', 'challenge', 'nonsense', 'nope',
    # Extended deny keywords
    'bullshit', 'bs', 'rubbish', 'ridiculous', 'absurd', 'nonsensical',
    'debunk', 'debunked', 'myth', 'propaganda', 'hoax', 'scam', 'fraud',
    'misleading', 'distortion', 'fabrication', 'false claim', 'not true',
    'thats wrong', "that's wrong", 'you are wrong', 'youre wrong', "you're wrong",
    'no way', 'cant believe', "can't believe", 'seriously?', 'really?'
]

QUERY_KEYWORDS = [
    'what', 'why', 'how', 'when', 'where', 'who', 'which', 'really',
    'source', 'proof', 'evidence', 'citation', 'link', 'explain',
    'elaborate', 'clarify', 'context', 'wonder', 'curious', 'question',
    # Extended query keywords
    '?', 'asking', 'wonder', 'wondering', 'anyone know', 'does anyone',
    'can someone', 'please explain', 'i dont understand', "i don't understand",
    'confused', 'unsure', 'uncertain', 'verify this', 'is this true',
    'confirm this', 'show me', 'need proof', 'got source', 'link please'
]


def extract_stance_keywords(text: str) -> Tuple[int, int, int]:
    """Extract stance keyword counts from text."""
    text_lower = text.lower()
    support_count = sum(1 for word in SUPPORT_KEYWORDS if word in text_lower)
    deny_count = sum(1 for word in DENY_KEYWORDS if word in text_lower)
    query_count = sum(1 for word in QUERY_KEYWORDS if word in text_lower)
    return support_count, deny_count, query_count


def extract_sentiment(text: str) -> Tuple[float, float]:
    """Extract sentiment polarity and subjectivity."""
    if not TEXTBLOB_AVAILABLE:
        return 0.0, 0.0
    try:
        sentiment = TextBlob(text).sentiment
        return sentiment.polarity, sentiment.subjectivity
    except:
        return 0.0, 0.0


def create_enhanced_features(texts: List[str], tfidf_features):
    """Combine TF-IDF with keyword and sentiment features."""
    keyword_features = []
    sentiment_features = []
    
    for text in texts:
        # Extract keyword features
        support_kw, deny_kw, query_kw = extract_stance_keywords(text)
        keyword_features.append([support_kw, deny_kw, query_kw])
        
        # Extract sentiment features
        polarity, subjectivity = extract_sentiment(text)
        sentiment_features.append([polarity, subjectivity])
    
    # Convert to arrays
    keyword_features = np.array(keyword_features, dtype=np.float32)
    sentiment_features = np.array(sentiment_features, dtype=np.float32)
    
    # Combine all features
    enhanced_features = hstack([tfidf_features, keyword_features, sentiment_features])
    return enhanced_features


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
    parser.add_argument("--class-weight", choices=["balanced", None], default="balanced", help="Use balanced class weights to handle imbalanced data")
    parser.add_argument("--use-enhanced-features", action="store_true", default=True, help="Add stance keywords and sentiment features")
    parser.add_argument("--use-smote", action="store_true", default=False, help="Apply SMOTE oversampling for minority classes")
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
    XTrain_tfidf = tfidf.fit_transform(trainTexts)
    print(f"[INFO] TF-IDF vectorizer trained. Feature shape: {XTrain_tfidf.shape}")
    
    # Create enhanced features if enabled
    if args.use_enhanced_features:
        print(f"[INFO] Adding stance keywords and sentiment features...")
        XTrain = create_enhanced_features(trainTexts, XTrain_tfidf)
        print(f"[INFO] Enhanced feature shape: {XTrain.shape}")
        print(f"[INFO] Added {XTrain.shape[1] - XTrain_tfidf.shape[1]} new features (keywords + sentiment)")
    else:
        XTrain = XTrain_tfidf
    
    # Apply SMOTE if enabled
    if args.use_smote:
        if not SMOTE_AVAILABLE:
            print("[WARN] SMOTE not available. Install with: pip install imbalanced-learn")
            print("[WARN] Skipping SMOTE oversampling")
        else:
            print(f"\n[INFO] Applying SMOTE oversampling...")
            print(f"[INFO] Before SMOTE: {np.bincount(trainLabels)}")
            
            # Target: make minority classes at least 30% of majority class
            majority_count = max(np.bincount(trainLabels))
            target_count = int(majority_count * 0.3)
            
            sampling_strategy = {
                0: max(target_count, np.bincount(trainLabels)[0]),  # support
                1: max(target_count, np.bincount(trainLabels)[1]),  # deny
                2: max(target_count, np.bincount(trainLabels)[2]),  # query
            }
            
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=args.seed)
            XTrain, trainLabels = smote.fit_resample(XTrain, trainLabels)
            
            print(f"[INFO] After SMOTE: {np.bincount(trainLabels)}")
            print(f"[INFO] New training set size: {len(trainLabels)}")

    # Train Logistic Regression
    print(f"\n[INFO] Training Logistic Regression model...")
    # Custom class weights - heavily boost minority classes
    if args.class_weight == "balanced":
        # Calculate balanced weights but multiply support/deny/query by additional factor
        from sklearn.utils.class_weight import compute_class_weight
        base_weights = compute_class_weight('balanced', classes=np.unique(trainLabels), y=trainLabels)
        classWeight = {
            0: base_weights[0] * 3.0,  # support - 3x boost
            1: base_weights[1] * 2.0,  # deny - 2x boost
            2: base_weights[2] * 2.0,  # query - 2x boost
            3: base_weights[3]         # comment - normal
        }
        print(f"[INFO] Custom class weights: {classWeight}")
    else:
        classWeight = None
    
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
        XVal_tfidf = tfidf.transform(valTexts)
        if args.use_enhanced_features:
            XVal = create_enhanced_features(valTexts, XVal_tfidf)
        else:
            XVal = XVal_tfidf
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
        XTest_tfidf = tfidf.transform(testTexts)
        if args.use_enhanced_features:
            XTest = create_enhanced_features(testTexts, XTest_tfidf)
        else:
            XTest = XTest_tfidf
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
        "useEnhancedFeatures": args.use_enhanced_features,
        "classWeight": args.class_weight,
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
