# Stance Classifier

Stance Classifier is a research-focused project on stance detection in social media conversations. The task is modeled as a four-class classification problem with labels support, deny, query, and comment. The central objective is to infer the stance expressed in a reply, either independently or relative to a target post.

This repository was built as a complete mini research pipeline and includes:

- an inference-ready Python package for transformer-based stance prediction
- multiple baseline training scripts for controlled comparison
- prediction scripts for reproducible evaluation on held-out data
- notebooks and a project report that document methodology, data handling, and model trade-offs

## Research Context and Data

The experimental setup follows the RumourEval stance detection framing used in the project notebooks and report. Data is organized as conversational pairs with source text and reply text, plus a stance label. The implementation supports flexible column names and label normalization so the same pipeline can be reused across datasets with similar annotation schemes.

The repository includes a six-page project report and supporting notebooks that describe:

- the task definition and motivation for misinformation and discourse analysis
- data exploration and quality checks
- baseline design choices from lexical to transformer-based models
- evaluation with accuracy, macro F1, and class-level diagnostics

## Implemented Modeling Approaches

The project compares complementary modeling families:

- TF-IDF plus Logistic Regression as a fast and interpretable lexical baseline
- Sentence Embeddings plus Logistic Regression as a semantic baseline
- Cross-Encoder based scoring with a downstream classifier
- Transformer inference wrappers for target-oblivious and target-aware prediction
- an ensemble strategy that selects the more confident prediction across target-aware and target-oblivious models

This design emphasizes methodological clarity. The same label space is preserved across models, which allows direct and fair baseline comparison.

## Reproducibility and Engineering Quality

The codebase is structured for repeatable experimentation:

- explicit command-line interfaces for train, validation, and test splits
- configurable label mappings and text columns
- handling for missing labels and class imbalance options
- saved model artifacts and metadata for consistent inference
- isolated scripts for training and prediction across each baseline family
