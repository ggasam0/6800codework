# 6800codework

## Contents

### `Classifier` (classifier.py)
Implements three classification algorithms:
- **Naïve Bayes** with automatic discrete/continuous feature handling.
- **Logistic Regression** using softmax multi-class optimization.
- **Decision Tree** using a basic CART-style splitter with Gini impurity.

Each classifier supports training, prediction, and evaluation with accuracy,
precision, and recall metrics printed for both train and test splits.

### Dataset loaders (load_data.py)
Provides helpers to load:
- **Iris** (`load_iris`)
- **Congressional Voting Records** (`load_congress_data`)
- **MONK’s Problems** (`load_monks`)

Each loader returns `(training_data, test_data)` matrices where the first column
is the label and remaining columns are features.

### Training entry point (train_and_test.py)
Command-line entry point for running the classifiers against datasets. Use
`--dataset` and `--classifier` to configure the run.

## Usage

```bash
python train_and_test.py --dataset iris --classifier naive_bayes
python train_and_test.py --dataset congress --classifier logistic_regression --epochs 800
python train_and_test.py --dataset monks1 --classifier decision_tree --max_depth 5
```

## Report

Create a report (PDF preferred) with performance metrics (e.g., accuracy,
precision, recall, F1) per dataset and classifier. A basic template is provided
in `REPORT.md` for convenience.
