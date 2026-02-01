import argparse

import load_data

from classifier import Classifier


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test classifiers on UCI datasets.")
    parser.add_argument(
        "--dataset",
        choices=["iris", "congress", "monks1", "monks2", "monks3", "all"],
        default="iris",
        help="Dataset to use.",
    )
    parser.add_argument(
        "--classifier",
        choices=["naive_bayes", "logistic_regression", "decision_tree"],
        default="naive_bayes",
        help="Classifier type.",
    )
    parser.add_argument("--training_ratio", type=float, default=0.7, help="Training split ratio.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for LR.")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for LR.")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for decision tree.")
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum samples to split for decision tree.",
    )
    return parser.parse_args()


def _load_dataset(dataset: str, training_ratio: float):
    if dataset == "iris":
        return load_data.load_iris(training_ratio)
    if dataset == "congress":
        return load_data.load_congress_data(training_ratio)
    if dataset == "monks1":
        return load_data.load_monks(1)
    if dataset == "monks2":
        return load_data.load_monks(2)
    if dataset == "monks3":
        return load_data.load_monks(3)
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    args = _parse_args()
    datasets = (
        ["iris", "congress", "monks1", "monks2", "monks3"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} | Classifier: {args.classifier} ===")
        training_data, test_data = _load_dataset(dataset, args.training_ratio)
        classifier = Classifier(
            args.classifier,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
        )
        classifier.train(training_data)
        classifier.test(test_data)


if __name__ == "__main__":
    main()
