import argparse
from dataclasses import dataclass

import load_data
from classifier import Classifier


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


DATASET_LABELS = {
    "iris": "Iris",
    "congress": "Congressional Voting Records",
    "monks1": "MONK-1",
    "monks2": "MONK-2",
    "monks3": "MONK-3",
}

CLASSIFIER_LABELS = {
    "naive_bayes": "朴素贝叶斯",
    "logistic_regression": "逻辑回归",
    "decision_tree": "决策树",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report.md for classifier benchmarks.")
    parser.add_argument(
        "--output",
        default="report.md",
        help="Output markdown file path (default: report.md).",
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


def _evaluate(dataset: str, classifier_type: str, args: argparse.Namespace) -> Metrics:
    training_data, test_data = _load_dataset(dataset, args.training_ratio)
    classifier = Classifier(
        classifier_type,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    )
    classifier.train(training_data)
    metrics = classifier.test(test_data)
    return Metrics(
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
    )


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _render_table(results: dict[str, Metrics]) -> list[str]:
    lines = [
        "| 分类器 | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for classifier_key in ("naive_bayes", "logistic_regression", "decision_tree"):
        metrics = results[classifier_key]
        lines.append(
            "| {name} | {accuracy} | {precision} | {recall} | {f1} |".format(
                name=CLASSIFIER_LABELS[classifier_key],
                accuracy=_format_metric(metrics.accuracy),
                precision=_format_metric(metrics.precision),
                recall=_format_metric(metrics.recall),
                f1=_format_metric(metrics.f1),
            )
        )
    return lines


def _write_report(results: dict[str, dict[str, Metrics]], output_path: str) -> None:
    lines = ["# 分类报告", "", "以下结果为测试集指标（accuracy、precision、recall、F1）。", ""]

    lines.append("## 指标汇总")
    lines.append("")

    lines.append("### Iris")
    lines.extend(_render_table(results["iris"]))
    lines.append("")

    lines.append("### Congressional Voting Records")
    lines.extend(_render_table(results["congress"]))
    lines.append("")

    lines.append("### MONK’s Problems")
    lines.append("")

    for monks_key in ("monks1", "monks2", "monks3"):
        lines.append(f"#### {DATASET_LABELS[monks_key]}")
        lines.extend(_render_table(results[monks_key]))
        lines.append("")

    lines.append("## 备注")
    lines.append("")
    lines.append("- 本次仅更新测试集指标；训练集指标未写入报告。")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = _parse_args()
    datasets = ["iris", "congress", "monks1", "monks2", "monks3"]
    classifiers = ["naive_bayes", "logistic_regression", "decision_tree"]

    results: dict[str, dict[str, Metrics]] = {}
    for dataset in datasets:
        dataset_results: dict[str, Metrics] = {}
        for classifier_type in classifiers:
            dataset_results[classifier_type] = _evaluate(dataset, classifier_type, args)
        results[dataset] = dataset_results

    _write_report(results, args.output)


if __name__ == "__main__":
    main()
