"""Class for a classification algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _split_features_labels(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    accuracy = float(np.mean(y_true == y_pred))
    precision_scores = []
    recall_scores = []
    for cls in classes:
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        false_negative = np.sum((y_true == cls) & (y_pred != cls))
        precision = true_positive / (true_positive + false_positive + 1e-12)
        recall = true_positive / (true_positive + false_negative + 1e-12)
        precision_scores.append(precision)
        recall_scores.append(recall)
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
    }


def _is_discrete_feature(values: np.ndarray, max_unique: int = 12) -> bool:
    if np.any(values != np.floor(values)):
        return False
    unique_values = np.unique(values)
    return len(unique_values) <= max_unique


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )


@dataclass
class DecisionTreeNode:
    is_leaf: bool
    prediction: int | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: "DecisionTreeNode | None" = None
    right: "DecisionTreeNode | None" = None


class Classifier:
    def __init__(self, classifier_type: str, **kwargs):
        """Initialize classifier with parameters."""
        self.classifier_type = classifier_type
        self.params = kwargs

    def train(self, training_data: np.ndarray) -> dict[str, float]:
        features, labels = _split_features_labels(training_data)
        if self.classifier_type == "naive_bayes":
            self._train_naive_bayes(features, labels)
        elif self.classifier_type == "logistic_regression":
            self._train_logistic_regression(features, labels)
        elif self.classifier_type == "decision_tree":
            self._train_decision_tree(features, labels)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

        train_predictions = np.array([self.predict(row) for row in training_data])
        metrics = _compute_metrics(labels, train_predictions)
        _print_metrics("Training", metrics)
        return metrics

    def predict(self, data: np.ndarray) -> int:
        if self.classifier_type == "naive_bayes":
            return self._predict_naive_bayes(data[1:])
        if self.classifier_type == "logistic_regression":
            return self._predict_logistic_regression(data[1:])
        if self.classifier_type == "decision_tree":
            return self._predict_decision_tree(data[1:])
        raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

    def test(self, test_data: np.ndarray) -> dict[str, float]:
        features, labels = _split_features_labels(test_data)
        predictions = np.array([self.predict(row) for row in test_data])
        metrics = _compute_metrics(labels, predictions)
        _print_metrics("Test", metrics)
        return metrics

    def _train_naive_bayes(self, features: np.ndarray, labels: np.ndarray) -> None:
        classes = np.unique(labels)
        num_features = features.shape[1]
        feature_types = []
        for i in range(num_features):
            feature_types.append("discrete" if _is_discrete_feature(features[:, i]) else "continuous")

        model = {
            "classes": classes,
            "class_priors": {},
            "feature_types": feature_types,
            "discrete": [],
            "continuous": [],
        }
        for cls in classes:
            cls_mask = labels == cls
            model["class_priors"][cls] = float(np.mean(cls_mask))
            cls_features = features[cls_mask]
            discrete_stats = []
            continuous_stats = []
            for idx, feature_type in enumerate(feature_types):
                column = cls_features[:, idx]
                if feature_type == "discrete":
                    values, counts = np.unique(column, return_counts=True)
                    probs = (counts + 1) / (counts.sum() + len(values))
                    discrete_stats.append({"values": values, "probs": probs})
                    continuous_stats.append(None)
                else:
                    mean = float(np.mean(column))
                    var = float(np.var(column) + 1e-6)
                    continuous_stats.append({"mean": mean, "var": var})
                    discrete_stats.append(None)
            model["discrete"].append(discrete_stats)
            model["continuous"].append(continuous_stats)
        self.params["naive_bayes_model"] = model

    def _predict_naive_bayes(self, features: np.ndarray) -> int:
        model = self.params["naive_bayes_model"]
        classes = model["classes"]
        log_probs = []
        for class_index, cls in enumerate(classes):
            log_prob = math.log(model["class_priors"][cls] + 1e-12)
            for idx, feature_type in enumerate(model["feature_types"]):
                value = features[idx]
                if feature_type == "discrete":
                    stats = model["discrete"][class_index][idx]
                    if stats is None:
                        continue
                    if value in stats["values"]:
                        value_index = int(np.where(stats["values"] == value)[0][0])
                        prob = stats["probs"][value_index]
                    else:
                        prob = 1.0 / (np.sum(stats["probs"]) + len(stats["values"]))
                    log_prob += math.log(prob + 1e-12)
                else:
                    stats = model["continuous"][class_index][idx]
                    if stats is None:
                        continue
                    mean = stats["mean"]
                    var = stats["var"]
                    exponent = -((value - mean) ** 2) / (2 * var)
                    log_prob += -0.5 * math.log(2 * math.pi * var) + exponent
            log_probs.append(log_prob)
        return int(classes[int(np.argmax(log_probs))])

    def _train_logistic_regression(self, features: np.ndarray, labels: np.ndarray) -> None:
        learning_rate = float(self.params.get("learning_rate", 0.1))
        epochs = int(self.params.get("epochs", 500))
        reg_strength = float(self.params.get("reg_strength", 0.0))

        num_samples, num_features = features.shape
        classes = np.unique(labels)
        num_classes = len(classes)
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        y = np.zeros((num_samples, num_classes))
        for i, label in enumerate(labels):
            y[i, class_to_index[label]] = 1.0

        X = np.hstack([np.ones((num_samples, 1)), features])
        weights = np.zeros((num_classes, num_features + 1))

        for _ in range(epochs):
            scores = X @ weights.T
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            gradient = ((probs - y).T @ X) / num_samples
            gradient += reg_strength * weights
            weights -= learning_rate * gradient

        self.params["log_reg_model"] = {
            "classes": classes,
            "weights": weights,
        }

    def _predict_logistic_regression(self, features: np.ndarray) -> int:
        model = self.params["log_reg_model"]
        weights = model["weights"]
        X = np.hstack([[1.0], features])
        scores = weights @ X
        class_index = int(np.argmax(scores))
        return int(model["classes"][class_index])

    def _train_decision_tree(self, features: np.ndarray, labels: np.ndarray) -> None:
        max_depth = int(self.params.get("max_depth", 6))
        min_samples_split = int(self.params.get("min_samples_split", 2))
        self.params["decision_tree_model"] = self._build_tree(
            features, labels, depth=0, max_depth=max_depth, min_samples_split=min_samples_split
        )

    def _predict_decision_tree(self, features: np.ndarray) -> int:
        node = self.params["decision_tree_model"]
        while not node.is_leaf:
            if features[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return int(node.prediction)

    def _build_tree(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        depth: int,
        max_depth: int,
        min_samples_split: int,
    ) -> DecisionTreeNode:
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            return DecisionTreeNode(is_leaf=True, prediction=int(unique_labels[0]))
        if depth >= max_depth or len(labels) < min_samples_split:
            prediction = int(self._majority_class(labels))
            return DecisionTreeNode(is_leaf=True, prediction=prediction)

        best_feature, best_threshold, best_gini = None, None, float("inf")
        num_features = features.shape[1]
        for feature_index in range(num_features):
            thresholds = np.unique(features[:, feature_index])
            if len(thresholds) > 12:
                thresholds = np.percentile(thresholds, [10, 25, 50, 75, 90])
            for threshold in thresholds:
                gini = self._gini_split(features, labels, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            prediction = int(self._majority_class(labels))
            return DecisionTreeNode(is_leaf=True, prediction=prediction)

        left_mask = features[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_node = self._build_tree(
            features[left_mask],
            labels[left_mask],
            depth + 1,
            max_depth,
            min_samples_split,
        )
        right_node = self._build_tree(
            features[right_mask],
            labels[right_mask],
            depth + 1,
            max_depth,
            min_samples_split,
        )
        return DecisionTreeNode(
            is_leaf=False,
            feature_index=best_feature,
            threshold=float(best_threshold),
            left=left_node,
            right=right_node,
        )

    @staticmethod
    def _majority_class(labels: np.ndarray) -> int:
        values, counts = np.unique(labels, return_counts=True)
        return int(values[int(np.argmax(counts))])

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        proportions = counts / counts.sum()
        return 1.0 - np.sum(proportions**2)

    def _gini_split(
        self, features: np.ndarray, labels: np.ndarray, feature_index: int, threshold: float
    ) -> float:
        left_mask = features[:, feature_index] <= threshold
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return float("inf")
        left_gini = self._gini(labels[left_mask])
        right_gini = self._gini(labels[right_mask])
        left_weight = np.sum(left_mask) / len(labels)
        right_weight = np.sum(right_mask) / len(labels)
        return left_weight * left_gini + right_weight * right_gini
