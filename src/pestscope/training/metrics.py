from __future__ import annotations

import torch


def confusion_matrix(
    targets: list[int],
    predictions: list[int],
    *,
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for expected, predicted in zip(targets, predictions, strict=True):
        matrix[expected][predicted] += 1
    return matrix


def classification_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> dict:
    predictions = logits.argmax(dim=1)
    target_list = [int(item) for item in targets.cpu().tolist()]
    prediction_list = [int(item) for item in predictions.cpu().tolist()]
    matrix = confusion_matrix(target_list, prediction_list, num_classes=num_classes)

    per_class = []
    f1_values = []
    recall_values = []
    for class_index, row in enumerate(matrix):
        true_positive = row[class_index]
        false_negative = sum(row) - true_positive
        false_positive = sum(matrix[row_index][class_index] for row_index in range(num_classes))
        false_positive -= true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class.append(
            {
                "index": class_index,
                "support": recall_denominator,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        f1_values.append(f1)
        recall_values.append(recall)

    top_k = min(3, num_classes)
    topk_predictions = logits.topk(top_k, dim=1).indices
    top3_correct = topk_predictions.eq(targets.view(-1, 1)).any(dim=1).float().mean().item()
    top1_correct = predictions.eq(targets).float().mean().item()
    lossless_total = max(1, targets.numel())
    return {
        "samples": int(targets.numel()),
        "top1_accuracy": top1_correct,
        "top3_accuracy": top3_correct,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "balanced_accuracy": (sum(recall_values) / len(recall_values) if recall_values else 0.0),
        "confusion_matrix": matrix,
        "per_class": per_class,
        "predicted_distribution": [
            prediction_list.count(class_index) / lossless_total
            for class_index in range(num_classes)
        ],
    }
