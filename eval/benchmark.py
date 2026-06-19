from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from models.adaptive_service import AdaptiveVisionService
from models.drift import DriftMonitor
from models.embeddings import CLIPImageEmbedder
from models.prototype_memory import PrototypeMemory

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_dataset(root: Path) -> dict[str, list[Path]]:
    dataset = {
        directory.name: sorted(
            path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for directory in sorted(root.iterdir())
        if directory.is_dir()
    }
    return {label: paths for label, paths in dataset.items() if paths}


def evaluate(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    dataset = load_dataset(args.dataset)
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two class directories")

    service = AdaptiveVisionService(
        embedder=CLIPImageEmbedder(args.model, args.device),
        memory=PrototypeMemory(),
        monitor=DriftMonitor(window_size=sum(map(len, dataset.values()))),
        confidence_threshold=args.threshold,
    )
    query_set: list[tuple[str, Path]] = []
    support_counts: dict[str, int] = {}

    for label, paths in dataset.items():
        shuffled = paths.copy()
        random.shuffle(shuffled)
        if len(shuffled) <= args.support_per_class:
            raise ValueError(
                f"Class {label!r} needs more than {args.support_per_class} images for evaluation"
            )
        support = shuffled[: args.support_per_class]
        query = shuffled[args.support_per_class :]
        support_images = []
        for path in support:
            with Image.open(path) as image:
                support_images.append(image.convert("RGB"))
        service.teach(label, support_images)
        query_set.extend((label, path) for path in query)
        support_counts[label] = len(support)

    correct = 0
    unknown = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for expected, path in query_set:
        with Image.open(path) as image:
            prediction = service.predict(image.convert("RGB"), top_k=1)
        predicted = prediction.label
        correct += int(predicted == expected)
        unknown += int(prediction.is_unknown)
        per_class[expected]["correct"] += int(predicted == expected)
        per_class[expected]["total"] += 1

    total = len(query_set)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "support_per_class": args.support_per_class,
        "support_counts": support_counts,
        "query_images": total,
        "top1_accuracy": correct / total,
        "unknown_rate": unknown / total,
        "confidence_threshold": args.threshold,
        "per_class_accuracy": {
            label: values["correct"] / values["total"] for label, values in per_class.items()
        },
        "drift": service.metrics(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate few-shot prototype classification")
    parser.add_argument("dataset", type=Path, help="Folder containing one subfolder per class")
    parser.add_argument("--support-per-class", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = evaluate(arguments)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
