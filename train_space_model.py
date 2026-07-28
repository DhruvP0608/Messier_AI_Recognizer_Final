from __future__ import annotations

import argparse
import json
from pathlib import Path

from astronomy_recognizer.ml_classifier import train_and_evaluate


DEFAULT_DATASET = Path("SpaceDataset/images")
DEFAULT_MODEL = Path("artifacts/space_category_model.pkl")
DEFAULT_METRICS = Path("artifacts/space_ml_metrics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a celestial-category model using SpaceDataset."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to SpaceDataset/images")
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL), help="Path to save trained model artifact")
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS), help="Path to save evaluation metrics JSON")
    return parser


def run() -> None:
    args = build_parser().parse_args()
    metrics = train_and_evaluate(args.dataset, args.model_output)

    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2))

    model_metrics = metrics["model"]
    print(f"Model written to: {args.model_output}")
    print(f"Metrics written to: {args.metrics_output}")
    print(f"Accuracy: {model_metrics['accuracy']}%")
    print(f"Macro F1: {model_metrics['macro_f1']}%")
    print(f"Top-3 Accuracy: {model_metrics['top3_accuracy']}%")


if __name__ == "__main__":
    run()
