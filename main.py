from __future__ import annotations

import argparse
from pathlib import Path

from astronomy_recognizer.recognizer import MessierRecognizer, build_reference_index


DEFAULT_DATASET = Path("Dataset")
DEFAULT_INDEX = Path("artifacts/reference_index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Astronomical object recognition system for Messier images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("build-index", help="Extract features for the dataset.")
    index_parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    index_parser.add_argument("--output", default=str(DEFAULT_INDEX))

    predict_parser = subparsers.add_parser("predict", help="Predict the Messier object in an image.")
    predict_parser.add_argument("image", help="Path to the query image.")
    predict_parser.add_argument("--index", default=str(DEFAULT_INDEX))
    predict_parser.add_argument("--top-k", type=int, default=3)

    return parser


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-index":
        build_reference_index(args.dataset, args.output)
        print(f"Reference index written to {args.output}")
        return

    recognizer = MessierRecognizer(args.index)
    matches = recognizer.predict(args.image, top_k=args.top_k)

    print(f"Input image: {args.image}")
    print()
    for rank, match in enumerate(matches, start=1):
        print(f"Match {rank}")
        print(f"  Object: {match.title}")
        print(f"  Messier ID: {match.messier_id}")
        print(f"  Category: {match.category}")
        print(f"  Confidence: {match.confidence_label} ({match.similarity_score}%)")
        print(f"  Details: {match.description}")
        print(f"  Reference image: {match.reference_image}")
        print(f"  Star map: {match.star_map_image or 'Not available'}")
        print()


if __name__ == "__main__":
    run()
