from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageEnhance, ImageFilter, ImageOps

from .features import (
    ImageFeatures,
    extract_features,
    extract_features_from_image,
    hamming_distance,
    l1_distance,
    load_image,
)
from .metadata import build_catalog_entry


@dataclass
class MatchResult:
    messier_id: str
    title: str
    common_name: str | None
    category: str
    description: str
    reference_image: str
    star_map_image: str | None
    similarity_score: float
    confidence_label: str


class MessierRecognizer:
    def __init__(self, index_path: Path | str):
        index_data = json.loads(Path(index_path).read_text())
        self.entries = []
        for item in index_data["entries"]:
            self.entries.append(
                {
                    "metadata": item["metadata"],
                    "features": ImageFeatures.from_dict(item["features"]),
                    "feature_sets": [
                        ImageFeatures.from_dict(feature_set["features"])
                        for feature_set in item.get("feature_sets", [{"features": item["features"]}])
                    ],
                }
            )

    def predict(
        self,
        image_path: Path | str,
        top_k: int = 3,
        allowed_categories: set[str] | None = None,
    ) -> list[MatchResult]:
        query_feature_sets = extract_query_feature_sets(image_path)
        normalized_allowed = None
        if allowed_categories:
            normalized_allowed = {item.strip().lower() for item in allowed_categories}
        scored = []
        for item in self.entries:
            metadata = item["metadata"]
            if normalized_allowed and metadata["category"].strip().lower() not in normalized_allowed:
                continue
            similarity_score = score_object(query_feature_sets, item["feature_sets"])
            scored.append(
                MatchResult(
                    messier_id=metadata["messier_id"],
                    title=metadata["title"],
                    common_name=metadata["common_name"],
                    category=metadata["category"],
                    description=metadata["description"],
                    reference_image=metadata["reference_image"],
                    star_map_image=metadata.get("star_map_image"),
                    similarity_score=round(similarity_score, 2),
                    confidence_label="Pending",
                )
            )
        scored.sort(key=lambda item: item.similarity_score, reverse=True)
        if scored:
            runner_up = scored[1].similarity_score if len(scored) > 1 else 0.0
            for item in scored[:top_k]:
                gap = item.similarity_score - runner_up if item is scored[0] else 0.0
                item.confidence_label = confidence_from_score(item.similarity_score, gap)
        return scored[:top_k]


def build_reference_index(dataset_root: Path | str, output_path: Path | str) -> None:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)

    entries = []
    for image_path in sorted(dataset_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        entries.append(
            {
                "metadata": build_catalog_entry(image_path),
                "features": extract_features(image_path).to_dict(),
                "feature_sets": build_reference_feature_sets(image_path),
            }
        )

    payload = {
        "dataset_root": str(dataset_root),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def combined_distance(query: ImageFeatures, reference: ImageFeatures) -> float:
    hash_distance = (
        hamming_distance(query.average_hash, reference.average_hash)
        + hamming_distance(query.difference_hash, reference.difference_hash)
        + hamming_distance(query.edge_hash, reference.edge_hash)
    ) / 192.0

    color_distance = l1_distance(query.color_histogram, reference.color_histogram) / 6.0
    grayscale_distance = l1_distance(query.grayscale_histogram, reference.grayscale_histogram) / 2.0
    aspect_distance = min(1.0, abs(query.aspect_ratio - reference.aspect_ratio))
    brightness_distance = abs(query.brightness - reference.brightness)
    contrast_distance = abs(query.contrast - reference.contrast)

    return (
        0.42 * hash_distance
        + 0.26 * color_distance
        + 0.16 * grayscale_distance
        + 0.06 * aspect_distance
        + 0.05 * brightness_distance
        + 0.05 * contrast_distance
    )


def build_reference_feature_sets(image_path: Path | str) -> list[dict]:
    image = load_image(image_path)
    return serialize_feature_sets(generate_image_variants(image, include_flip=False))


def extract_query_feature_sets(image_path: Path | str) -> list[ImageFeatures]:
    image = load_image(image_path)
    return generate_image_variants(image, include_flip=True)


def generate_image_variants(image, include_flip: bool) -> list[ImageFeatures]:
    variants = [
        image,
        ImageOps.autocontrast(image),
        ImageEnhance.Contrast(image).enhance(1.18),
        ImageEnhance.Contrast(image).enhance(0.88),
        ImageEnhance.Brightness(image).enhance(1.08),
        ImageEnhance.Brightness(image).enhance(0.92),
        image.filter(ImageFilter.SHARPEN),
        image.filter(ImageFilter.SMOOTH),
    ]
    if include_flip:
        variants.append(ImageOps.mirror(image))
    return [extract_features_from_image(variant.convert("RGB")) for variant in variants]


def serialize_feature_sets(feature_sets: list[ImageFeatures]) -> list[dict]:
    return [
        {
            "features": feature_set.to_dict(),
        }
        for feature_set in feature_sets
    ]


def score_object(query_feature_sets: list[ImageFeatures], reference_feature_sets: list[ImageFeatures]) -> float:
    best_score = 0.0
    for query_features in query_feature_sets:
        for reference_features in reference_feature_sets:
            distance = combined_distance(query_features, reference_features)
            score = max(0.0, 100.0 - distance * 100.0)
            if score > best_score:
                best_score = score
    return best_score


def confidence_from_score(score: float, top_gap: float = 0.0) -> str:
    if score >= 90 and top_gap >= 6:
        return "Very high"
    if score >= 82 and top_gap >= 3:
        return "High"
    if score >= 70:
        return "Medium"
    return "Low"
