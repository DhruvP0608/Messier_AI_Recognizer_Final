from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter, ImageOps, ImageStat


@dataclass
class ImageFeatures:
    average_hash: str
    difference_hash: str
    edge_hash: str
    color_histogram: list[float]
    grayscale_histogram: list[float]
    aspect_ratio: float
    brightness: float
    contrast: float

    def to_dict(self) -> dict:
        return {
            "average_hash": self.average_hash,
            "difference_hash": self.difference_hash,
            "edge_hash": self.edge_hash,
            "color_histogram": self.color_histogram,
            "grayscale_histogram": self.grayscale_histogram,
            "aspect_ratio": self.aspect_ratio,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ImageFeatures":
        return cls(
            average_hash=payload["average_hash"],
            difference_hash=payload["difference_hash"],
            edge_hash=payload["edge_hash"],
            color_histogram=list(payload["color_histogram"]),
            grayscale_histogram=list(payload["grayscale_histogram"]),
            aspect_ratio=float(payload["aspect_ratio"]),
            brightness=float(payload["brightness"]),
            contrast=float(payload["contrast"]),
        )


def load_image(image_path: Path | str) -> Image.Image:
    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert("RGB")


def extract_features(image_path: Path | str) -> ImageFeatures:
    image = load_image(image_path)
    return extract_features_from_image(image)


def extract_features_from_image(image: Image.Image) -> ImageFeatures:
    grayscale = ImageOps.grayscale(image)
    average_hash = compute_average_hash(grayscale)
    difference_hash = compute_difference_hash(grayscale)
    edge_hash = compute_average_hash(grayscale.filter(ImageFilter.FIND_EDGES))
    color_histogram = normalized_color_histogram(image)
    grayscale_histogram = normalized_histogram(grayscale.histogram(), bins=16)
    aspect_ratio = image.width / image.height if image.height else 1.0

    stats = ImageStat.Stat(grayscale)
    brightness = stats.mean[0]
    contrast = stats.stddev[0]

    return ImageFeatures(
        average_hash=average_hash,
        difference_hash=difference_hash,
        edge_hash=edge_hash,
        color_histogram=color_histogram,
        grayscale_histogram=grayscale_histogram,
        aspect_ratio=aspect_ratio,
        brightness=brightness / 255.0,
        contrast=contrast / 255.0,
    )


def compute_average_hash(grayscale_image: Image.Image, size: int = 8) -> str:
    resized = grayscale_image.resize((size, size))
    pixels = list(resized.getdata())
    average = sum(pixels) / len(pixels)
    return "".join("1" if value >= average else "0" for value in pixels)


def compute_difference_hash(grayscale_image: Image.Image, size: int = 8) -> str:
    resized = grayscale_image.resize((size + 1, size))
    bits: list[str] = []
    for y in range(size):
        for x in range(size):
            left = resized.getpixel((x, y))
            right = resized.getpixel((x + 1, y))
            bits.append("1" if left > right else "0")
    return "".join(bits)


def normalized_color_histogram(image: Image.Image, bins_per_channel: int = 8) -> list[float]:
    channels = image.split()
    histogram: list[float] = []
    for channel in channels:
        histogram.extend(normalized_histogram(channel.histogram(), bins_per_channel))
    return histogram


def normalized_histogram(values: Iterable[int], bins: int) -> list[float]:
    values = list(values)
    bucket_size = len(values) // bins
    compressed = [
        sum(values[index : index + bucket_size])
        for index in range(0, len(values), bucket_size)
    ]
    compressed = compressed[:bins]
    total = sum(compressed) or 1
    return [value / total for value in compressed]


def hamming_distance(first: str, second: str) -> int:
    return sum(1 for a, b in zip(first, second) if a != b)


def l1_distance(first: Iterable[float], second: Iterable[float]) -> float:
    return sum(abs(a - b) for a, b in zip(first, second))
