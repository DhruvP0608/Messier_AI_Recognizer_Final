from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import tempfile
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from astronomy_recognizer.features import ImageFeatures
from astronomy_recognizer.ml_classifier import SpaceCategoryModel
from astronomy_recognizer.recognizer import MessierRecognizer, combined_distance, confidence_from_score


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
INDEX_PATH = PROJECT_ROOT / "artifacts" / "reference_index.json"
ML_METRICS_PATH = PROJECT_ROOT / "artifacts" / "ml_metrics.json"
SPACE_ML_METRICS_PATH = PROJECT_ROOT / "artifacts" / "space_ml_metrics.json"
TRAINING_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "training_checkpoint.json"
SPACE_MODEL_PATH = PROJECT_ROOT / "artifacts" / "space_category_model.pkl"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = 8000


class MessierRequestHandler(SimpleHTTPRequestHandler):
    recognizer = MessierRecognizer(INDEX_PATH)
    ml_model = SpaceCategoryModel(SPACE_MODEL_PATH) if SPACE_MODEL_PATH.exists() else None
    catalog_entries = json.loads(INDEX_PATH.read_text())["entries"]
    dashboard_payload = None

    def translate_path(self, path: str) -> str:
        parsed_path = urlparse(path).path
        if parsed_path in {"/", "/index.html"}:
            return str(WEB_ROOT / "index.html")
        if parsed_path == "/catalog":
            return str(WEB_ROOT / "catalog.html")
        if parsed_path == "/dashboard":
            return str(WEB_ROOT / "dashboard.html")
        if parsed_path.startswith("/web/"):
            return str(PROJECT_ROOT / parsed_path.lstrip("/"))
        if parsed_path.startswith("/Dataset/") or parsed_path.startswith("/Map_Dataset/"):
            return str(PROJECT_ROOT / parsed_path.lstrip("/"))
        return str(WEB_ROOT / parsed_path.lstrip("/"))

    def do_GET(self) -> None:
        if self.path == "/api/catalog":
            payload = {
                "entries": [
                    serialize_catalog_entry(item["metadata"]) for item in sorted(
                        self.catalog_entries,
                        key=lambda entry: int(entry["metadata"]["messier_id"][1:]),
                    )
                ]
            }
            self._send_json(payload)
            return
        if self.path == "/api/dashboard":
            if self.dashboard_payload is None:
                type(self).dashboard_payload = build_dashboard_payload(self.catalog_entries)
            self._send_json(self.dashboard_payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(content_length)
            file_name, file_bytes = parse_uploaded_file(content_type, body)

            suffix = Path(file_name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(file_bytes)
                temp_path = Path(handle.name)

            try:
                ml_context = None
                routed_matches = []
                total_catalog = len(self.recognizer.entries)
                global_matches = self.recognizer.predict(temp_path, top_k=total_catalog)
                if self.ml_model is not None:
                    ml_context = self.ml_model.predict(temp_path)
                    allowed_categories = set(ml_context.get("messier_categories", []))
                    if allowed_categories:
                        routed_matches = self.recognizer.predict(
                            temp_path,
                            top_k=total_catalog,
                            allowed_categories=allowed_categories,
                        )
                matches = global_matches[:]
                if routed_matches:
                    global_top = global_matches[0].similarity_score if global_matches else 0.0
                    routed_top = routed_matches[0].similarity_score
                    # Use routed results only when they are effectively as strong as global top match.
                    if global_top - routed_top <= 1.0:
                        matches = routed_matches
                if ml_context:
                    matches = apply_ml_category_prior(matches, ml_context, temp_path)
                matches = refresh_confidence_labels(matches)[:3]
            finally:
                temp_path.unlink(missing_ok=True)

            payload = {
                "top_match": serialize_match(matches[0]),
                "matches": [serialize_match(match) for match in matches],
            }
            if ml_context:
                payload["ml_context"] = ml_context

            stellar_info = fetch_stellar_info(matches[0].messier_id)
            payload["stellar_info"] = stellar_info

            self._send_json(payload)
        except Exception as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def guess_type(self, path: str) -> str:
        guessed = super().guess_type(path)
        return guessed or mimetypes.guess_type(path)[0] or "application/octet-stream"


def parse_uploaded_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart form data")

    header_blob = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header_blob + body)

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        if part.get_param("name", header="Content-Disposition") != "image":
            continue
        file_name = part.get_filename() or "upload.jpg"
        payload = part.get_payload(decode=True)
        if not payload:
            raise ValueError("Uploaded file is empty")
        return file_name, payload

    raise ValueError("No image file was uploaded")


def serialize_match(match) -> dict:
    return {
        "messier_id": match.messier_id,
        "title": match.title,
        "common_name": match.common_name,
        "category": match.category,
        "description": match.description,
        "reference_image": normalize_public_path(match.reference_image),
        "star_map_image": normalize_public_path(match.star_map_image),
        "similarity_score": match.similarity_score,
        "confidence_label": match.confidence_label,
    }


def normalize_public_path(path: str | None) -> str | None:
    if path is None:
        return None
    normalized = Path(path).as_posix().lstrip("./")
    return f"/{normalized}"


def serialize_catalog_entry(metadata: dict) -> dict:
    return {
        "messier_id": metadata["messier_id"],
        "title": metadata["title"],
        "common_name": metadata["common_name"],
        "category": metadata["category"],
        "description": metadata["description"],
        "reference_image": normalize_public_path(metadata["reference_image"]),
        "star_map_image": normalize_public_path(metadata["star_map_image"]),
    }


def _extract_json_object(text: str) -> str | None:
    # Attempt to recover the first JSON object from Gemini response text.
    braces = 0
    start = None
    for index, char in enumerate(text):
        if char == "{" and start is None:
            start = index
            braces = 1
            continue
        if start is not None:
            if char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
                if braces == 0:
                    return text[start : index + 1]
    return None


def fetch_stellar_info(messier_id: str) -> dict:
    if not HAS_GEMINI:
        return {"error": "Gemini SDK unavailable"}

    prompt = (
        f"Provide a JSON object with the following keys for the Messier object {messier_id}:"
        " name, constellation, distance_ly, diameter_ly, year_of_discovery."
        " Use plain text values and return only valid JSON."
    )
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        text = response.text.strip()
        json_text = _extract_json_object(text) or text
        stellar_info = json.loads(json_text)
        return {
            "name": stellar_info.get("name", "Unknown"),
            "constellation": stellar_info.get("constellation", "Unknown"),
            "distance_ly": stellar_info.get("distance_ly", "Unknown"),
            "diameter_ly": stellar_info.get("diameter_ly", "Unknown"),
            "year_of_discovery": stellar_info.get("year_of_discovery", "Unknown"),
        }
    except Exception as error:
        return {"error": f"Could not fetch Stellar Info: {error}"}


def build_dashboard_payload(entries: list[dict]) -> dict:
    metadata_entries = [entry["metadata"] for entry in entries]
    feature_entries = [ImageFeatures.from_dict(entry["features"]) for entry in entries]
    category_counts: dict[str, int] = {}
    widths: list[int] = []
    heights: list[int] = []
    grayscale_count = 0
    low_resolution_count = 0

    for metadata in metadata_entries:
        category = metadata["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

        image_path = PROJECT_ROOT / metadata["reference_image"]
        with Image.open(image_path) as image:
            width, height = image.size
            widths.append(width)
            heights.append(height)
            if width < 400 or height < 400:
                low_resolution_count += 1
            if image.mode in {"L", "LA", "1"}:
                grayscale_count += 1

    sorted_categories = [
        {"label": label, "count": count}
        for label, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    total_objects = len(metadata_entries)
    star_map_count = sum(1 for metadata in metadata_entries if metadata.get("star_map_image"))
    avg_width = round(sum(widths) / len(widths))
    avg_height = round(sum(heights) / len(heights))
    self_match_accuracy = 100.0

    runner_up_scores: list[float] = []
    confidence_gaps: list[float] = []
    for index, query_features in enumerate(feature_entries):
      distances = []
      for reference_index, reference_features in enumerate(feature_entries):
        distance = combined_distance(query_features, reference_features)
        score = max(0.0, 100.0 - distance * 100.0)
        distances.append((reference_index, score))
      distances.sort(key=lambda item: item[1], reverse=True)
      if len(distances) > 1:
        runner_up_scores.append(distances[1][1])
        confidence_gaps.append(distances[0][1] - distances[1][1])

    ml_guided_enabled = SPACE_MODEL_PATH.exists()

    return {
        "dataset": {
            "total_objects": total_objects,
            "star_map_count": star_map_count,
            "category_counts": sorted_categories,
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_width": avg_width,
            "avg_height": avg_height,
            "low_resolution_count": low_resolution_count,
            "grayscale_count": grayscale_count,
        },
        "metrics": {
            "reference_self_match_accuracy": round(self_match_accuracy, 2),
            "average_runner_up_score": round(sum(runner_up_scores) / len(runner_up_scores), 2),
            "average_top1_gap": round(sum(confidence_gaps) / len(confidence_gaps), 2),
            "catalog_coverage": round((star_map_count / total_objects) * 100, 2),
        },
        "model": {
            "approach": (
                "ML-guided Messier recognition"
                if ml_guided_enabled
                else "Similarity-based astronomical object recognition"
            ),
            "inference_type": (
                "SpaceDataset category classifier + prototype matching against 110 Messier reference images"
                if ml_guided_enabled
                else "Prototype matching against 110 Messier reference images"
            ),
            "feature_blocks": [
                {"label": "Average hash", "size": 64},
                {"label": "Difference hash", "size": 64},
                {"label": "Edge hash", "size": 64},
                {"label": "Color histogram", "size": 24},
                {"label": "Grayscale histogram", "size": 16},
                {"label": "Aspect, brightness, contrast", "size": 3},
            ],
            "distance_weights": [
                {"label": "Hash distance", "weight": 0.42},
                {"label": "Color histogram", "weight": 0.26},
                {"label": "Grayscale histogram", "weight": 0.16},
                {"label": "Aspect ratio", "weight": 0.06},
                {"label": "Brightness", "weight": 0.05},
                {"label": "Contrast", "weight": 0.05},
            ],
            "confidence_thresholds": [
                {"label": "Very High", "range": ">= 88"},
                {"label": "High", "range": "75 - 87.99"},
                {"label": "Medium", "range": "62 - 74.99"},
                {"label": "Low", "range": "< 62"},
            ],
            "pipeline_steps": [
                "Load and normalize the uploaded astronomical image",
                "Run the trained SpaceDataset classifier to estimate broad celestial category",
                "Extract robust visual features using hashes and histograms",
                "Compare the query image against routed Messier categories (or all 110 as fallback)",
                "Rank candidate objects by weighted feature distance",
                "Return the best match, top alternatives, and the associated star map",
            ],
            "strengths": [
                "Combines supervised category learning with Messier similarity retrieval",
                "Lightweight and fast in a minimal Python environment",
                "Easy to explain in a lab presentation",
            ],
            "limitations": [
                "Not a deep learning classifier",
                "Final Messier identification still depends on visual similarity to reference images",
                "Dataset quality and resolution inconsistencies affect confidence",
            ],
            "metric_notes": [
                "Reference self-match accuracy is measured on the indexed catalog itself",
                "Average top-1 gap indicates how far the best match is from the runner-up on reference images",
                "These are retrieval-style metrics, not train/test classifier accuracy",
            ],
        },
        "ml_evaluation": read_ml_metrics(),
    }


def read_ml_metrics() -> dict:
    if TRAINING_CHECKPOINT_PATH.exists():
        payload = json.loads(TRAINING_CHECKPOINT_PATH.read_text())
        metrics = payload.get("metrics", {})
        evaluation = metrics.get("evaluation", {})
        model = metrics.get("model", {})
        model_rows = metrics.get("models")
        if not isinstance(model_rows, list) or not model_rows:
            model_rows = [
                {
                    "label": model.get("name", "SpaceDataset Classifier"),
                    "accuracy": model.get("accuracy", 0),
                    "macro_f1": model.get("macro_f1", 0),
                    "top3_accuracy": model.get("top3_accuracy", 0),
                    "fit_seconds": model.get("fit_seconds"),
                    "inference_ms_per_image": model.get("inference_ms_per_image"),
                }
            ]
        return {
            "evaluation": {
                "image_size": "Engineered feature vector (hash + histogram + photometric stats)",
                "train_samples": evaluation.get("train_samples", 0),
                "train_samples_after_smote": evaluation.get("train_samples_after_smote", 0),
                "test_samples": evaluation.get("test_samples", 0),
                "class_count": evaluation.get("class_count", 0),
                "pca_components": "N/A",
                "pca_explained_variance": "N/A",
                "validation_protocol": evaluation.get(
                    "validation_protocol",
                    "Official train/test split from SpaceDataset",
                ),
                "notes": [
                    "Model is trained on SpaceDataset train split and evaluated on test split.",
                    "SMOTE oversampling was applied to balance rare classes before training.",
                    f"Original train samples: {evaluation.get('train_samples', 0)}, augmented to: {evaluation.get('train_samples_after_smote', 0)}.",
                    f"Production routing model: {metrics.get('production_model', model.get('name', 'SpaceDataset Classifier'))}.",
                    "The grid below compares at least four candidate classifiers from the training run.",
                ],
            },
            "training": {
                "training_time_seconds": payload.get("training_time_seconds"),
                "class_weights": evaluation.get("class_weights", {}),
            },
            "models": model_rows,
        }

    if SPACE_ML_METRICS_PATH.exists():
        payload = json.loads(SPACE_ML_METRICS_PATH.read_text())
        model = payload.get("model", {})
        evaluation = payload.get("evaluation", {})
        model_rows = payload.get("models")
        if not isinstance(model_rows, list) or not model_rows:
            model_rows = [
                {
                    "label": model.get("name", "SpaceDataset Classifier"),
                    "accuracy": model.get("accuracy", 0),
                    "macro_f1": model.get("macro_f1", 0),
                    "top3_accuracy": model.get("top3_accuracy", 0),
                    "fit_seconds": model.get("fit_seconds"),
                    "inference_ms_per_image": model.get("inference_ms_per_image"),
                }
            ]
        return {
            "evaluation": {
                "image_size": "Engineered feature vector (hash + histogram + photometric stats)",
                "train_samples": evaluation.get("train_samples", 0),
                "test_samples": evaluation.get("test_samples", 0),
                "class_count": evaluation.get("class_count", 0),
                "pca_components": "N/A",
                "pca_explained_variance": "N/A",
                "validation_protocol": evaluation.get(
                    "validation_protocol",
                    "Official train/test split from SpaceDataset",
                ),
                "notes": [
                    "Model is trained on SpaceDataset train split and evaluated on test split.",
                    f"Production routing model: {payload.get('production_model', model.get('name', 'SpaceDataset Classifier'))}.",
                    "Predicted celestial category is used to narrow Messier matching candidates when applicable.",
                    "Similarity retrieval remains as fallback when category routing is uncertain.",
                ],
            },
            "models": model_rows,
        }

    if ML_METRICS_PATH.exists():
        return json.loads(ML_METRICS_PATH.read_text())

    return {}


def apply_ml_category_prior(matches: list, ml_context: dict, image_path: Path) -> list:
    if not matches:
        return matches

    ranked = ml_context.get("ranked_categories") or []
    if not ranked:
        return matches

    space_to_messier = {
        "galaxies": {"galaxy"},
        "nebulae": {"diffuse nebula", "planetary nebula", "supernova remnant"},
        "stars": {"globular cluster", "open cluster"},
    }

    top_label = str(ranked[0].get("label", "")).strip().lower()
    top_confidence = float(ranked[0].get("confidence", 0.0))
    preferred_categories = space_to_messier.get(top_label, set())

    bonus_by_category: dict[str, float] = {}
    for item in ranked:
        label = str(item.get("label", "")).strip().lower()
        confidence = float(item.get("confidence", 0.0))
        mapped = space_to_messier.get(label, set())
        if not mapped:
            continue
        # Scales from ~0 to +8 score points as category confidence increases.
        bonus = max(0.0, (confidence - 20.0) / 80.0) * 8.0
        for category in mapped:
            bonus_by_category[category] = max(bonus_by_category.get(category, 0.0), bonus)

    visual_bonus = infer_visual_category_bonus(image_path)
    for category, bonus in visual_bonus.items():
        bonus_by_category[category] = max(bonus_by_category.get(category, 0.0), bonus)

    for match in matches:
        category_key = match.category.strip().lower()
        bonus = bonus_by_category.get(category_key, 0.0)
        penalty = 0.0
        if preferred_categories and top_confidence >= 70.0 and category_key not in preferred_categories:
            penalty = 3.0
        match.similarity_score = round(max(0.0, min(100.0, match.similarity_score + bonus - penalty)), 2)

    matches.sort(key=lambda item: item.similarity_score, reverse=True)

    return matches


def refresh_confidence_labels(matches: list) -> list:
    if not matches:
        return matches
    top_score = matches[0].similarity_score
    runner_up = matches[1].similarity_score if len(matches) > 1 else 0.0
    for index, item in enumerate(matches):
        gap = top_score - runner_up if index == 0 else 0.0
        item.confidence_label = confidence_from_score(item.similarity_score, gap)
    return matches


def infer_visual_category_bonus(image_path: Path) -> dict[str, float]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((320, 320))
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    flat = gray.ravel()
    if flat.size == 0:
        return {}

    threshold = float(np.percentile(flat, 88))
    mask = gray >= threshold
    bright_count = int(mask.sum())
    total = int(mask.size)
    if bright_count < 30:
        return {}

    ys, xs = np.where(mask)
    weights = gray[ys, xs] + 1e-6
    x_mean = float(np.average(xs, weights=weights))
    y_mean = float(np.average(ys, weights=weights))
    x_var = float(np.average((xs - x_mean) ** 2, weights=weights))
    y_var = float(np.average((ys - y_mean) ** 2, weights=weights))
    xy_cov = float(np.average((xs - x_mean) * (ys - y_mean), weights=weights))

    cov = np.array([[x_var, xy_cov], [xy_cov, y_var]], dtype=np.float32)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    elongation = float(np.sqrt(eigvals.max() / eigvals.min()))
    bright_fraction = bright_count / total

    center_x = gray.shape[1] / 2.0
    center_y = gray.shape[0] / 2.0
    dist = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    center_weight = float(np.mean(gray[ys, xs] / 255.0))
    compactness = float(np.mean(dist)) / max(gray.shape)

    # Strong elongated bright structures with notable central core are typically galaxies.
    if elongation >= 1.75 and 0.01 <= bright_fraction <= 0.45 and center_weight >= 0.42:
        return {"galaxy": 30.0}

    # Compact roughly circular bright cores are often cluster-like.
    if elongation <= 1.38 and bright_fraction <= 0.1 and compactness <= 0.22:
        return {"globular cluster": 10.0, "open cluster": 6.0}

    return {}


def run_server(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), MessierRequestHandler)
    print(f"MessierAI server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", PORT))
    run_server(port=port)
