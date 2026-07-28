from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .features import extract_features


class XGBClassifierWithLabelEncoding(BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.encoder = LabelEncoder()
        self.model = None

    def fit(self, X, y):
        # Encode string labels to numeric
        y_encoded = self.encoder.fit_transform(y)

        # Create and fit XGBoost model
        self.model = xgb.XGBClassifier(**self.kwargs)
        self.model.fit(X, y_encoded)
        self.fitted_ = True  # Mark as fitted for sklearn compatibility
        return self

    def predict(self, X):
        # Get numeric predictions
        y_encoded_pred = self.model.predict(X)
        # Convert back to string labels
        return self.encoder.inverse_transform(y_encoded_pred)

    def predict_proba(self, X):
        # Get probability predictions
        proba = self.model.predict_proba(X)
        return proba


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORY_TO_MESSIER = {
    "galaxies": ["Galaxy"],
    "nebulae": ["Diffuse nebula", "Planetary nebula", "Supernova remnant"],
    "stars": ["Globular cluster", "Open cluster"],
}


def normalize_category(category: str) -> str:
    return category.strip().lower().replace(" ", "_")


def image_to_vector(image_path: Path | str) -> np.ndarray:
    features = extract_features(image_path)
    hash_values = [1.0 if bit == "1" else 0.0 for bit in (features.average_hash + features.difference_hash + features.edge_hash)]
    numeric_values = (
        features.color_histogram
        + features.grayscale_histogram
        + [features.aspect_ratio, features.brightness, features.contrast]
    )
    vector = np.array(hash_values + numeric_values, dtype=np.float32)
    return vector


def _collect_split(split_root: Path, cache_path: Optional[Path] = None) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    
    # Check if we have cached features
    if cache_path and cache_path.exists():
        print(f"Loading cached features from: {cache_path}", flush=True)
        try:
            cached = np.load(str(cache_path), allow_pickle=True)
            return cached['features'], cached['labels']
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}", flush=True)

    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    image_paths: list[tuple[str, Path]] = []
    for category_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        category_label = normalize_category(category_dir.name)
        for image_path in sorted(category_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_paths.append((category_label, image_path))

    total_images = len(image_paths)
    print(f"Loading {total_images} images from: {split_root}", flush=True)

    for index, (category_label, image_path) in enumerate(image_paths, start=1):
        features.append(image_to_vector(image_path))
        labels.append(category_label)
        if index == 1 or index % 250 == 0 or index == total_images:
            print(f"  Processed {index}/{total_images}", flush=True)

    if not features:
        raise ValueError(f"No valid images found under {split_root}")

    features_array = np.stack(features)
    labels_array = np.array(labels)
    
    # Save cache for future runs
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path) + ".tmp", {"features": features_array, "labels": labels_array}, allow_pickle=True)
            import shutil
            shutil.move(str(cache_path) + ".tmp", str(cache_path))
            print(f"Cached features to: {cache_path}", flush=True)
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}", flush=True)

    return features_array, labels_array


def train_and_evaluate(
    dataset_images_root: Path | str,
    model_output_path: Path | str,
) -> dict:
    dataset_images_root = Path(dataset_images_root)
    model_output_path = Path(model_output_path)

    train_root = dataset_images_root / "train" / "celestial objects"
    test_root = dataset_images_root / "test" / "celestial objects"
    
    # Cache paths to avoid re-extracting features if training fails
    train_cache_path = Path("artifacts/train_features_cache.npy")
    test_cache_path = Path("artifacts/test_features_cache.npy")

    x_train, y_train = _collect_split(train_root, cache_path=train_cache_path)
    x_test, y_test = _collect_split(test_root, cache_path=test_cache_path)
    
    # Apply SMOTE to balance training data (on top of class weights)
    print("Applying SMOTE oversampling to training data...", flush=True)
    smote = SMOTE(random_state=42, k_neighbors=5)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    print(f"  Original train samples: {len(x_train)}", flush=True)
    print(f"  SMOTE balanced train samples: {len(x_train_balanced)}", flush=True)
    
    # Calculate class weights from original data for reference
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    print(f"Class weights (for reference):", flush=True)
    for cls in sorted(class_weight_dict.keys()):
        original_count = np.sum(y_train == cls)
        print(f"  {cls} (n={original_count}): weight={class_weight_dict[cls]:.4f}", flush=True)
    
    print("Fitting comparative models...", flush=True)

    models: dict[str, Pipeline] = {
        "SGD Logistic (Production)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-4,
                        max_iter=5000,
                        tol=1e-3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        ),
        "Linear SVM (Balanced)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(C=1.0, random_state=42, class_weight="balanced", max_iter=3000)),
            ]
        ),
        "Random Forest (Balanced)": Pipeline(
            steps=[
                ("clf", RandomForestClassifier(
                    n_estimators=350, 
                    random_state=42, 
                    n_jobs=-1,
                    class_weight="balanced",
                    max_depth=25,
                    min_samples_split=5,
                    min_samples_leaf=2,
                )),
            ]
        ),
        "KNN (k=7)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
            ]
        ),
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        models["XGBoost (Balanced)"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifierWithLabelEncoding(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric="mlogloss",
                        verbosity=0,
                    ),
                ),
            ]
        )

    model_results: list[dict] = []
    production_model = None
    production_predictions = None

    for label, model in models.items():
        try:
            print(f"  Training: {label}", flush=True)
            start_fit = time.perf_counter()
            model.fit(x_train_balanced, y_train_balanced)
            fit_seconds = time.perf_counter() - start_fit
            print(f"    ✓ Fit complete ({fit_seconds:.2f}s), predicting...", flush=True)

            y_pred = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
            weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
            top3_accuracy = float(top_k_accuracy(model, x_test, y_test, k=3))

            sample_size = min(1500, len(x_test))
            start_infer = time.perf_counter()
            model.predict(x_test[:sample_size])
            infer_seconds = time.perf_counter() - start_infer
            infer_ms_per_image = (infer_seconds / sample_size) * 1000 if sample_size else 0.0

            result = {
                "label": label,
                "accuracy": round(accuracy * 100, 2),
                "macro_f1": round(macro_f1 * 100, 2),
                "weighted_f1": round(weighted_f1 * 100, 2),
                "top3_accuracy": round(top3_accuracy * 100, 2),
                "fit_seconds": round(fit_seconds, 2),
                "inference_ms_per_image": round(infer_ms_per_image, 4),
            }
            model_results.append(result)

            if label == "SGD Logistic (Production)":
                production_model = model
                production_predictions = y_pred
        except Exception as e:
            print(f"    ✗ {label} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    if production_model is None or production_predictions is None:
        raise RuntimeError("Production model training failed.")

    class_report = classification_report(
        y_test,
        production_predictions,
        output_dict=True,
        zero_division=0,
    )
    model_results.sort(key=lambda item: (item["accuracy"], item["macro_f1"]), reverse=True)

    production_summary = next(item for item in model_results if item["label"] == "SGD Logistic (Production)")

    metrics = {
        "evaluation": {
            "dataset_root": str(dataset_images_root),
            "train_samples": int(len(y_train)),
            "train_samples_after_smote": int(len(y_train_balanced)),
            "test_samples": int(len(y_test)),
            "class_count": int(len(sorted(set(y_train.tolist())))),
            "feature_size": int(x_train.shape[1]),
            "validation_protocol": "Official train/test split from SpaceDataset with SMOTE + class balancing",
            "class_weights": class_weight_dict,
        },
        "model": {
            "name": production_summary["label"],
            "accuracy": production_summary["accuracy"],
            "macro_f1": production_summary["macro_f1"],
            "weighted_f1": production_summary["weighted_f1"],
            "top3_accuracy": production_summary["top3_accuracy"],
            "fit_seconds": production_summary["fit_seconds"],
            "inference_ms_per_image": production_summary["inference_ms_per_image"],
        },
        "models": model_results,
        "production_model": "SGD Logistic (Production)",
        "classes": sorted(set(y_train.tolist())),
        "class_metrics": {
            label: {
                "precision": round(stats["precision"] * 100, 2),
                "recall": round(stats["recall"] * 100, 2),
                "f1_score": round(stats["f1-score"] * 100, 2),
                "support": int(stats["support"]),
            }
            for label, stats in class_report.items()
            if isinstance(stats, dict) and "precision" in stats
        },
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with model_output_path.open("wb") as handle:
        pickle.dump(production_model, handle)

    return metrics


def top_k_accuracy(estimator, features: np.ndarray, targets: np.ndarray, k: int = 3) -> float:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
    else:
        predictions = estimator.predict(features)
        return float(np.mean(predictions == targets))

    top_indices = np.argsort(scores, axis=1)[:, -k:]
    classes = estimator.classes_
    matches = 0
    for index, target in enumerate(targets):
        top_labels = classes[top_indices[index]]
        if target in top_labels:
            matches += 1
    return matches / len(targets)


class SpaceCategoryModel:
    def __init__(self, model_path: Path | str):
        model_path = Path(model_path)
        with model_path.open("rb") as handle:
            self.model = pickle.load(handle)

    def predict(self, image_path: Path | str) -> dict:
        vector = image_to_vector(image_path).reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            classes = self.model.classes_
            top_idx = int(np.argmax(probabilities))
            top_label = str(classes[top_idx])
            confidence = float(probabilities[top_idx])
            ranked = sorted(
                (
                    {"label": str(label), "confidence": round(float(prob) * 100, 2)}
                    for label, prob in zip(classes, probabilities, strict=False)
                ),
                key=lambda item: item["confidence"],
                reverse=True,
            )
        else:
            top_label = str(self.model.predict(vector)[0])
            confidence = 0.0
            ranked = [{"label": top_label, "confidence": 0.0}]

        messier_categories = CATEGORY_TO_MESSIER.get(top_label, [])

        return {
            "predicted_category": top_label,
            "confidence": round(confidence * 100, 2),
            "messier_categories": messier_categories,
            "ranked_categories": ranked[:3],
        }
