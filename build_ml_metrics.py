from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "ml_metrics.json"
IMAGE_SIZE = (48, 48)
PCA_COMPONENTS = 64


def load_base_images() -> list[tuple[str, str, Path]]:
    rows = []
    for image_path in sorted(DATASET_ROOT.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rows.append(
            (
                image_path.stem.upper(),
                image_path.parent.name.replace("_", " "),
                image_path,
            )
        )
    return rows


def preprocess_image(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    resized = grayscale.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array.flatten()


def generate_variants(image_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    base = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    train_variants = [
        base,
        ImageEnhance.Contrast(base).enhance(1.15),
        ImageEnhance.Brightness(base).enhance(0.92),
        base.filter(ImageFilter.SHARPEN),
    ]

    test_variants = [
        ImageEnhance.Contrast(base).enhance(0.88),
        ImageEnhance.Brightness(base).enhance(1.08),
        base.filter(ImageFilter.SMOOTH),
    ]

    train_features = [preprocess_image(image) for image in train_variants]
    test_features = [preprocess_image(image) for image in test_variants]
    return train_features, test_features


def top_k_accuracy(estimator, features: np.ndarray, targets: np.ndarray, k: int = 3) -> float:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    else:
        scores = estimator.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

    top_indices = np.argsort(scores, axis=1)[:, -k:]
    matches = 0
    for index, target in enumerate(targets):
        if target in top_indices[index]:
            matches += 1
    return matches / len(targets)


def build_dataset():
    base_images = load_base_images()
    labels: list[str] = []
    categories: dict[str, str] = {}
    x_train: list[np.ndarray] = []
    y_train: list[str] = []
    x_test: list[np.ndarray] = []
    y_test: list[str] = []

    for messier_id, category, image_path in base_images:
        categories[messier_id] = category
        labels.append(messier_id)
        train_variants, test_variants = generate_variants(image_path)
        x_train.extend(train_variants)
        y_train.extend([messier_id] * len(train_variants))
        x_test.extend(test_variants)
        y_test.extend([messier_id] * len(test_variants))

    label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    y_train_idx = np.array([label_to_index[label] for label in y_train], dtype=np.int32)
    y_test_idx = np.array([label_to_index[label] for label in y_test], dtype=np.int32)

    return (
        np.stack(x_train),
        y_train_idx,
        np.stack(x_test),
        y_test_idx,
        label_to_index,
        categories,
    )


def evaluate_models():
    x_train, y_train, x_test, y_test, label_to_index, categories = build_dataset()

    pca_probe = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=True, random_state=42)
    pca_probe.fit(x_train)

    models = {
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=True, random_state=42)),
                ("clf", KNeighborsClassifier(n_neighbors=3, weights="distance")),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=True, random_state=42)),
                ("clf", SVC(kernel="rbf", probability=True, gamma="scale", C=8.0, random_state=42)),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=True, random_state=42)),
                ("clf", LogisticRegression(max_iter=2500, solver="lbfgs")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=True, random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=True, random_state=42)),
                ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", max_iter=700, random_state=42)),
            ]
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        top3 = top_k_accuracy(model, x_test, y_test, k=3)
        results.append(
            {
                "label": name,
                "accuracy": round(accuracy * 100, 2),
                "macro_f1": round(macro_f1 * 100, 2),
                "top3_accuracy": round(top3 * 100, 2),
            }
        )

    results.sort(key=lambda item: item["accuracy"], reverse=True)

    payload = {
        "evaluation": {
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} grayscale",
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "class_count": len(label_to_index),
            "pca_components": PCA_COMPONENTS,
            "pca_explained_variance": round(float(np.sum(pca_probe.explained_variance_ratio_) * 100), 2),
            "validation_protocol": "Augmented hold-out evaluation derived from the Messier reference catalog",
            "notes": [
                "Each Messier object contributes multiple train and test image variants generated from its reference image.",
                "Metrics represent comparative classifier behavior on sparse-data astronomical recognition.",
                "The deployed application still uses the similarity engine because it is the most reliable for one-reference-per-object inference.",
            ],
        },
        "models": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote ML metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_models()
