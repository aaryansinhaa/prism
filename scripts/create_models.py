#!/usr/bin/env python3
"""Create multiple classical ML models and export them to ONNX for PRISM.

This script intentionally focuses on classical scikit-learn models that are
served through the existing ONNX adapter path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = ROOT / "model_store"
REPORT_PATH = ROOT / "docs" / "benchmarks" / "classical-model-generation.json"

RNG_SEED = 42


@dataclass
class TrainedArtifact:
    model_id: str
    estimator: Any
    sample_input: list[list[float]]
    note: str


def _build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_class, y_class = make_classification(
        n_samples=800,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        class_sep=1.7,
        random_state=RNG_SEED,
    )
    x_cluster, _ = make_blobs(
        n_samples=800,
        n_features=4,
        centers=3,
        cluster_std=1.1,
        random_state=RNG_SEED,
    )
    return (
        x_class.astype(np.float32),
        y_class.astype(np.int64),
        x_cluster.astype(np.float32),
    )


def _build_models() -> list[TrainedArtifact]:
    x_class, y_class, x_cluster = _build_training_data()

    models: list[TrainedArtifact] = []

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, random_state=RNG_SEED),
    )
    logistic.fit(x_class, y_class)
    models.append(
        TrainedArtifact(
            model_id="logistic_regression",
            estimator=logistic,
            sample_input=x_class[:1].tolist(),
            note="Multiclass logistic regression with standard scaling.",
        )
    )

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    knn.fit(x_class, y_class)
    models.append(
        TrainedArtifact(
            model_id="knn_classifier",
            estimator=knn,
            sample_input=x_class[1:2].tolist(),
            note="K-nearest neighbors classifier (k=5).",
        )
    )

    svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=RNG_SEED
        ),
    )
    svm.fit(x_class, y_class)
    models.append(
        TrainedArtifact(
            model_id="svm_classifier",
            estimator=svm,
            sample_input=x_class[2:3].tolist(),
            note="RBF-kernel support vector classifier.",
        )
    )

    random_forest = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=RNG_SEED,
    )
    random_forest.fit(x_class, y_class)
    models.append(
        TrainedArtifact(
            model_id="random_forest_classifier",
            estimator=random_forest,
            sample_input=x_class[3:4].tolist(),
            note="Random forest classifier.",
        )
    )

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(x_class, y_class)
    models.append(
        TrainedArtifact(
            model_id="gaussian_nb",
            estimator=gaussian_nb,
            sample_input=x_class[4:5].tolist(),
            note="Gaussian Naive Bayes classifier.",
        )
    )

    kmeans = KMeans(n_clusters=3, random_state=RNG_SEED, n_init=15)
    kmeans.fit(x_cluster)
    models.append(
        TrainedArtifact(
            model_id="kmeans",
            estimator=kmeans,
            sample_input=x_cluster[:1].tolist(),
            note="KMeans clustering model (3 clusters).",
        )
    )

    return models


def _export_model(artifact: TrainedArtifact) -> dict[str, Any]:
    MODEL_STORE.mkdir(parents=True, exist_ok=True)

    pkl_path = MODEL_STORE / f"{artifact.model_id}.pkl"
    onnx_path = MODEL_STORE / f"{artifact.model_id}.onnx"

    joblib.dump(artifact.estimator, pkl_path)

    n_features = len(artifact.sample_input[0])
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(artifact.estimator, initial_types=initial_types)
    onnx_path.write_bytes(onnx_model.SerializeToString())

    sklearn_prediction = artifact.estimator.predict(
        np.asarray(artifact.sample_input, dtype=np.float32)
    ).tolist()

    return {
        "model_id": artifact.model_id,
        "pkl_path": str(pkl_path.relative_to(ROOT)),
        "onnx_path": str(onnx_path.relative_to(ROOT)),
        "sample_input": artifact.sample_input,
        "sample_sklearn_prediction": sklearn_prediction,
        "note": artifact.note,
    }


def main() -> None:
    artifacts = _build_models()
    exported = [_export_model(item) for item in artifacts]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "seed": RNG_SEED,
                "generated_models": exported,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Generated classical models:")
    for model in exported:
        print(f"- {model['model_id']}: {model['onnx_path']}")
    print(f"\nGeneration report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
