"""Scikit-learn model adapter for PRISM runtime."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import BaseModel, ModelLoadError, ModelPredictError


def _to_list(value: Any) -> Any:
	"""Convert numpy values into Python-native types."""

	if hasattr(value, "tolist"):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


class SklearnModel(BaseModel):
	"""Adapter that wraps a pickled scikit-learn estimator."""

	def __init__(self, estimator: Any, metadata: Dict[str, Any] | None = None) -> None:
		super().__init__(metadata)
		self._estimator = estimator

	@classmethod
	def from_path(
		cls,
		model_path: str | Path,
		metadata: Dict[str, Any] | None = None,
		**_: Any,
	) -> "SklearnModel":
		path = Path(model_path).expanduser()
		if not path.exists():
			raise ModelLoadError(f"Model artifact not found: {path}")

		try:
			with path.open("rb") as handle:
				estimator = pickle.load(handle)
		except (OSError, pickle.PickleError) as exc:
			raise ModelLoadError(f"Failed to load sklearn artifact from {path}") from exc

		if not hasattr(estimator, "predict"):
			raise ModelLoadError("Loaded object does not implement `predict`.")

		return cls(estimator=estimator, metadata=metadata)

	def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
		if "inputs" not in input_data:
			raise ModelPredictError("SklearnModel expects an 'inputs' key with iterable features.")

		features = np.asarray(input_data["inputs"])
		if features.ndim == 1:
			features = features.reshape(1, -1)

		try:
			predictions = self._estimator.predict(features)
		except Exception as exc:  # noqa: BLE001
			raise ModelPredictError("SklearnModel prediction failed") from exc

		result: Dict[str, Any] = {"predictions": _to_list(predictions)}

		if input_data.get("return_proba") and hasattr(self._estimator, "predict_proba"):
			try:
				probabilities = self._estimator.predict_proba(features)
			except Exception as exc:  # noqa: BLE001
				raise ModelPredictError("SklearnModel probability prediction failed") from exc
			result["probabilities"] = _to_list(probabilities)

		return result


__all__ = ["SklearnModel"]
