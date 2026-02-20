"""ONNX model adapter for PRISM runtime."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping

import numpy as np

try:  # pragma: no cover - optional dependency
	import onnxruntime as ort
except ImportError:  # noqa: F401
	ort = None

if TYPE_CHECKING:  # pragma: no cover
	from onnxruntime import InferenceSession

from .base import BaseModel, ModelLoadError, ModelPredictError


def _to_serializable(value: Any) -> Any:
	"""Convert ONNXRuntime outputs into JSON-safe payloads."""

	if isinstance(value, np.ndarray):
		return value.tolist()

	if type(value).__name__ == "SparseTensor":  # pragma: no cover - optional type
		try:
			dense = value.toarray()
		except AttributeError:
			dense = None
		if dense is not None:
			return np.asarray(dense).tolist()

	try:
		return value.tolist()
	except Exception:  # noqa: BLE001 - fallback for non-array outputs
		return value


class ONNXModel(BaseModel):
	"""Adapter that executes ONNX graphs through onnxruntime."""

	def __init__(
		self,
		session: "InferenceSession",
		metadata: Dict[str, Any] | None = None,
	) -> None:
		super().__init__(metadata)
		self._session = session
		self._input_names = [tensor.name for tensor in session.get_inputs()]
		self._output_names = [tensor.name for tensor in session.get_outputs()]

	@classmethod
	def from_path(
		cls,
		model_path: str | Path,
		metadata: Dict[str, Any] | None = None,
		providers: list[str] | None = None,
		sess_options: Any | None = None,
	) -> "ONNXModel":
		if ort is None:
			raise ModelLoadError("onnxruntime is required to load ONNX models. Install `onnxruntime`.")

		path = Path(model_path).expanduser()
		if not path.exists():
			raise ModelLoadError(f"Model artifact not found: {path}")

		try:
			session = ort.InferenceSession(
				path.as_posix(),
				sess_options=sess_options,
				providers=providers or ort.get_available_providers(),
			)
		except Exception as exc:  # noqa: BLE001
			raise ModelLoadError(f"Failed to initialize ONNX runtime for {path}") from exc

		return cls(session=session, metadata=metadata)

	def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
		payload = input_data.get("inputs", input_data)
		if not isinstance(payload, Mapping):
			raise ModelPredictError("ONNXModel expects a mapping of input tensor names.")

		feed_dict: Dict[str, Any] = {}
		for name in self._input_names:
			if name not in payload:
				raise ModelPredictError(f"Missing input '{name}' for ONNX model.")
			feed_dict[name] = np.asarray(payload[name])

		try:
			outputs = self._session.run(self._output_names, feed_dict)
		except Exception as exc:  # noqa: BLE001
			raise ModelPredictError("ONNX inference failed") from exc

		predictions = {
			name: _to_serializable(value)
			for name, value in zip(self._output_names, outputs, strict=False)
		}

		return {"predictions": predictions}


__all__ = ["ONNXModel"]
