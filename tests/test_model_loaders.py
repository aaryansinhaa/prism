from pathlib import Path

import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.adapters.base import ModelPredictError
from runtime.adapters.onnx_adapters import ONNXModel
from runtime.adapters.scikitlearn_adapters import SklearnModel
from runtime.model_loaders import detect_model_type


def test_detect_model_type_by_extension():
    assert detect_model_type("model.onnx") == "onnx"
    assert detect_model_type("model.pkl") == "sklearn"
    assert detect_model_type("model.joblib") == "sklearn"
    assert detect_model_type("model.txt") is None


def test_onnx_validate_input_accepts_single_input_alias():
    model = ONNXModel.__new__(ONNXModel)
    model._input_names = ["feature"]

    payload = model.validate_input({"input": [1, 2, 3]})

    assert payload["feature"] == [1, 2, 3]
    assert payload["input"] == [1, 2, 3]


def test_onnx_validate_input_rejects_missing_inputs():
    model = ONNXModel.__new__(ONNXModel)
    model._input_names = ["feature"]

    with pytest.raises(ModelPredictError):
        model.validate_input({"other": [1, 2, 3]})


def test_sklearn_validate_input_maps_input_alias():
    model = SklearnModel.__new__(SklearnModel)

    payload = model.validate_input({"input": [1, 2, 3]})

    assert payload["inputs"] == [1, 2, 3]


def test_sklearn_validate_input_requires_features():
    model = SklearnModel.__new__(SklearnModel)

    with pytest.raises(ModelPredictError):
        model.validate_input({"other": [1, 2, 3]})
