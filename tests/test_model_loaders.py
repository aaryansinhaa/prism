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


def test_onnx_predict_reshapes_1d_to_row_vector_for_two_dim_input():
    class _FakeTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class _FakeSession:
        def __init__(self):
            self.last_feed = None

        def get_inputs(self):
            return [_FakeTensor("X", [None, 4])]

        def get_outputs(self):
            return [_FakeTensor("Y", [None])]

        def run(self, output_names, feed_dict):
            self.last_feed = feed_dict
            return [__import__("numpy").array([2])]

    fake_session = _FakeSession()
    model = ONNXModel(session=fake_session)

    result = model.predict({"X": [6.2, 3.4, 5.4, 2.3]})

    assert "predictions" in result
    assert fake_session.last_feed is not None
    assert fake_session.last_feed["X"].shape == (1, 4)


def test_sklearn_validate_input_maps_input_alias():
    model = SklearnModel.__new__(SklearnModel)

    payload = model.validate_input({"input": [1, 2, 3]})

    assert payload["inputs"] == [1, 2, 3]


def test_sklearn_validate_input_requires_features():
    model = SklearnModel.__new__(SklearnModel)

    with pytest.raises(ModelPredictError):
        model.validate_input({"other": [1, 2, 3]})
