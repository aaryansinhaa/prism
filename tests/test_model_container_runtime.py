from model_container.runtime import predict_usage


def test_predict_usage_returns_post_instructions():
    response = predict_usage()

    assert response["detail"] == "Use POST /predict with JSON body."
    assert response["example_request"]["method"] == "POST"
    assert response["example_request"]["path"] == "/predict"
    assert "input" in response["example_request"]["json"]
    assert response["input_schema"]["type"] == "object"
