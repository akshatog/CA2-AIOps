"""These tests use YOUR exported Colab model and saved prediction."""
import json

import pytest
from fastapi.testclient import TestClient

from main import ARTIFACT_DIR, app


@pytest.fixture(scope="module")
def sample():
    return json.loads((ARTIFACT_DIR / "sample_request.json").read_text(encoding="utf-8-sig"))


@pytest.fixture(scope="module")
def client():
    # The context manager runs the app's model-loading startup.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_prediction_matches_colab(client, sample):
    expected = json.loads((ARTIFACT_DIR / "expected_prediction.json").read_text(encoding="utf-8-sig"))
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["predicted_quality"] == pytest.approx(
        expected["predicted_quality"], rel=1e-8, abs=1e-8
    )


def test_json_key_order_does_not_change_prediction(client, sample):
    reversed_sample = dict(reversed(list(sample.items())))
    normal = client.post("/predict", json=sample)
    reordered = client.post("/predict", json=reversed_sample)
    assert normal.status_code == reordered.status_code == 200
    assert normal.json() == reordered.json()


@pytest.mark.parametrize("case", ["missing", "negative", "text", "boolean", "extra", "null", "ph"])
def test_invalid_measurements_are_rejected(client, sample, case):
    bad = sample.copy()
    if case == "missing":
        bad.pop("alcohol")
    elif case == "negative":
        bad["alcohol"] = -1
    elif case == "text":
        bad["alcohol"] = "twelve"
    elif case == "boolean":
        bad["alcohol"] = True
    elif case == "extra":
        bad["quality"] = 6
    elif case == "null":
        bad["alcohol"] = None
    elif case == "ph":
        bad["ph"] = 15
    assert client.post("/predict", json=bad).status_code == 422


def test_integer_measurements_are_accepted(client, sample):
    data = sample.copy()
    data["alcohol"] = 12
    assert client.post("/predict", json=data).status_code == 200
