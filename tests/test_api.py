from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlplatform import api
from mlplatform.features import FEATURE_SETS
from mlplatform.models import logreg


@pytest.fixture
def client(monkeypatch):
    y = pd.Series(["A", "B"] * 20)
    X = pd.DataFrame({"Timestamp": np.where(y == "A", "2025-01-10", "2025-04-10"), "F22": 10.0, "F17": np.nan})
    model = logreg.build_pipeline({"C": 1.0}, FEATURE_SETS["eda_minimal"]).fit(X, y)
    version = SimpleNamespace(version="3", tags={"model": "logreg", "feature_set": "eda_minimal"}, run_id="run")
    stored = []
    monkeypatch.setattr(api, "load_production", lambda: (model, version))
    monkeypatch.setattr(api, "store", stored.append)
    with TestClient(api.app) as client:
        client.stored = stored
        yield client
    api.state.clear()


def test_predict_uses_the_loaded_model_and_stores_the_rows(client):
    response = client.post("/predict", json={"rows": [{"Timestamp": "2025-04-01", "F22": 10.0, "F17": None, "F99": "extra"}]})
    body = response.json()
    assert response.status_code == 200
    assert body["predictions"][0]["predicted_value"] == "B"
    assert body["model_version"] == 3 and body["feature_set"] == "eda_minimal"
    assert client.stored[0]["input_uri"].tolist() == ["api"]


def test_predict_rejects_rows_missing_required_columns(client):
    response = client.post("/predict", json={"rows": [{"Timestamp": "2025-04-01"}]})
    assert response.status_code == 422 and "F22" in response.json()["detail"]


def test_health_and_reload_report_the_model(client):
    assert client.get("/health").json()["model"]["required_columns"] == ["Timestamp", "F22", "F17"]
    assert client.post("/reload").json()["model_family"] == "logreg"


def test_root_serves_the_predict_page(client):
    response = client.get("/")
    assert response.status_code == 200 and "mlplatform · predict" in response.text
