import numpy as np
import pandas as pd
import pytest

from mlplatform.inference import predict
from mlplatform.models import logreg


@pytest.fixture
def fitted():
    y = pd.Series(["A", "B"] * 20)
    X = pd.DataFrame({
        "Timestamp": np.where(y == "A", "2025-01-10", "2025-04-10"),
        "F22": np.where(y == "A", 10.0, 20.0),
        "F17": np.nan,
    })
    return logreg.build_pipeline({"C": 1.0}).fit(X, y), X


def test_predict_returns_label_probabilities_and_raw_features(fitted):
    model, X = fitted
    out = predict(model, X.assign(F17=[np.nan] * 39 + [1.5]))
    assert list(out.columns) == ["predicted_value", "probabilities", "features"]
    assert out["predicted_value"].tolist()[:2] == ["A", "B"]
    assert set(out["probabilities"][0]) == {"A", "B"} and abs(sum(out["probabilities"][0].values()) - 1) < 1e-9
    assert out["features"][0]["F17"] is None and out["features"][39]["F17"] == 1.5


def test_predict_rejects_batches_missing_columns(fitted):
    model, X = fitted
    with pytest.raises(ValueError, match="F22"):
        predict(model, X.drop(columns=["F22"]))
