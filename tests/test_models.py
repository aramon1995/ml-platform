import numpy as np
import optuna
import pandas as pd
import pytest

from mlplatform.features import FEATURE_SETS
from mlplatform.models import MODELS


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 120
    y = pd.Series(rng.choice(["A", "B", "C"], n))
    months = y.map({"A": "01", "B": "04", "C": "07"})
    X = pd.DataFrame({
        "Timestamp": "2025-" + months + "-15 10:00:00",
        "F22": y.map({"A": 10.0, "B": 20.0, "C": 30.0}) + rng.normal(0, 2, n),
        "F17": np.where(rng.random(n) < 0.9, np.nan, rng.normal(size=n)),
    })
    return X, y


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_suggests_builds_and_predicts_labels(name, data):
    X, y = data
    module = MODELS[name]
    params = module.suggest(optuna.trial.FixedTrial(_fixed(name)))
    pipeline = module.build_pipeline(params, FEATURE_SETS["eda_minimal"]).fit(X, y)
    assert set(pipeline.predict(X)) <= set(y)
    assert (pipeline.predict(X) == y).mean() > 0.9
    assert pipeline.predict_proba(X).shape == (len(X), 3)


def _fixed(name: str) -> dict:
    return {
        "logreg": {"C": 1.0},
        "svm": {"C": 1.0, "gamma": 0.1},
        "random_forest": {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 1},
        "xgboost": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0},
        "mlp": {"n_layers": 1, "units_0": 32, "activation": "ReLU", "dropout": 0.0, "lr": 1e-2, "weight_decay": 1e-4},
    }[name]
