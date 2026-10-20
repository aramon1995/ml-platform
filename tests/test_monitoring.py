import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from mlplatform.features import FEATURE_SETS
from mlplatform.models import logreg
from mlplatform.monitoring import category_psi, evaluate, numeric_psi, reference_profile, window_score

MONTH_OF = {"A": "01", "B": "04", "C": "07"}
LEVEL_OF = {"A": 10.0, "B": 20.0, "C": 30.0}


def sample(n, rng, months=MONTH_OF, f22_shift=0.0):
    y = pd.Series(rng.choice(list(MONTH_OF), n))
    X = pd.DataFrame({
        "Timestamp": "2025-" + y.map(months) + "-15 10:00:00",
        "F22": y.map(LEVEL_OF) + rng.normal(0, 2, n) + f22_shift,
        "F17": np.where(rng.random(n) < 0.9, np.nan, rng.normal(size=n)),
    })
    return X, y


@pytest.fixture
def trained():
    rng = np.random.default_rng(0)
    X, y = sample(300, rng)
    model = logreg.build_pipeline({"C": 1.0}, FEATURE_SETS["eda_minimal"]).fit(X, y)
    X_test, y_test = sample(100, rng)
    profile = reference_profile(X, model, X_test, f1_score(y_test, model.predict(X_test), average="macro"), "f1_macro")
    return model, profile, rng


def window(model, X, actual=None):
    proba = model.predict_proba(X)
    return pd.DataFrame({
        "features": X.astype(object).where(X.notna(), None).to_dict("records"),
        "predicted_value": model.classes_[proba.argmax(axis=1)],
        "probabilities": [dict(zip(model.classes_, map(float, row))) for row in proba],
        "actual_value": actual if actual is not None else None,
    })


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=2000)
    assert numeric_psi(reference, rng.normal(size=2000)) < 0.05
    assert numeric_psi(reference, rng.normal(loc=1.5, size=2000)) > 0.25
    assert category_psi({"a": 0.5, "b": 0.5}, pd.Series(["a", "b"] * 50)) == pytest.approx(0, abs=1e-9)
    assert category_psi({"a": 0.5, "b": 0.5}, pd.Series(["c"] * 100)) > 0.25


def test_a_window_like_the_training_data_raises_no_alert(trained):
    model, profile, rng = trained
    X, y = sample(200, rng)
    checks = statuses(evaluate(profile, window(model, X, y)))
    assert "alert" not in checks.values()
    assert checks["performance_drop"] == "ok"


def test_unseen_months_and_shifted_f22_raise_drift_alerts(trained):
    model, profile, rng = trained
    X, _ = sample(200, rng, months={"A": "12", "B": "12", "C": "02"}, f22_shift=15.0)
    checks = statuses(evaluate(profile, window(model, X)))
    assert checks["rare_month_share"] == checks["psi.month"] == checks["psi.F22"] == "alert"
    assert checks["performance_drop"] == "insufficient_data"


def test_wrong_predictions_raise_a_performance_alert(trained):
    model, profile, rng = trained
    X, y = sample(200, rng)
    checks = statuses(evaluate(profile, window(model, X, y.map({"A": "B", "B": "C", "C": "A"}))))
    assert checks["performance_drop"] == "alert"


def test_small_windows_are_not_evaluated(trained):
    model, profile, rng = trained
    X, _ = sample(10, rng)
    assert statuses(evaluate(profile, window(model, X))) == {"window_size": "insufficient_data"}


def test_window_score_replays_stored_predictions(trained):
    model, _, rng = trained
    X, y = sample(100, rng)
    labeled = window(model, X, y)
    assert window_score(labeled, "f1_macro") == pytest.approx(f1_score(y, model.predict(X), average="macro"))
    assert 0 <= window_score(labeled, "roc_auc_ovr") <= 1


def test_category_psi_pools_rare_and_unseen_categories():
    reference = {"a": 0.495, "b": 0.495, "c": 0.01}
    assert category_psi(reference, pd.Series(["a", "b"] * 49 + ["z", "c"]), min_freq=0.02) < 0.01
    assert category_psi(reference, pd.Series(["a", "b"] * 49 + ["z", "c"])) > 0.02
