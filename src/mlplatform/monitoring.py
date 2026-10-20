"""Production monitoring: compare recent predictions with the reference profile stored at training time.

Every check measures how much worse the window is than the reference (larger is worse) and gets a status
from its warn/alert thresholds. Drift checks say the inputs or outputs changed; only `performance_drop`,
which needs ground-truth labels, says the model got worse. Nothing is retrained automatically.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from mlplatform.config import get_settings
from mlplatform.data import read_csv
from mlplatform.db import engine, ground_truth, monitoring_checks, predictions
from mlplatform.features import TIMESTAMP, extract_month
from mlplatform.inference import ALIAS
from mlplatform.selection import PRODUCTION_MODEL

EXPERIMENT = "monitoring"
PROFILE = "reference_profile.json"
SAMPLE_SIZE = 5000


@dataclass
class Check:
    name: str
    value: float | None
    warn: float
    alert: float
    status: str  # ok | warn | alert | insufficient_data
    detail: str = ""


def check(name: str, value: float, warn: float, alert: float, detail: str = "") -> Check:
    status = "alert" if value >= alert else "warn" if value >= warn else "ok"
    return Check(name, float(value), warn, alert, status, detail)


def skipped(name: str, detail: str, warn: float = 0.0, alert: float = 0.0) -> Check:
    return Check(name, None, warn, alert, "insufficient_data", detail)


# Drift measures


def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant shift."""
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(reference: np.ndarray, values: np.ndarray) -> float:
    """PSI over the reference deciles."""
    edges = np.unique(np.quantile(reference, np.linspace(0.1, 0.9, 9)))

    def proportions(x):
        return np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1) / len(x)

    return _psi(proportions(reference), proportions(values))


def category_psi(reference: dict[str, float], values: pd.Series, min_freq: float = 0.0) -> float:
    """PSI over categories; those below `min_freq` in the reference (or unseen) are pooled as one."""
    common = {c for c, f in reference.items() if f >= min_freq}
    pooled = values.astype(str).where(values.astype(str).isin(common), "__other__")
    reference = {c: f for c, f in reference.items() if c in common} | {
        "__other__": sum(f for c, f in reference.items() if c not in common)
    }
    actual = pooled.value_counts(normalize=True)
    categories = sorted(set(reference) | set(actual.index))
    return _psi(np.array([reference.get(c, 0.0) for c in categories]), actual.reindex(categories, fill_value=0.0).to_numpy())


def _frequencies(values: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in values.value_counts(normalize=True).items()}


# Reference profile (computed at training time, stored with the model)


def reference_profile(X_train: pd.DataFrame, model: Pipeline, X_test: pd.DataFrame, test_score: float, metric: str) -> dict:
    """Statistics of the training inputs and of the model's test predictions that monitoring compares against."""
    numeric = X_train.select_dtypes("number").columns
    proba = model.predict_proba(X_test)
    return {
        "metric": metric,
        "test_score": float(test_score),
        "null_rates": {c: float(r) for c, r in X_train.isna().mean().items()},
        "numeric": {
            c: X_train[c].dropna().sample(n=min(SAMPLE_SIZE, X_train[c].notna().sum()), random_state=0).tolist()
            for c in numeric
        },
        "month": _frequencies(extract_month(X_train)["month"]) if TIMESTAMP in X_train else None,
        "classes": _frequencies(pd.Series(model.classes_[proba.argmax(axis=1)])),
        "confidence": float(proba.max(axis=1).mean()),
    }


# Checks


class _StoredPredictions(ClassifierMixin, BaseEstimator):
    """Replays stored predictions so any sklearn scorer (label or probability based) can score them."""

    def __init__(self, predicted: np.ndarray, probabilities: np.ndarray, classes: np.ndarray):
        self.predicted = predicted
        self.probabilities = probabilities
        self.classes = classes
        self.classes_ = classes

    def predict(self, X):
        return self.predicted

    def predict_proba(self, X):
        return self.probabilities


def window_score(labeled: pd.DataFrame, metric: str) -> float:
    classes = np.array(sorted(labeled["probabilities"].iloc[0]))
    stored = _StoredPredictions(
        labeled["predicted_value"].to_numpy(),
        np.array([[p[c] for c in classes] for p in labeled["probabilities"]]),
        classes,
    )
    return get_scorer(metric)(stored, np.zeros((len(labeled), 1)), labeled["actual_value"].to_numpy())


def evaluate(profile: dict, window: pd.DataFrame) -> list[Check]:
    """`window`: one row per prediction with `features`, `predicted_value`, `probabilities` and `actual_value`."""
    s = get_settings()
    if len(window) < s.monitor_min_rows:
        return [skipped("window_size", f"{len(window)} predictions < {s.monitor_min_rows}")]

    X = pd.DataFrame(window["features"].tolist())
    checks = []
    for col, reference in profile["numeric"].items():
        values = pd.to_numeric(X[col], errors="coerce").dropna() if col in X else pd.Series(dtype=float)
        if len(values) < s.monitor_min_rows:  # PSI on a handful of values (e.g. a mostly-null column) is noise
            detail = f"{len(values)} non-null values < {s.monitor_min_rows}"
            checks.append(skipped(f"psi.{col}", detail, s.psi_warn, s.psi_alert))
            continue
        ks = ks_2samp(reference, values).statistic
        checks.append(check(f"psi.{col}", numeric_psi(np.array(reference), values.to_numpy()), s.psi_warn, s.psi_alert, f"KS={ks:.3f}"))

    if profile["month"] is not None:
        months = extract_month(X)["month"]
        psi_month = category_psi(profile["month"], months, min_freq=s.rare_month_freq)
        checks.append(check("psi.month", psi_month, s.psi_warn, s.psi_alert))
        known = {m for m, f in profile["month"].items() if f >= s.rare_month_freq}
        reference_share = sum(f for m, f in profile["month"].items() if m not in known)
        rare = months[~months.isin(known)]
        share = len(rare) / len(months)
        detail = f"{share:.1%} vs {reference_share:.1%} in train; rare months: {sorted(rare.unique())}"
        checks.append(check("rare_month_share", share - reference_share, s.rare_month_warn, s.rare_month_alert, detail))

    for col, reference_rate in profile["null_rates"].items():
        rate = X[col].isna().mean() if col in X else 1.0
        detail = f"{rate:.1%} vs {reference_rate:.1%}"
        checks.append(check(f"null_increase.{col}", rate - reference_rate, s.null_warn, s.null_alert, detail))

    checks.append(check("prediction_psi", category_psi(profile["classes"], window["predicted_value"]), s.psi_warn, s.psi_alert))
    confidence = window["probabilities"].map(lambda p: max(p.values())).mean()
    checks.append(check(
        "confidence_drop", profile["confidence"] - confidence, s.confidence_warn, s.confidence_alert,
        f"{confidence:.3f} vs {profile['confidence']:.3f}",
    ))

    labeled = window.dropna(subset=["actual_value"])
    if len(labeled) < s.monitor_min_labels:
        detail = f"{len(labeled)} labels < {s.monitor_min_labels}"
        checks.append(skipped("performance_drop", detail, s.performance_warn, s.performance_alert))
    else:
        score = window_score(labeled, profile["metric"])
        checks.append(check(
            "performance_drop", profile["test_score"] - score, s.performance_warn, s.performance_alert,
            f"{profile['metric']} {score:.3f} vs {profile['test_score']:.3f} on {len(labeled)} labels; "
            "if confirmed, add the labeled data to the dataset and run train-select",
        ))
    return checks


# Storage and orchestration-free entry points


def load_window(version, start: datetime, input_uri: str | None = None) -> pd.DataFrame:
    """Predictions of `version` since `start`, optionally only those of one input file, with their labels."""
    query = (
        select(
            predictions.c.prediction_id,
            predictions.c.predicted_value,
            predictions.c.probabilities,
            predictions.c.features,
            ground_truth.c.actual_value,
        )
        .select_from(predictions.outerjoin(ground_truth))
        .where(
            predictions.c.model_name == PRODUCTION_MODEL,
            predictions.c.model_version == int(version.version),
            predictions.c.timestamp >= start,
        )
    )
    if input_uri:
        query = query.where(predictions.c.input_uri == input_uri)
    with engine().connect() as conn:
        return pd.read_sql(query, conn)


def run_monitor(input_uri: str | None = None) -> tuple[list[Check], dict]:
    """Check the production model's recent predictions (all, or only those of `input_uri`);
    store the results in Postgres and MLflow."""
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    version = MlflowClient().get_model_version_by_alias(PRODUCTION_MODEL, ALIAS)
    end = datetime.now(UTC)
    start = end - timedelta(hours=s.monitor_window_hours)
    context = {
        "model_version": int(version.version),
        "model_family": version.tags.get("model", "unknown"),
        "feature_set": version.tags.get("feature_set", "unknown"),
        "window_start": start,
        "window_end": end,
        "input_uri": input_uri or "all",
    }
    try:
        profile = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{PROFILE}")
    except MlflowException:
        checks = [skipped("reference_profile", "the production model has no reference profile: retrain it")]
    else:
        window = load_window(version, start, input_uri)
        context["rows"] = len(window)
        checks = evaluate(profile, window)

    _store(checks, context)
    _log(checks, context)
    return checks, context


def _store(checks: list[Check], context: dict) -> None:
    keys = ("model_version", "model_family", "feature_set", "window_start", "window_end")
    rows = [{k: context[k] for k in keys} | {"check_name": c.name} | asdict(c) for c in checks]
    for row in rows:
        row.pop("name")
    with engine().begin() as conn:
        conn.execute(monitoring_checks.insert(), rows)


def _log(checks: list[Check], context: dict) -> None:
    mlflow.set_experiment(EXPERIMENT)
    statuses = [c.status for c in checks]
    tags = {"alerts": str(statuses.count("alert")), "warnings": str(statuses.count("warn"))}
    with mlflow.start_run(run_name=f"monitor-v{context['model_version']}-{context['window_end']:%Y%m%d-%H%M}", tags=tags):
        mlflow.log_params({k: str(v) for k, v in context.items()})
        mlflow.log_metrics({c.name: c.value for c in checks if c.value is not None})
        mlflow.log_table(pd.DataFrame([asdict(c) for c in checks]), artifact_file="checks.json")


def ingest_labels(input_uri: str) -> dict:
    """Load ground-truth labels from an S3 CSV; a new label for an id replaces the old one.

    Either `prediction_id` + `actual_value` (the production format), or an inference file that also carries
    the target column: its labels go to the latest batch predicted from that same file, in row order.
    """
    target = get_settings().target
    labels = read_csv(input_uri)
    if {"prediction_id", "actual_value"} <= set(labels.columns):
        labels = labels[["prediction_id", "actual_value"]]
    elif target in labels:
        ids = _latest_batch_ids(input_uri)
        if len(ids) != len(labels):
            raise ValueError(f"{input_uri} has {len(labels)} rows but its latest batch has {len(ids)} predictions")
        labels = pd.DataFrame({"prediction_id": ids, "actual_value": labels[target]})
    else:
        raise ValueError(f"labels file needs `prediction_id` + `actual_value`, or a `{target}` column")
    labels = labels.dropna().drop_duplicates("prediction_id", keep="last").astype({"prediction_id": int, "actual_value": str})

    with engine().begin() as conn:
        ids = labels["prediction_id"].tolist()
        known = set(conn.execute(select(predictions.c.prediction_id).where(predictions.c.prediction_id.in_(ids))).scalars())
        rows = labels[labels["prediction_id"].isin(known)].to_dict("records")
        if rows:
            statement = insert(ground_truth).values(rows)
            conn.execute(statement.on_conflict_do_update(
                index_elements=["prediction_id"],
                set_={"actual_value": statement.excluded.actual_value, "labeled_at": func.now()},
            ))
    return {"received": len(labels), "stored": len(rows), "unknown_ids": len(labels) - len(rows)}


def _latest_batch_ids(input_uri: str) -> list[int]:
    latest = (
        select(predictions.c.batch_id)
        .where(predictions.c.input_uri == input_uri)
        .order_by(predictions.c.prediction_id.desc())
        .limit(1)
        .scalar_subquery()
    )
    query = select(predictions.c.prediction_id).where(predictions.c.batch_id == latest).order_by(predictions.c.prediction_id)
    with engine().connect() as conn:
        return list(conn.execute(query).scalars())
