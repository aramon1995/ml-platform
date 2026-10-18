"""Batch inference with the production model; predictions go to Postgres, a summary to MLflow."""

import argparse
import json
import uuid

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline
from sqlalchemy import BigInteger, Column, DateTime, Integer, MetaData, String, Table, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB

from mlplatform.config import get_settings
from mlplatform.data import features_target, load_dataset, make_split, read_csv
from mlplatform.selection import PRODUCTION_MODEL

EXPERIMENT = "inference"
ALIAS = "production"

metadata = MetaData()
predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", BigInteger, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("predicted_value", String, nullable=False),
    Column("probabilities", JSONB, nullable=False),
    Column("model_name", String, nullable=False),
    Column("model_version", Integer, nullable=False),
    Column("model_family", String, nullable=False),
    Column("feature_set", String, nullable=False),
    Column("run_id", String, nullable=False),
    Column("batch_id", String, nullable=False),
    Column("input_uri", String, nullable=False),
    Column("features", JSONB, nullable=False),
)


def load_production():
    """Resolve the alias once and load that exact version, so the stored version is the one that predicted."""
    version = MlflowClient().get_model_version_by_alias(PRODUCTION_MODEL, ALIAS)
    return mlflow.sklearn.load_model(f"models:/{PRODUCTION_MODEL}/{version.version}"), version


def load_batch(input_uri: str | None = None, rows: list[dict] | None = None) -> tuple[pd.DataFrame, str]:
    """Raw rows from S3 (`input_uri`) or passed inline (`rows`); without either, the test split is the sample batch."""
    s = get_settings()
    if input_uri and rows:
        raise ValueError("pass either input_uri or rows, not both")
    if rows:
        return pd.DataFrame(rows).drop(columns=[s.target], errors="ignore"), "inline"
    if input_uri:
        return read_csv(input_uri).drop(columns=[s.target], errors="ignore"), input_uri
    X, y = features_target(load_dataset())
    return X.loc[make_split(y).test].reset_index(drop=True), f"{s.data_uri}#test"


def predict(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Uses only the columns the model was trained on; others are ignored, missing ones are an error."""
    required = list(model.feature_names_in_)
    missing = set(required) - set(X.columns)
    if missing:
        raise ValueError(f"batch is missing columns the model needs: {sorted(missing)}")
    X = X[required]
    proba = model.predict_proba(X)
    return pd.DataFrame({
        "predicted_value": model.classes_[proba.argmax(axis=1)],
        "probabilities": [dict(zip(model.classes_, map(float, row))) for row in proba],
        "features": X.astype(object).where(X.notna(), None).to_dict("records"),
    })


def annotate(results: pd.DataFrame, version, batch_id: str, source: str) -> pd.DataFrame:
    """Add the metadata stored with every prediction: which model predicted and where the rows came from."""
    return results.assign(
        model_name=PRODUCTION_MODEL,
        model_version=int(version.version),
        model_family=version.tags.get("model", "unknown"),
        feature_set=version.tags.get("feature_set", "unknown"),
        run_id=version.run_id,
        batch_id=batch_id,
        input_uri=source,
    )


def store(results: pd.DataFrame) -> None:
    engine = create_engine(get_settings().database_url)
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(predictions.insert(), results.to_dict("records"))


def batch_inference(input_uri: str | None = None, rows: list[dict] | None = None) -> str:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    model, version = load_production()
    X, source = load_batch(input_uri, rows)
    batch_id = uuid.uuid4().hex
    results = annotate(predict(model, X), version, batch_id, source)
    store(results)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"batch-{batch_id[:8]}", tags={"batch_id": batch_id}):
        mlflow.log_params({"input_uri": source, "model_version": version.version, "model_family": results["model_family"][0],
                           "feature_set": results["feature_set"][0]})
        counts = results["predicted_value"].value_counts()
        confidence = results["probabilities"].map(lambda p: max(p.values()))
        mlflow.log_metrics({"n_rows": len(results), "mean_confidence": confidence.mean()} | {f"n_{k}": v for k, v in counts.items()})
    return batch_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with classifier@production and store the results in Postgres")
    parser.add_argument("input_uri", nargs="?", help="s3://... CSV; defaults to the test split")
    parser.add_argument("--rows", type=json.loads, help='inline JSON rows, e.g. \'[{"Timestamp": "2025-04-10", "F22": 18.5}]\'')
    args = parser.parse_args()
    print(batch_inference(args.input_uri, args.rows))
