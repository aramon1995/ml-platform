"""Champion selection.

Only studies with the current evaluation key (same data, split and metric) compete; a change in any of
them means retraining. Within those, feature sets compete like any other part of the pipeline.

1. Per model: its study with the best CV mean, whatever the feature set (registered model `<model>`,
   alias `champion`).
2. Global, one-standard-error rule: every champion whose CV mean is within one SE of the best is
   considered equally good; the simplest one (lowest COMPLEXITY) wins, ties broken by CV mean.
3. Test is a sanity check, not a selector: a candidate more than `test_tolerance` below the best
   test score is discarded.
The winner is registered as `classifier`, alias `production`.
"""

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlplatform.config import get_settings
from mlplatform.data import evaluation_key, load_dataset
from mlplatform.models import MODELS

EXPERIMENT = "selection"
PRODUCTION_MODEL = "classifier"


def comparable_studies(key: str) -> pd.DataFrame:
    metric = get_settings().metric
    runs = mlflow.search_runs(
        experiment_names=["tuning"],
        filter_string=f"tags.evaluation_key = '{key}' and tags.model_uri LIKE 'models:/%'",
    )
    if runs.empty:
        raise RuntimeError(f"No tuning studies for evaluation key {key}: retrain the models first")
    runs = runs[runs["tags.model"].isin(MODELS)]
    return pd.DataFrame({
        "model": runs["tags.model"],
        "feature_set": runs["tags.feature_set"],
        "run_id": runs["run_id"],
        "model_uri": runs["tags.model_uri"],
        "complexity": runs["tags.model"].map(lambda m: MODELS[m].COMPLEXITY),
        "cv_mean": runs["metrics.cv_mean"],
        "cv_se": runs["metrics.cv_se"],
        "test": runs[f"metrics.test_{metric}"],
    }).reset_index(drop=True)


def champions(studies: pd.DataFrame) -> pd.DataFrame:
    """Best study (by CV mean) of each model."""
    return studies.loc[studies.groupby("model")["cv_mean"].idxmax()].reset_index(drop=True)


def choose(studies: pd.DataFrame, test_tolerance: float) -> tuple[pd.Series, pd.DataFrame]:
    """Apply the 1-SE rule plus the test check; returns the winner and the annotated table."""
    table = studies.copy()
    best = table.loc[table["cv_mean"].idxmax()]
    table["within_1se"] = table["cv_mean"] >= best["cv_mean"] - best["cv_se"]
    table["test_ok"] = table["test"] >= table["test"].max() - test_tolerance
    eligible = table[table["within_1se"] & table["test_ok"]].sort_values(
        ["complexity", "cv_mean"], ascending=[True, False]
    )
    winner = eligible.iloc[0] if len(eligible) else best
    table["selected"] = table["model"] == winner["model"]
    return winner, table


def register(name: str, model_uri: str, alias: str, tags: dict) -> str:
    """Register `model_uri` under `name@alias`; no-op when the alias already points to it."""
    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(name, alias)
        if current.source == model_uri:
            return current.version
    except MlflowException:
        pass
    version = mlflow.register_model(model_uri, name, tags={k: str(v) for k, v in tags.items()})
    client.set_registered_model_alias(name, alias, version.version)
    return version.version


def select() -> pd.DataFrame:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    key = evaluation_key(load_dataset())
    studies = comparable_studies(key)
    best_per_model = champions(studies)
    tag_columns = ["model", "feature_set", "cv_mean", "test", "run_id"]
    for row in best_per_model.to_dict("records"):
        register(row["model"], row["model_uri"], "champion", {k: row[k] for k in tag_columns})

    winner, table = choose(best_per_model, s.test_tolerance)
    version = register(PRODUCTION_MODEL, winner["model_uri"], "production", winner[tag_columns].to_dict())

    mlflow.set_experiment(EXPERIMENT)
    best = table.loc[table["cv_mean"].idxmax()]
    with mlflow.start_run(run_name=f"production-{winner['model']}", tags={"winner": winner["model"], "evaluation_key": key}):
        mlflow.log_params({
            "rule": "1-SE + complexity",
            "metric": s.metric,
            "best_cv_model": best["model"],
            "threshold": round(best["cv_mean"] - best["cv_se"], 4),
            "test_tolerance": s.test_tolerance,
            "winner": winner["model"],
            "winner_feature_set": winner["feature_set"],
            "missing_models": ",".join(sorted(set(MODELS) - set(studies["model"]))) or "none",
            "production_version": version,
        })
        mlflow.log_table(table.drop(columns=["model_uri"]), artifact_file="candidates.json")
        mlflow.log_table(studies.drop(columns=["model_uri"]), artifact_file="all_studies.json")
    return table


if __name__ == "__main__":
    columns = ["model", "feature_set", "complexity", "cv_mean", "cv_se", "test", "within_1se", "test_ok", "selected"]
    print(select()[columns].round(4).to_string(index=False))
