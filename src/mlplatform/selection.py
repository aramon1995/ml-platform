"""Champion selection.

1. Per model: the latest tuning study is that model's champion (registered model `<model>`, alias `champion`).
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
from mlplatform.models import MODELS

EXPERIMENT = "selection"
PRODUCTION_MODEL = "classifier"


def latest_studies() -> pd.DataFrame:
    metric = get_settings().metric
    runs = mlflow.search_runs(
        experiment_names=["tuning"], filter_string="tags.model_uri LIKE 'models:/%'", order_by=["start_time DESC"]
    ).drop_duplicates("tags.model")
    runs = runs[runs["tags.model"].isin(MODELS)]
    return pd.DataFrame({
        "model": runs["tags.model"],
        "run_id": runs["run_id"],
        "model_uri": runs["tags.model_uri"],
        "complexity": runs["tags.model"].map(lambda m: MODELS[m].COMPLEXITY),
        "cv_mean": runs["metrics.cv_mean"],
        "cv_se": runs["metrics.cv_se"],
        "test": runs[f"metrics.test_{metric}"],
    }).reset_index(drop=True)


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
    studies = latest_studies()
    for row in studies.itertuples():
        register(row.model, row.model_uri, "champion", {"cv_mean": row.cv_mean, "test": row.test, "run_id": row.run_id})

    winner, table = choose(studies, s.test_tolerance)
    version = register(
        PRODUCTION_MODEL,
        winner["model_uri"],
        "production",
        {"model": winner["model"], "cv_mean": winner["cv_mean"], "test": winner["test"], "run_id": winner["run_id"]},
    )

    mlflow.set_experiment(EXPERIMENT)
    best = table.loc[table["cv_mean"].idxmax()]
    with mlflow.start_run(run_name=f"production-{winner['model']}", tags={"winner": winner["model"]}):
        mlflow.log_params({
            "rule": "1-SE + complexity",
            "metric": s.metric,
            "best_cv_model": best["model"],
            "threshold": round(best["cv_mean"] - best["cv_se"], 4),
            "test_tolerance": s.test_tolerance,
            "winner": winner["model"],
            "production_version": version,
        })
        mlflow.log_table(table.drop(columns=["model_uri"]), artifact_file="candidates.json")
    return table


if __name__ == "__main__":
    columns = ["model", "complexity", "cv_mean", "cv_se", "test", "within_1se", "test_ok", "selected"]
    print(select()[columns].round(4).to_string(index=False))
