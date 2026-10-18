import argparse
import time
from datetime import datetime

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, get_scorer
from sklearn.model_selection import cross_val_score

from mlplatform import reporting
from mlplatform.config import get_settings
from mlplatform.data import digest, features_target, load_dataset, make_split
from mlplatform.models import MODELS

EXPERIMENT = "tuning"
REPORT_METRICS = ("accuracy", "balanced_accuracy")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_study(name: str, n_trials: int | None = None) -> str:
    """Tune one model with Optuna on the shared folds, re-score the top trials on fresh folds,
    refit the winner on train and evaluate it once on test.

    MLflow layout: parent run `<name>` → nested `<name>/trial-NNN` runs and a `<name>/final` run with the model.
    Returns the parent run id.
    """
    s = get_settings()
    model = MODELS[name]
    df = load_dataset()
    X, y = features_target(df)
    split = make_split(y)
    X_train, y_train, X_test, y_test = X.loc[split.train], y.loc[split.train], X.loc[split.test], y.loc[split.test]

    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    study = optuna.create_study(
        study_name=f"{name}-{datetime.now():%Y%m%d-%H%M%S}",
        storage=s.optuna_storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=s.seed),
    )

    def objective(trial: optuna.Trial) -> float:
        params = model.suggest(trial)
        scores = cross_val_score(model.build_pipeline(params), X, y, cv=split.folds, scoring=s.metric)
        trial.set_user_attr("params", params)
        trial.set_user_attr("cv_scores", scores.tolist())
        with mlflow.start_run(run_name=f"{name}/trial-{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({"cv_mean": scores.mean(), "cv_std": scores.std()})
            for fold, score in enumerate(scores):
                mlflow.log_metric("cv_fold", score, step=fold)
        return scores.mean()

    tags = {"model": name, "data_digest": digest(df), "optuna_study": study.study_name}
    with mlflow.start_run(run_name=name, tags=tags) as parent:
        mlflow.log_input(mlflow.data.from_pandas(df, source=s.data_uri, targets=s.target, name="raw"), context="training")
        mlflow.log_params(s.model_dump(include={"seed", "test_size", "n_splits", "n_repeats", "metric"}))
        mlflow.log_params({"complexity": model.COMPLEXITY, "n_trials": n_trials or s.n_trials})
        mlflow.log_dict(
            {"train": split.train, "test": split.test, "folds": split.folds, "rescore_folds": split.rescore_folds},
            "split.json",
        )

        study.optimize(objective, n_trials=n_trials or s.n_trials)
        ranking = rescore(study, model, X, y, split.rescore_folds, s.metric, s.rescore_top_k)
        mlflow.log_table(ranking, artifact_file="rescore.json")
        chosen = ranking.iloc[0]
        best = study.trials[int(chosen["trial"])]
        cv = {
            "cv_mean": chosen["rescore_mean"],
            "cv_std": chosen["rescore_std"],
            "cv_se": chosen["rescore_se"],
            "search_cv_mean": chosen["search_mean"],
            "optimism": study.best_value - ranking.loc[ranking["trial"] == study.best_trial.number, "rescore_mean"].item(),
        }

        with mlflow.start_run(run_name=f"{name}/final", nested=True):
            pipeline = model.build_pipeline(best.user_attrs["params"])
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            test = {f"test_{m}": get_scorer(m)(pipeline, X_test, y_test) for m in (s.metric, *REPORT_METRICS)}
            mlflow.log_params(best.user_attrs["params"])
            mlflow.log_metrics(cv | test | {"fit_seconds": fit_seconds})
            _log_report(pipeline, X_test, y_test)
            info = mlflow.sklearn.log_model(
                pipeline,
                name=name,
                serialization_format="cloudpickle",
                signature=infer_signature(X_test, pipeline.predict(X_test)),
            )

        mlflow.set_tag("model_uri", info.model_uri)
        mlflow.log_params({f"best_{k}": v for k, v in best.user_attrs["params"].items()})
        mlflow.log_metrics(cv | test | {"fit_seconds": fit_seconds, "best_trial": best.number})
        for fig_name, fig in reporting.optuna_figures(study).items():
            mlflow.log_figure(fig, f"optuna/{fig_name}.png")
    return parent.info.run_id


def rescore(study: optuna.Study, model, X, y, folds, metric: str, top_k: int) -> pd.DataFrame:
    """Re-measure the top trials on folds that played no part in choosing them; best re-score first."""
    complete = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    rows = []
    for trial in sorted(complete, key=lambda t: t.value, reverse=True)[:top_k]:
        scores = cross_val_score(model.build_pipeline(trial.user_attrs["params"]), X, y, cv=folds, scoring=metric)
        rows.append({
            "trial": trial.number,
            "search_mean": trial.value,
            "rescore_mean": scores.mean(),
            "rescore_std": scores.std(),
            "rescore_se": scores.std() / np.sqrt(len(scores)),
        })
    return pd.DataFrame(rows).sort_values("rescore_mean", ascending=False, ignore_index=True)


def _log_report(pipeline, X, y) -> None:
    mlflow.log_text(classification_report(y, pipeline.predict(X), zero_division=0), "classification_report.txt")
    mlflow.log_figure(reporting.confusion_matrix(pipeline, X, y), "confusion_matrix.png")
    importances = reporting.permutation_importances(pipeline, X, y)
    mlflow.log_dict(importances.round(4).to_dict(), "permutation_importance.json")
    mlflow.log_figure(reporting.bar(importances[importances.abs() > 0], "Permutation importance (test)"), "permutation_importance.png")
    if history := getattr(pipeline[-1], "history_", None):
        mlflow.log_figure(reporting.training_curve(history), "training_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune models with Optuna and log them to MLflow")
    parser.add_argument("models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    for name in args.models:
        print(name, run_study(name, args.trials))


if __name__ == "__main__":
    main()
