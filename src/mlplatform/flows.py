"""Prefect flows: a thin layer over the plain-Python core (tuning, selection, inference)."""

import urllib.request

from prefect import flow, get_run_logger, serve, task
from prefect.schedules import Cron

from mlplatform import inference, selection, tuning
from mlplatform.config import get_settings
from mlplatform.features import DEFAULT_FEATURE_SET
from mlplatform.models import MODELS


@task(task_run_name="tune-{name}-{feature_set}")
def tune(name: str, feature_set: str, n_trials: int | None) -> str:
    return tuning.run_study(name, feature_set, n_trials)


@flow(name="train")
def train(
    models: list[str] | None = None, feature_set: str = DEFAULT_FEATURE_SET, n_trials: int | None = None
) -> dict[str, str]:
    """One Optuna study per model on one feature set, sequentially, on the shared split and folds."""
    return {name: tune(name, feature_set, n_trials) for name in models or list(MODELS)}


@task
def reload_api() -> None:
    """Point the online API at the current production model. The API is optional: failures only warn."""
    request = urllib.request.Request(f"{get_settings().api_url}/reload", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120):
            pass
    except OSError as e:
        get_run_logger().warning(f"API reload skipped: {e}")


@flow(name="select")
def select() -> str:
    """Per-model champions and the global production model (1-SE rule + complexity); then reload the API."""
    table = selection.select()
    reload_api()
    return table.loc[table["selected"], "model"].item()


@flow(name="train-and-select")
def train_and_select(
    models: list[str] | None = None, feature_set: str = DEFAULT_FEATURE_SET, n_trials: int | None = None
) -> str:
    train(models, feature_set, n_trials)
    return select()


@flow(name="batch-inference")
def batch_inference(input_uri: str | None = None, rows: list[dict] | None = None) -> str:
    """Predict an S3 batch (`input_uri`) or inline rows with classifier@production and store them in Postgres."""
    return inference.batch_inference(input_uri, rows)


if __name__ == "__main__":
    serve(
        train.to_deployment(name="train"),
        select.to_deployment(name="select"),
        train_and_select.to_deployment(name="train-and-select", schedule=Cron("0 3 * * *", active=False)),
        batch_inference.to_deployment(name="batch-inference"),
    )
