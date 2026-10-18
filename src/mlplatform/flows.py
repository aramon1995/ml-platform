"""Prefect flows: a thin layer over the plain-Python core (tuning, selection, inference)."""

from prefect import flow, serve, task
from prefect.schedules import Cron

from mlplatform import inference, selection, tuning
from mlplatform.models import MODELS


@task(task_run_name="tune-{name}")
def tune(name: str, n_trials: int | None) -> str:
    return tuning.run_study(name, n_trials)


@flow(name="train")
def train(models: list[str] | None = None, n_trials: int | None = None) -> dict[str, str]:
    """One Optuna study per model, sequentially, on the shared split and folds."""
    return {name: tune(name, n_trials) for name in models or list(MODELS)}


@flow(name="select")
def select() -> str:
    """Per-model champions and the global production model (1-SE rule + complexity)."""
    table = selection.select()
    return table.loc[table["selected"], "model"].item()


@flow(name="train-and-select")
def train_and_select(models: list[str] | None = None, n_trials: int | None = None) -> str:
    train(models, n_trials)
    return select()


@flow(name="batch-inference")
def batch_inference(input_uri: str | None = None) -> str:
    """Predict an S3 batch with classifier@production and store it in Postgres."""
    return inference.batch_inference(input_uri)


if __name__ == "__main__":
    serve(
        train.to_deployment(name="train"),
        select.to_deployment(name="select"),
        train_and_select.to_deployment(name="train-and-select", schedule=Cron("0 3 * * *", active=False)),
        batch_inference.to_deployment(name="batch-inference"),
    )
