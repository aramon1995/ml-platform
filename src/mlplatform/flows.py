"""Prefect flows: a thin layer over the plain-Python core (tuning, selection, inference, monitoring)."""

import urllib.request
from dataclasses import asdict

from prefect import flow, get_run_logger, runtime, serve, task
from prefect.artifacts import create_table_artifact
from prefect.automations import Automation, DoNothing, EventTrigger
from prefect.client.orchestration import get_client
from prefect.events import emit_event
from prefect.schedules import Cron

from mlplatform import inference, monitoring, selection, tuning
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


ALERT_EVENT = "mlplatform.monitoring.alert"
ALERT_AUTOMATION = "mlplatform-monitoring-alerts"


@flow(name="monitor")
def monitor(input_uri: str | None = None) -> dict[str, int]:
    """Drift, data-quality and performance checks on the production model's recent predictions.

    Alerts become Prefect events (`mlplatform.monitoring.alert`) and the whole result a table artifact.
    The run stays Completed: detecting drift is not a failure of the process.
    """
    checks, context = monitoring.run_monitor(input_uri)
    window = f"{context['window_start']:%Y-%m-%d %H:%M} – {context['window_end']:%Y-%m-%d %H:%M} UTC"
    create_table_artifact(
        [asdict(c) for c in checks],
        key="monitoring",
        description=(
            f"{context['model_family']} v{context['model_version']} ({context['feature_set']}) · {window}"
            f" · input: {context['input_uri']}"
        ),
    )
    alerts = [c for c in checks if c.status == "alert"]
    for c in alerts:
        emit_event(
            event=ALERT_EVENT,
            resource={"prefect.resource.id": f"mlplatform.check.{c.name}", "prefect.resource.name": c.name},
            payload=asdict(c) | {"model_version": context["model_version"], "window": window},
        )
        get_run_logger().warning(f"ALERT {c.name} = {c.value:.3f} (alert >= {c.alert}): {c.detail}")
    if alerts:
        with get_client(sync_client=True) as client:
            client.update_flow_run(runtime.flow_run.id, tags=[*runtime.flow_run.tags, f"alerts:{len(alerts)}"])
    statuses = [c.status for c in checks]
    return {status: statuses.count(status) for status in ("ok", "warn", "alert", "insufficient_data")}


@flow(name="ingest-labels")
def ingest_labels(input_uri: str) -> dict[str, int]:
    """Load ground-truth labels from an S3 CSV (`prediction_id` + `actual_value`, or an inference file with its target)."""
    return monitoring.ingest_labels(input_uri)


def ensure_alert_automation() -> None:
    """Automation reacting to monitoring alerts. It does nothing until its action is swapped for a
    SendNotification pointing at a notification block (Slack, email, webhook) in the Prefect UI."""
    try:
        Automation.read(name=ALERT_AUTOMATION)
    except ValueError:
        Automation(
            name=ALERT_AUTOMATION,
            description="Monitoring alerts from the monitor flow; replace DoNothing with a notification",
            trigger=EventTrigger(expect={ALERT_EVENT}, posture="Reactive", threshold=1),
            actions=[DoNothing()],
        ).create()


if __name__ == "__main__":
    ensure_alert_automation()
    serve(
        train.to_deployment(name="train"),
        select.to_deployment(name="select"),
        train_and_select.to_deployment(name="train-and-select", schedule=Cron("0 3 * * *", active=False)),
        batch_inference.to_deployment(name="batch-inference"),
        monitor.to_deployment(name="monitor", schedule=Cron("0 6 * * *")),
        ingest_labels.to_deployment(name="ingest-labels"),
    )
