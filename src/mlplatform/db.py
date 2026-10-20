"""Database schema of the `app` database: predictions, ground-truth labels and monitoring checks.

`python -m mlplatform.db` creates the missing tables; the `db-init` service runs it on `make start`.
`create_all` does not alter existing tables: a column change needs a migration (or a manual ALTER).
"""

from functools import cache

from sqlalchemy import BigInteger, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB

from mlplatform.config import get_settings

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

ground_truth = Table(
    "ground_truth",
    metadata,
    Column("prediction_id", BigInteger, ForeignKey("predictions.prediction_id"), primary_key=True),
    Column("actual_value", String, nullable=False),
    Column("labeled_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

monitoring_checks = Table(
    "monitoring",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("window_start", DateTime(timezone=True), nullable=False),
    Column("window_end", DateTime(timezone=True), nullable=False),
    Column("model_version", Integer, nullable=False),
    Column("model_family", String, nullable=False),
    Column("feature_set", String, nullable=False),
    Column("check_name", String, nullable=False),
    Column("value", Float),
    Column("warn", Float, nullable=False),
    Column("alert", Float, nullable=False),
    Column("status", String, nullable=False),
    Column("detail", String, nullable=False),
)


@cache
def engine():
    return create_engine(get_settings().database_url)


def init_db() -> list[str]:
    metadata.create_all(engine())
    return sorted(metadata.tables)


if __name__ == "__main__":
    print("tables:", ", ".join(init_db()))
