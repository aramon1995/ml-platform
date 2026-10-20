from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # data and validation
    data_uri: str = "s3://data/raw/data.csv"
    target: str = "Target"
    seed: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    n_repeats: int = 2
    metric: str = "f1_macro"
    n_trials: int = 30
    rescore_top_k: int = 5
    test_tolerance: float = 0.05

    # monitoring: every check reports how much worse the window is than the reference
    monitor_window_hours: int = 24
    monitor_min_rows: int = 50
    monitor_min_labels: int = 30
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    rare_month_freq: float = 0.01
    rare_month_warn: float = 0.05
    rare_month_alert: float = 0.15
    null_warn: float = 0.05
    null_alert: float = 0.15
    confidence_warn: float = 0.05
    confidence_alert: float = 0.10
    performance_warn: float = 0.03
    performance_alert: float = 0.05

    # infrastructure
    aws_endpoint_url: str = "http://localhost:9100"
    aws_access_key_id: str = "s3admin"
    aws_secret_access_key: str = "s3admin12345"
    mlflow_tracking_uri: str = "http://localhost:5000"
    database_url: str = "postgresql+psycopg2://mlp:mlp@localhost:5432/app"
    optuna_storage: str = "postgresql+psycopg2://mlp:mlp@localhost:5432/optuna"
    api_url: str = "http://localhost:8000"

    @property
    def storage_options(self) -> dict:
        return {
            "key": self.aws_access_key_id,
            "secret": self.aws_secret_access_key,
            "client_kwargs": {"endpoint_url": self.aws_endpoint_url},
        }


@cache
def get_settings() -> Settings:
    return Settings()
