from optuna import Trial
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlplatform.config import get_settings
from mlplatform.features import linear_preprocessor

NAME = "logreg"
COMPLEXITY = 1


def suggest(trial: Trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


def build_pipeline(params: dict) -> Pipeline:
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=get_settings().seed, **params)
    return Pipeline([("prep", linear_preprocessor()), ("model", model)])
