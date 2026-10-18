from optuna import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from mlplatform.config import get_settings
from mlplatform.features import FeatureSet

NAME = "random_forest"
COMPLEXITY = 3


def suggest(trial: Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 10]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def build_pipeline(params: dict, features: FeatureSet) -> Pipeline:
    model = RandomForestClassifier(class_weight="balanced", random_state=get_settings().seed, **params)
    return Pipeline([("features", features.build()), ("model", model)])
