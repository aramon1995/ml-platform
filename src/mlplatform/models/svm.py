from optuna import Trial
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from mlplatform.config import get_settings
from mlplatform.features import FeatureSet

NAME = "svm"
COMPLEXITY = 2


def suggest(trial: Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-1, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def build_pipeline(params: dict, features: FeatureSet) -> Pipeline:
    model = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=get_settings().seed, **params)
    return Pipeline([("prep", features.preprocessor(scale=True)), ("model", model)])
