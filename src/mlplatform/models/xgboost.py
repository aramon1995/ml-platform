from optuna import Trial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from mlplatform.config import get_settings
from mlplatform.features import FeatureSet

NAME = "xgboost"
COMPLEXITY = 3


class LabelEncodedXGB(ClassifierMixin, BaseEstimator):
    """XGBoost needs integer labels; this keeps string labels at fit and predict time."""

    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.model_ = XGBClassifier(**self.params).fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def suggest(trial: Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def build_pipeline(params: dict, features: FeatureSet) -> Pipeline:
    model = LabelEncodedXGB(n_jobs=1, random_state=get_settings().seed, **params)
    return Pipeline([("prep", features.preprocessor(scale=False)), ("model", model)])
