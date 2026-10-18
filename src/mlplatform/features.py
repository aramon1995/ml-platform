"""Named feature sets: feature engineering shared by every model.

A feature set is a function decorated with `@feature_set` that returns a ColumnTransformer built with
plain scikit-learn steps (any imputer, encoder, discretizer...). Its docstring is its description and
its input columns are read from the transformer itself. Model-specific steps such as scaling belong
to the model module, not here.

A feature set is immutable once a model has been trained with it (a team rule, not enforced): to try
a different preprocessing, add a new feature set instead of editing an existing one.

A model is trained only on the columns its feature set uses; any other input column is ignored.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

TIMESTAMP = "Timestamp"
FLAT = ["F3", "F6", "F9", "F12", "F18", "F21", "F24", "F27"]  # >90 % zeros (lab/eda.ipynb §3)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    build: Callable[[], ColumnTransformer]

    @property
    def description(self) -> str:
        return (self.build.__doc__ or "").strip()

    @property
    def columns(self) -> list[str]:
        """Input contract: the only raw columns a model trained on this feature set needs."""
        return list(dict.fromkeys(c for _, _, cols in self.build().transformers for c in cols))


FEATURE_SETS: dict[str, FeatureSet] = {}


def feature_set(build: Callable[[], ColumnTransformer]) -> Callable[[], ColumnTransformer]:
    FEATURE_SETS[build.__name__] = FeatureSet(build.__name__, build)
    return build


# Building blocks


def columns(*blocks: tuple[str, object, list[str]]) -> ColumnTransformer:
    """ColumnTransformer with dense output; unlisted columns are dropped."""
    return ColumnTransformer(list(blocks), remainder="drop", sparse_threshold=0, verbose_feature_names_out=False)


def extract_month(X: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(X[TIMESTAMP], errors="coerce").dt.strftime("%m")
    return month.fillna("missing").to_frame("month")


def _month_names(transformer, names):
    return ["month"]


def month_onehot() -> Pipeline:
    """Month of `Timestamp` as a category: one-hot, so no order between months is assumed."""
    return make_pipeline(
        FunctionTransformer(extract_month, feature_names_out=_month_names),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )


# Feature sets


@feature_set
def eda_minimal() -> ColumnTransformer:
    """EDA decisions (lab/eda.ipynb §9): month one-hot, F22, F17 with missing indicator."""
    return columns(
        ("month", month_onehot(), [TIMESTAMP]),
        ("strong", SimpleImputer(strategy="median"), ["F22"]),
        ("sparse", SimpleImputer(strategy="median", add_indicator=True), ["F17"]),
    )


@feature_set
def with_flat_columns() -> ColumnTransformer:
    """eda_minimal plus the eight flat columns (>90 % zeros) the EDA dropped."""
    return columns(
        ("month", month_onehot(), [TIMESTAMP]),
        ("numeric", SimpleImputer(strategy="median"), ["F22", *FLAT]),
        ("sparse", SimpleImputer(strategy="median", add_indicator=True), ["F17"]),
    )


@feature_set
def reduced_set() -> ColumnTransformer:
    """EDA decisions also removing F17, which holds low MI."""
    return columns(
        ("month", month_onehot(), [TIMESTAMP]),
        ("strong", SimpleImputer(strategy="median"), ["F22"]),
    )

DEFAULT_FEATURE_SET = "eda_minimal"
