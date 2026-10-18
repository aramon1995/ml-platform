"""Named feature sets and their preprocessing.

A feature set is immutable once a model has been trained with it (a team rule, not enforced): to try
a different preprocessing, add a new feature set instead of editing an existing one.

A model is trained only on the columns its feature set uses; any other input column is ignored.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TIMESTAMP = "Timestamp"
FLAT = ("F3", "F6", "F9", "F12", "F18", "F21", "F24", "F27")  # >90 % zeros (lab/eda.ipynb §3)


def extract_month(X: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(X[TIMESTAMP], errors="coerce").dt.strftime("%m")
    return month.fillna("missing").to_frame("month")


def _month_names(transformer, names):
    return ["month"]


def _month() -> Pipeline:
    """Month as a category: one-hot, so no order between months is assumed."""
    return make_pipeline(
        FunctionTransformer(extract_month, feature_names_out=_month_names),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )


def _numeric(scale: bool, indicator: bool = False) -> Pipeline:
    steps = [SimpleImputer(strategy="median", add_indicator=indicator)]
    return make_pipeline(*steps, StandardScaler()) if scale else make_pipeline(*steps)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    numeric: tuple[str, ...]
    sparse: tuple[str, ...] = ()  # mostly null: median imputation plus a missing indicator

    @property
    def columns(self) -> list[str]:
        """Input contract: the only raw columns a model trained on this feature set needs."""
        return [TIMESTAMP, *self.numeric, *self.sparse]

    def preprocessor(self, scale: bool) -> ColumnTransformer:
        """`scale=True` for distance/gradient-based models (logreg, SVM, MLP); trees don't need it."""
        blocks = [("month", _month(), [TIMESTAMP]), ("numeric", _numeric(scale), list(self.numeric))]
        if self.sparse:
            blocks.append(("sparse", _numeric(scale, indicator=True), list(self.sparse)))
        return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)


FEATURE_SETS = {
    fs.name: fs
    for fs in (
        FeatureSet(
            "eda_minimal",
            "EDA decisions (lab/eda.ipynb §9): month one-hot, F22, F17 with missing indicator",
            numeric=("F22",),
            sparse=("F17",),
        ),
        FeatureSet(
            "with_flat_columns",
            "eda_minimal plus the eight flat columns (>90 % zeros) the EDA dropped",
            numeric=("F22", *FLAT),
            sparse=("F17",),
        ),
        FeatureSet(
            "reduced_set",
            "EDA decisions also removing F17 that holds low MI",
            numeric=("F22",),
        ),
    )
}
DEFAULT_FEATURE_SET = "eda_minimal"
