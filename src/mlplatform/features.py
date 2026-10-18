"""Feature selection and preprocessing, as decided in lab/eda.ipynb (section 9).

Only three inputs survive the EDA: the month extracted from `Timestamp`, `F22` and `F17`.
Every other column is dropped by the ColumnTransformer (remainder="drop").
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TIMESTAMP = "Timestamp"
STRONG = ["F22"]
SPARSE = ["F17"]  # 90 % nulls: the value is informative, so impute and flag missingness


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


def _preprocessor(scale: bool) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("month", _month(), [TIMESTAMP]),
            ("strong", _numeric(scale), STRONG),
            ("sparse", _numeric(scale, indicator=True), SPARSE),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def linear_preprocessor() -> ColumnTransformer:
    """For distance/gradient-based models (logistic regression, SVM, MLP): scaled numerics."""
    return _preprocessor(scale=True)


def tree_preprocessor() -> ColumnTransformer:
    """For tree ensembles: no scaling needed."""
    return _preprocessor(scale=False)
