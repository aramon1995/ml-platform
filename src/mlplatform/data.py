import hashlib
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from mlplatform.config import get_settings


@dataclass(frozen=True)
class Split:
    train: list[int]
    test: list[int]
    folds: list[tuple[list[int], list[int]]]
    rescore_folds: list[tuple[list[int], list[int]]]


def read_csv(uri: str) -> pd.DataFrame:
    return pd.read_csv(uri, storage_options=get_settings().storage_options)


def load_dataset(uri: str | None = None) -> pd.DataFrame:
    s = get_settings()
    return read_csv(uri or s.data_uri).dropna(subset=[s.target]).reset_index(drop=True)


def digest(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = get_settings().target
    return df.drop(columns=[target]), df[target]


def _folds(y: pd.Series, train, seed: int) -> list[tuple[list[int], list[int]]]:
    s = get_settings()
    cv = RepeatedStratifiedKFold(n_splits=s.n_splits, n_repeats=s.n_repeats, random_state=seed)
    return [(train[a].tolist(), train[b].tolist()) for a, b in cv.split(train, y.loc[train])]


def make_split(y: pd.Series) -> Split:
    """Deterministic for a given dataset and settings: every model sees the same train/test and folds.

    `folds` drive the hyperparameter search; `rescore_folds` are a different partition of the same
    train rows, used to re-measure the best trials without the luck of the folds they were chosen on.
    """
    s = get_settings()
    idx = y.index.to_numpy()
    train, test = train_test_split(idx, test_size=s.test_size, stratify=y, random_state=s.seed)
    return Split(train.tolist(), test.tolist(), _folds(y, train, s.seed), _folds(y, train, s.seed + 1))
