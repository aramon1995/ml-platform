import pandas as pd

from mlplatform.data import make_split


def test_split_is_deterministic_stratified_and_disjoint():
    y = pd.Series(["A"] * 80 + ["B"] * 40 + ["D"] * 10)
    first, second = make_split(y), make_split(y)
    assert first == second
    assert not set(first.train) & set(first.test)
    assert (y.loc[first.test] == "D").sum() == 2
    assert len(first.folds) == 10
    for folds in (first.folds, first.rescore_folds):
        assert all(set(val) <= set(first.train) and not set(tr) & set(val) for tr, val in folds)
    assert first.rescore_folds != first.folds


def test_evaluation_key_changes_with_data_or_validation_settings(monkeypatch):
    from mlplatform import data
    from mlplatform.config import Settings

    df = pd.DataFrame({"F22": [1.0, 2.0], "Target": ["A", "B"]})
    base = data.evaluation_key(df)
    assert data.evaluation_key(df) == base
    assert data.evaluation_key(df.assign(F22=[1.0, 3.0])) != base
    monkeypatch.setattr(data, "get_settings", lambda: Settings(seed=7))
    assert data.evaluation_key(df) != base
