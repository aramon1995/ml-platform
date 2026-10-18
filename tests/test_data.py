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
