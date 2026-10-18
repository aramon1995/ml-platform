import numpy as np
import pandas as pd
import pytest

from mlplatform.features import FEATURE_SETS, FLAT


@pytest.fixture
def raw():
    return pd.DataFrame({
        "Timestamp": ["2025-01-03 10:00:00", "2025-04-01 00:00:00", None],
        "F22": [10.0, np.nan, 30.0],
        "F17": [np.nan, 5.0, np.nan],
        **{c: [0.0, 3.0, 0.0] for c in FLAT},
        "F1": ["noise", "noise", "noise"],
    })


@pytest.mark.parametrize("name", list(FEATURE_SETS))
def test_every_feature_set_transforms_its_columns_without_nans(name, raw):
    features = FEATURE_SETS[name]
    out = features.build().fit_transform(raw[features.columns])
    assert len(out) == len(raw) and not np.isnan(out).any()


def test_unseen_month_encodes_as_all_zeros(raw):
    features = FEATURE_SETS["eda_minimal"]
    prep = features.build().fit(raw[features.columns])
    out = prep.transform(pd.DataFrame({"Timestamp": ["2025-12-24"], "F22": [1.0], "F17": [np.nan]}))
    assert out[0, :3].tolist() == [0.0, 0.0, 0.0]


def test_feature_set_columns_and_description_come_from_its_function():
    features = FEATURE_SETS["eda_minimal"]
    assert features.columns == ["Timestamp", "F22", "F17"]
    assert features.description.startswith("EDA decisions")
