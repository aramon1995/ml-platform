import numpy as np
import pandas as pd

from mlplatform.features import linear_preprocessor, tree_preprocessor


def frame(timestamps, f22, f17):
    return pd.DataFrame({"Timestamp": timestamps, "F22": f22, "F17": f17, "F1": "noise"})


def test_preprocessors_keep_only_selected_features():
    X = frame(["2025-01-03 10:00:00", "2025-04-01 00:00:00", None], [10.0, np.nan, 30.0], [np.nan, 5.0, np.nan])
    for prep in (linear_preprocessor(), tree_preprocessor()):
        out = prep.fit_transform(X)
        assert list(prep.get_feature_names_out()) == [
            "month_01", "month_04", "month_missing", "F22", "F17", "missingindicator_F17"
        ]
        assert not np.isnan(out).any()


def test_unseen_month_encodes_as_all_zeros():
    prep = tree_preprocessor().fit(frame(["2025-01-03", "2025-04-01"], [1.0, 2.0], [1.0, np.nan]))
    out = prep.transform(frame(["2025-12-24"], [1.0], [np.nan]))
    assert out[0, :2].tolist() == [0.0, 0.0]
    assert out[0, -1] == 1.0
