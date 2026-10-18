import numpy as np
import optuna
import pandas as pd

from mlplatform.data import make_split
from mlplatform.models import logreg
from mlplatform.tuning import rescore


def test_rescore_ranks_top_trials_on_fresh_folds():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.choice(["A", "B"], 100))
    X = pd.DataFrame({
        "Timestamp": np.where(y == "A", "2025-01-10", "2025-04-10"),
        "F22": np.where(y == "A", 10.0, 20.0) + rng.normal(0, 5, 100),
        "F17": np.nan,
    })
    split = make_split(y)
    study = optuna.create_study(direction="maximize")
    for value, c in [(0.9, 1.0), (0.8, 0.01), (0.7, 10.0)]:
        study.add_trial(optuna.trial.create_trial(
            params={"C": c},
            distributions={"C": optuna.distributions.FloatDistribution(1e-3, 1e2, log=True)},
            value=value,
            user_attrs={"params": {"C": c}},
        ))
    ranking = rescore(study, logreg, X, y, split.rescore_folds, "f1_macro", top_k=2)
    assert set(ranking["trial"]) == {0, 1}
    assert ranking["rescore_mean"].is_monotonic_decreasing
    assert ranking.set_index("trial").loc[0, "search_mean"] == 0.9
