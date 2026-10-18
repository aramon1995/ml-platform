import pandas as pd

from mlplatform.selection import choose


def studies(**rows):
    return pd.DataFrame(
        [{"model": m, "complexity": c, "cv_mean": cv, "cv_se": se, "test": t} for m, (c, cv, se, t) in rows.items()]
    )


def test_simplest_model_within_one_se_wins():
    winner, table = choose(studies(simple=(1, 0.95, 0.01, 0.95), complex=(4, 0.955, 0.01, 0.96)), test_tolerance=0.05)
    assert winner["model"] == "simple"
    assert table["within_1se"].all()


def test_model_outside_one_se_is_excluded():
    winner, _ = choose(studies(simple=(1, 0.90, 0.01, 0.95), best=(3, 0.98, 0.005, 0.96)), test_tolerance=0.05)
    assert winner["model"] == "best"


def test_test_check_discards_a_candidate_that_collapses_on_test():
    winner, table = choose(studies(simple=(1, 0.975, 0.01, 0.80), best=(3, 0.98, 0.01, 0.97)), test_tolerance=0.05)
    assert winner["model"] == "best"
    assert not table.set_index("model").loc["simple", "test_ok"]
