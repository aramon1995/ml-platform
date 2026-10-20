"""Monitoring scenarios built from the test split (rows the model never trained on).

Each CSV is an inference batch that also carries `Target`, so `make labels` can attach its ground truth.
Run with `uv run python data-monitoring/generate.py`; `make scenario NAME=<file stem>` plays one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mlplatform.data import make_split

HERE = Path(__file__).parent
rng = np.random.default_rng(42)

data = pd.read_csv(HERE.parent / "data.csv").dropna(subset=["Target"]).reset_index(drop=True)
test = data.loc[make_split(data["Target"]).test].reset_index(drop=True)


def with_month(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    days = pd.to_datetime(df["Timestamp"], errors="coerce").dt.day.fillna(15).astype(int)
    return df.assign(Timestamp=[f"2025-{m}-{d:02d} 10:00:00" for m, d in zip(rng.choice(months, len(df)), days)])


def with_nulls(df: pd.DataFrame, columns: list[str], share: float) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df.loc[rng.random(len(df)) < share, col] = np.nan
    return df


scenarios = {
    "01_normal": test,
    "02_numeric_drift": test.assign(F22=test["F22"] + 15),
    "03_rare_months": with_month(test, ["02", "12"]),
    "04_missing_values": with_nulls(test, ["Timestamp", "F22"], 0.4),
    "05_class_shift": test[test["Target"] == "C"].sample(len(test), replace=True, random_state=42),
    "06_label_errors": test.assign(Target=test["Target"].map({"A": "B", "B": "C", "C": "D", "D": "A"})),
}

for name, df in scenarios.items():
    df.to_csv(HERE / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} rows")
