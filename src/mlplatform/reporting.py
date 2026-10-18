"""Figures and tables logged to MLflow; everything stays in memory."""

import matplotlib
from sklearn.utils import Bunch
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from optuna import Study
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

from mlplatform.config import get_settings


def confusion_matrix(model, X, y) -> plt.Figure:
    return ConfusionMatrixDisplay.from_estimator(model, X, y, cmap="Blues").figure_


def permutation_importances(model: Pipeline, X:pd.DataFrame, y) -> pd.Series:
    """Model-agnostic: drop in score when each raw input column is shuffled."""
    s = get_settings()
    result: Bunch = permutation_importance(model, X, y, scoring=s.metric, n_repeats=10, random_state=s.seed)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def bar(series: pd.Series, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    series.plot.bar(ax=ax, title=title)
    fig.tight_layout()
    return fig


def training_curve(history: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    pd.DataFrame(history).set_index("epoch").plot(ax=ax, title="Loss por época")
    fig.tight_layout()
    return fig


def optuna_figures(study: Study) -> dict[str, plt.Figure]:
    figures = {"optimization_history": plot_optimization_history(study).figure}
    if len(study.best_trial.params) > 1:
        figures["param_importances"] = plot_param_importances(study).figure
    return figures
