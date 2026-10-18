import copy

import numpy as np
import torch
from optuna import Trial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from mlplatform.config import get_settings
from mlplatform.features import linear_preprocessor

NAME = "mlp"
COMPLEXITY = 4


class TorchMLP(ClassifierMixin, BaseEstimator):
    """Minimal PyTorch MLP with early stopping on a stratified validation split.

    `history_` keeps train/validation loss per epoch for the training curve.
    """

    def __init__(
        self,
        hidden=(64,),
        activation="ReLU",
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=200,
        batch_size=32,
        patience=15,
        val_fraction=0.15,
        seed=42,
    ):
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed

    def _network(self, n_in: int, n_out: int) -> nn.Sequential:
        layers, width = [], n_in
        for units in self.hidden:
            layers += [nn.Linear(width, units), getattr(nn, self.activation)(), nn.Dropout(self.dropout)]
            width = units
        return nn.Sequential(*layers, nn.Linear(width, n_out))

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        X = np.asarray(X, dtype=np.float32)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_idx, test_size=self.val_fraction, stratify=y_idx, random_state=self.seed
        )
        X_tr, X_val, y_tr, y_val = map(torch.from_numpy, (X_tr, X_val, y_tr, y_val))

        self.net_ = self._network(X.shape[1], len(self.classes_))
        weights = compute_class_weight("balanced", classes=np.arange(len(self.classes_)), y=y_idx)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.history_, best, best_state, stale = [], float("inf"), None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            for batch in torch.randperm(len(X_tr)).split(self.batch_size):
                optimizer.zero_grad()
                loss = loss_fn(self.net_(X_tr[batch]), y_tr[batch])
                loss.backward()
                optimizer.step()
            self.net_.eval()
            with torch.no_grad():
                train_loss = loss_fn(self.net_(X_tr), y_tr).item()
                val_loss = loss_fn(self.net_(X_val), y_val).item()
            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best:
                best, best_state, stale = val_loss, copy.deepcopy(self.net_.state_dict()), 0
            elif (stale := stale + 1) >= self.patience:
                break
        self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def suggest(trial: Trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    return {
        "hidden": [trial.suggest_categorical(f"units_{i}", [16, 32, 64, 128]) for i in range(n_layers)],
        "activation": trial.suggest_categorical("activation", ["ReLU", "GELU", "Tanh"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


def build_pipeline(params: dict) -> Pipeline:
    model = TorchMLP(seed=get_settings().seed, **params)
    return Pipeline([("prep", linear_preprocessor()), ("model", model)])
