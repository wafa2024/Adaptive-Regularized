from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost is optional during development
    XGBClassifier = None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_sklearn_baselines(x_train, y_train, x_test, y_test, random_seed: int = 42) -> dict[str, dict[str, float]]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=random_seed),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="binary:logistic" if len(np.unique(y_train)) == 2 else "multi:softprob",
            eval_metric="logloss",
            random_state=random_seed,
        )

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = _metrics(y_test, predictions)
    return results


class LSTMBaseline(torch.nn.Module):
    """A compact LSTM baseline for ordered flow sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat each flow as a one-step sequence when no longer sequence window is provided.
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def train_lstm_baseline(x_train, y_train, x_test, y_test, hidden_dim: int = 64, epochs: int = 50) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)

    model = LSTMBaseline(x_train.shape[1], hidden_dim, int(np.max(y_train)) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x_test_t).argmax(dim=1).cpu().numpy()
    return _metrics(y_test, predictions)
