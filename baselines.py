"""Non-graph baselines (manuscript Section 6.6).

All baselines use the same splits, encoded features and scaler as ARTGL.
  * Logistic Regression, Random Forest, XGBoost: fitted with their own solvers
    (settings below); no early stopping.
  * MLP and LSTM: trained like ARTGL, with Adam (lr 1e-3, weight decay 1e-4),
    batch size 1024, at most 200 epochs, early stopping on validation F1
    (patience 20), and dropout 0.30.
  * LSTM input: the current flow and the T-1 flows that precede it in time
    within the same split (T = 10, zero-padded at the start). It predicts the
    label of the current flow.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from metrics import classification_metrics

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

SEQ_LEN = 10


def run_classical(name: str, x_tr, y_tr, x_te, y_te, seed: int, binary: bool) -> dict:
    if name == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    elif name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.9, random_state=seed, n_jobs=-1, eval_metric="logloss")
    else:
        raise ValueError(name)
    model.fit(x_tr, y_tr)
    return classification_metrics(y_te, model.predict(x_te), binary)


class MLP(nn.Module):
    def __init__(self, d, c, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, c))

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, d, c, hidden=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, c)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1]))


def _windows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(b, T, d) windows ending at each index in idx; positions before 0 are zero."""

    offsets = torch.arange(SEQ_LEN - 1, -1, -1, device=idx.device)
    pos = idx.unsqueeze(1) - offsets.unsqueeze(0)
    out = x[pos.clamp_min(0)]
    out[pos < 0] = 0.0
    return out


def run_neural(name: str, splits: dict, cfg, seed: int, binary: bool, device) -> dict:
    torch.manual_seed(seed)
    xs = {k: torch.from_numpy(v.x).to(device) for k, v in splits.items()}
    ys = {k: torch.from_numpy(v.y).to(device) for k, v in splits.items()}
    d = xs["train"].size(1)
    c = int(max(int(ys["train"].max()) + 1, 2))
    model = (MLP(d, c, cfg.hidden_dim, cfg.dropout) if name == "mlp" else LSTMClassifier(d, c, cfg.hidden_dim, cfg.dropout)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)

    def batch_inputs(split, idx):
        return xs[split][idx] if name == "mlp" else _windows(xs[split], idx)

    @torch.no_grad()
    def predict(split):
        model.eval()
        n = xs[split].size(0)
        preds = []
        for s in range(0, n, 8192):
            idx = torch.arange(s, min(n, s + 8192), device=device)
            preds.append(model(batch_inputs(split, idx)).argmax(1))
        return torch.cat(preds).cpu().numpy()

    best, best_state, wait = -1.0, copy.deepcopy(model.state_dict()), 0
    n = xs["train"].size(0)
    for _ in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_inputs("train", idx)), ys["train"][idx])
            loss.backward()
            opt.step()
        f1 = classification_metrics(splits["val"].y, predict("val"), binary)["f1"]
        if f1 > best:
            best, best_state, wait = f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return classification_metrics(splits["test"].y, predict("test"), binary)
