"""Evaluation metrics, timing and result files."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """ACC, AUC and MSE (Table III).

    ACC: scores >= threshold; AUC: raw scores; MSE: mean squared difference
    between the scores clipped to [0, 1] (probabilities) and the labels.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    return {
        "acc": float(accuracy_score(y_true, (scores >= threshold).astype(int))),
        "auc": float(roc_auc_score(y_true, scores)),
        "mse": float(np.mean((np.clip(scores, 0.0, 1.0) - y_true) ** 2)),
    }


class Timer:
    """Usage: with Timer() as t: ...   then t.seconds"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def markdown_table(headers, rows) -> str:
    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines)
