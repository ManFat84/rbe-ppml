"""Logistic regression trained with mini-batch SGD, plaintext and RBE-encrypted (Section III-B).

All models use the same algorithm (zero initial weights, same mini-batch order,
same learning rate), so their results can be compared directly:

    PlainLogisticRegression(cfg, sigmoid)            baseline ("unencrypted LR")
    PlainLogisticRegression(cfg, ChebyshevSigmoid)   control: polynomial sigmoid, no encryption
    EncryptedLogisticRegression                      "encrypted LR"
"""
from __future__ import annotations

import math
import time

import numpy as np
from tqdm import tqdm

from .config import LRConfig
from .crypto import CloudContext, DataOwner


def batches(n_samples: int, cfg: LRConfig):
    """Yield (epoch, indices) for every mini-batch; identical for all models."""
    rng = np.random.default_rng(cfg.shuffle_seed)
    for epoch in range(cfg.epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, cfg.batch_size):
            yield epoch, order[start:start + cfg.batch_size]


def n_updates(n_samples: int, cfg: LRConfig) -> int:
    return cfg.epochs * math.ceil(n_samples / cfg.batch_size)


class PlainLogisticRegression:
    """Plaintext floating-point LR with binary cross-entropy gradient."""

    def __init__(self, cfg: LRConfig, activation):
        self.cfg = cfg
        self.activation = activation

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _, idx in batches(len(X), self.cfg):
            e = self.activation(X[idx] @ self.w + self.b) - y[idx]  # prediction error
            step = self.cfg.learning_rate / len(idx)
            self.w = self.w - (X[idx].T @ e) * step
            self.b = self.b - e.sum() * step
        return self

    def predict_scores(self, X):
        return self.activation(np.asarray(X, float) @ self.w + self.b)


class EncryptedLogisticRegression:
    """LR whose training data, labels and parameters stay RBE-encrypted.

    Cloud side, for every mini-batch B (public operations only):
        z_i = <W, x_i> + b             RBE multiplications and additions
        p_i = poly_sigmoid(z_i)        Horner's method
        e_i = p_i - y_i
        W  <- W - (lr/|B|) sum_i e_i x_i,   b <- b - (lr/|B|) sum_i e_i

    Key-owner side, after every update: the parameters are refreshed
    (decrypt -> round -> re-encrypt). RBE cannot reduce the fixed-point scale
    homomorphically, and one update raises it from Delta to Delta^11
    (see README, "Refresh").
    """

    def __init__(self, cfg: LRConfig, ctx: CloudContext, activation):
        self.cfg = cfg
        self.ctx = ctx
        self.activation = activation

    def fit(self, X_enc, y_enc, owner: DataOwner):
        n_samples, n_features = len(X_enc), len(X_enc[0])
        self.W = [owner.encrypt(0.0) for _ in range(n_features)]  # initial model, as in the baseline
        self.b = owner.encrypt(0.0)
        self.cloud_seconds = 0.0
        self.max_scale_exp = 0
        total = n_updates(n_samples, self.cfg)
        for _, idx in tqdm(batches(n_samples, self.cfg), total=total, desc="encrypted SGD", unit="step"):
            t0 = time.perf_counter()
            gW, gb = self._batch_gradient(X_enc, y_enc, idx)
            step = self.cfg.learning_rate / len(idx)
            W_new = [w - g * step for w, g in zip(self.W, gW)]
            b_new = self.b - gb * step
            self.cloud_seconds += time.perf_counter() - t0
            self.max_scale_exp = max(self.max_scale_exp, W_new[0].exp, b_new.exp)
            self.W = [owner.refresh(w) for w in W_new]  # key-owner side
            self.b = owner.refresh(b_new)
        return self

    def _batch_gradient(self, X_enc, y_enc, idx):
        gW, gb = None, None
        for i in idx:
            x = X_enc[i]
            z = self.ctx.dot(self.W, x) + self.b
            e = self.activation.encrypted(z) - y_enc[i]
            g = [e * xj for xj in x]
            gW = g if gW is None else [a + c for a, c in zip(gW, g)]
            gb = e if gb is None else gb + e
        return gW, gb

    def predict_scores(self, X_enc):
        """Encrypted inference: Enc(poly_sigmoid(<W, x> + b)) for every encrypted row."""
        return [
            self.activation.encrypted(self.ctx.dot(self.W, x) + self.b)
            for x in tqdm(X_enc, desc="encrypted inference", unit="row")
        ]
