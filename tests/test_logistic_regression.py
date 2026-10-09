"""Encrypted training gives (almost) the same model as plaintext training with the same polynomial."""
from dataclasses import replace

import numpy as np

from rbe_ppml.config import LRConfig, RBEConfig
from rbe_ppml.crypto import DataOwner
from rbe_ppml.logistic_regression import EncryptedLogisticRegression, PlainLogisticRegression
from rbe_ppml.sigmoid import ChebyshevSigmoid


def test_encrypted_lr_matches_plaintext_polynomial_lr():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(40, 3))
    y = (X @ np.array([1.5, -2.0, 0.5]) + 0.1 > 0).astype(int)
    cfg = replace(LRConfig(), epochs=3, batch_size=16)
    poly = ChebyshevSigmoid(3, 8.0)

    plain = PlainLogisticRegression(cfg, poly).fit(X, y)
    owner = DataOwner(RBEConfig())
    model = EncryptedLogisticRegression(cfg, owner.ctx, poly).fit(owner.encrypt_matrix(X), owner.encrypt_vector(y), owner)

    assert np.max(np.abs(owner.decrypt_vector(model.W) - plain.w)) < 1e-3
    assert abs(owner.decrypt(model.b) - plain.b) < 1e-3
