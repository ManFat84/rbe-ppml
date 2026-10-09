"""CIFAR-10 pipeline: HOG + PCA features, plaintext linear SVM, encrypted inference (Section III-C)."""
from __future__ import annotations

import time

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.svm import SVC
from tqdm import tqdm

from .crypto import CloudContext, EncryptedNumber


def hog_features(images, orientations: int = 9, pixels_per_cell: int = 8, cells_per_block: int = 2) -> np.ndarray:
    """HOG descriptor of every 32x32x3 image, computed on the grayscale image."""
    return np.stack([
        hog(
            rgb2gray(img),
            orientations=orientations,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        for img in tqdm(images, desc="HOG", unit="img")
    ])


def train_linear_svm(Z, y, C: float = 1.0):
    """Plaintext training of a linear-kernel SVM. Returns (w, b, seconds)."""
    t0 = time.perf_counter()
    clf = SVC(kernel="linear", C=C).fit(Z, y)
    seconds = time.perf_counter() - t0
    return clf.coef_.ravel().copy(), float(clf.intercept_[0]), seconds


def encrypted_decision_function(ctx: CloudContext, Z_enc, w, b) -> list[EncryptedNumber]:
    """f(z) = <w, z> + b on encrypted feature vectors.

    If b is an EncryptedNumber, w must be a list of EncryptedNumbers (encrypted
    model: ciphertext x ciphertext products). Otherwise w and b are plaintext
    numbers (plaintext x ciphertext products).
    """
    if isinstance(b, EncryptedNumber):
        return [ctx.dot(w, z) + b for z in tqdm(Z_enc, desc="encrypted inference", unit="img")]
    scores = []
    for z in tqdm(Z_enc, desc="encrypted inference", unit="img"):
        acc = z[0] * float(w[0])
        for wj, zj in zip(w[1:], z[1:]):
            acc = acc + zj * float(wj)
        scores.append(acc + float(b))
    return scores
