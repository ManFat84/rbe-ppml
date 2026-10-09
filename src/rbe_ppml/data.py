"""Download and load the three datasets of the paper (all stored under data/raw/).

    breast cancer  UCI Breast Cancer Wisconsin (Diagnostic), 569 x 30  [13]
    heart disease  UCI Heart Disease, Cleveland subset,      303 x 13  [14]
    CIFAR-10       python version, 50,000 train / 10,000 test images   [8]
"""
from __future__ import annotations

import hashlib
import pickle
import shutil
import tarfile
import urllib.error
import warnings
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import RAW_DIR

BREAST_CANCER_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
HEART_DISEASE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

BREAST_CANCER_DIR = RAW_DIR / "breast_cancer"
HEART_DISEASE_DIR = RAW_DIR / "heart_disease"
CIFAR10_ARCHIVE = RAW_DIR / "cifar-10-python.tar.gz"
CIFAR10_DIR = RAW_DIR / "cifar-10-batches-py"

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# Downloading ----------------------------------------------------------------
def download_file(url: str, dest: Path, chunk_size: int = 1 << 16) -> Path:
    """Download url to dest with a progress bar. An interrupted download resumes on the next call."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "rbe-ppml"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as err:
        if err.code == 416 and start:  # the partial file is already complete
            partial.replace(dest)
            return dest
        raise
    with response:
        resume = bool(start) and response.status == 206
        if not resume:
            start = 0
        total = start + response.length if response.length is not None else None
        with open(partial, "ab" if resume else "wb") as f, tqdm(
            total=total, initial=start, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := response.read(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    partial.replace(dest)
    return dest


def md5sum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_zip_members(url: str, folder: Path, members: tuple[str, ...]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    archive = download_file(url, folder / "download.zip")
    with zipfile.ZipFile(archive) as z:
        for name in members:
            with z.open(name) as src, open(folder / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    archive.unlink()


def download_breast_cancer(force: bool = False) -> None:
    if (BREAST_CANCER_DIR / "wdbc.data").exists() and not force:
        print(f"  already present: {BREAST_CANCER_DIR / 'wdbc.data'}")
        return
    _download_zip_members(BREAST_CANCER_URL, BREAST_CANCER_DIR, ("wdbc.data", "wdbc.names"))


def download_heart_disease(force: bool = False) -> None:
    if (HEART_DISEASE_DIR / "processed.cleveland.data").exists() and not force:
        print(f"  already present: {HEART_DISEASE_DIR / 'processed.cleveland.data'}")
        return
    _download_zip_members(HEART_DISEASE_URL, HEART_DISEASE_DIR, ("processed.cleveland.data", "heart-disease.names"))


def download_cifar10(force: bool = False) -> None:
    if (CIFAR10_DIR / "test_batch").exists() and not force:
        print(f"  already present: {CIFAR10_DIR}")
        return
    if not CIFAR10_ARCHIVE.exists() or force:
        print("  downloading 163 MB; if it stops, run the script again and it will resume")
        download_file(CIFAR10_URL, CIFAR10_ARCHIVE)
    print("  checking MD5 ...")
    if md5sum(CIFAR10_ARCHIVE) != CIFAR10_MD5:
        raise RuntimeError(f"{CIFAR10_ARCHIVE} is corrupted (MD5 mismatch): delete it and run the script again")
    print("  extracting ...")
    with tarfile.open(CIFAR10_ARCHIVE, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(RAW_DIR, filter="data")
        else:
            tar.extractall(RAW_DIR)


# Loading ------------------------------------------------------------------
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: poetry run python scripts/01_download_data.py")


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """569 x 30 features; label 1 = malignant (M), 0 = benign (B)."""
    path = BREAST_CANCER_DIR / "wdbc.data"
    _require(path)
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 2:].to_numpy(dtype=float)
    y = (df.iloc[:, 1] == "M").astype(int).to_numpy()
    return X, y


def load_heart_disease() -> tuple[np.ndarray, np.ndarray]:
    """303 x 13 features (Cleveland); label 1 = disease (num > 0). Missing values ('?') become NaN."""
    path = HEART_DISEASE_DIR / "processed.cleveland.data"
    _require(path)
    df = pd.read_csv(path, header=None, na_values="?")
    X = df.iloc[:, :13].to_numpy(dtype=float)
    y = (df.iloc[:, 13] > 0).astype(int).to_numpy()
    return X, y


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NumPy >= 2.4 warns about the dtype format of these old pickle files
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3) uint8
    return images, np.asarray(batch[b"labels"], dtype=np.int64)


def load_cifar10_binary(classes: tuple[int, int] = (0, 1)):
    """Images of two CIFAR-10 classes. Returns X_train, y_train, X_test, y_test with y = 1 for classes[1].

    Each batch file is filtered as soon as it is loaded, so only the images of the two
    classes (about 30 MB) are kept in memory instead of all 50,000 training images.
    The selected images and their order are the same as when filtering after loading everything.
    """
    _require(CIFAR10_DIR / "test_batch")

    def load_selected(path: Path):
        images, labels = _load_cifar_batch(path)
        mask = np.isin(labels, classes)
        return images[mask], (labels[mask] == classes[1]).astype(int)

    train = [load_selected(CIFAR10_DIR / f"data_batch_{i}") for i in range(1, 6)]
    X_train = np.concatenate([images for images, _ in train])
    y_train = np.concatenate([labels for _, labels in train])
    del train
    X_test, y_test = load_selected(CIFAR10_DIR / "test_batch")
    return X_train, y_train, X_test, y_test
