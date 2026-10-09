"""All experiment parameters in one place.

(paper)  = value stated in the paper
(choice) = not stated in the paper; chosen for this implementation (see README.md)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Project folders ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True)
class RBEConfig:
    ell: int = 3                      # ciphertext length l (paper: l = 3)
    modulus: int = 103 * 2**250 + 1   # 257-bit prime, n - 1 = 2^250 * 103 (choice, see README)
    frac_bits: int = 16               # fixed-point precision, Delta = 2^16 (choice)
    const_exp: int = 2                # real constants (coefficients, step size) encoded with Delta^2 (choice)
    seed: int = 2026                  # seed of Python's `random` used by the RBE code (choice)

    @property
    def scale(self) -> int:
        return 2**self.frac_bits


@dataclass(frozen=True)
class LRConfig:
    test_size: float = 0.2            # 80/20 split -> 114 and 61 test samples, as implied by the paper's ACC values
    split_seed: int = 42              # (choice)
    epochs: int = 80                  # (paper: 80 iterations)
    batch_size: int = 64              # mini-batch SGD (choice)
    learning_rate: float = 0.2        # (choice)
    shuffle_seed: int = 0             # (choice)
    sigmoid_degree: int = 3           # (paper: Chebyshev approximation of degree 3)
    sigmoid_range: float = 8.0        # fitted on z in [-8, 8], i.e. scaled input u = z/8 in [-1, 1] (choice)


@dataclass(frozen=True)
class SVMConfig:
    classes: tuple[int, int] = (0, 1)  # airplane (0) vs automobile (1) (choice)
    hog_orientations: int = 9          # Dalal-Triggs HOG settings (choice)
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2
    pca_components: int = 64           # (paper: 64 features)
    C: float = 1.0                     # (choice)
    encrypt_model: bool = True         # encrypt w and b as well as the features (choice)


RBE = RBEConfig()
LR = LRConfig()
SVM = SVMConfig()
