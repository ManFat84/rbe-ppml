# RBE-based privacy-preserving machine learning

Python implementation of Section III ("Privacy-Preserving Based on RBE Technique") of the paper
*RBE-based privacy-preserving machine learning model* (Mansour, Laouid, Ferik, Bounceur, Hammoudeh, Chait):

- logistic regression (LR) trained and evaluated on RBE-encrypted data: breast cancer and heart disease (Tables II and III);
- linear-SVM inference on RBE-encrypted CIFAR-10 features (Table IV);
- the sigmoid / Chebyshev approximation plot (Fig. 2).

The RBE scheme is the original implementation in `src/rbe/` (`base.py`, `utils.py`, `types.py`, `toy_fhe.py`),
used without modification.

## 1. Project structure

```
rbe-ppml/
├── pyproject.toml          Poetry configuration (dependencies, packages)
├── poetry.lock             exact versions installed by Poetry (commit it)
├── README.md
├── .gitignore
├── .vscode/settings.json   enables the pytest test panel in VS Code
├── data/
│   ├── README.md           expected dataset layout
│   └── raw/                datasets, downloaded by script 01 (not stored in git)
├── results/                tables (.md, .json) and figures written by the scripts
├── scripts/                run them in this order
│   ├── 01_download_data.py
│   ├── 02_check_rbe.py
│   ├── 03_plot_sigmoid_approximation.py    Fig. 2
│   ├── 04_run_logistic_regression.py       Tables II and III
│   └── 05_run_svm_cifar10.py               Table IV
├── src/
│   ├── rbe/                original RBE implementation (unchanged)
│   └── rbe_ppml/           implementation of the paper's method
│       ├── config.py       every parameter (paper values and implementation choices)
│       ├── data.py         download and loading of the three datasets
│       ├── crypto.py       fixed-point arithmetic on RBE ciphertexts; DataOwner and CloudContext roles
│       ├── sigmoid.py      degree-3 Chebyshev sigmoid, Horner evaluation
│       ├── logistic_regression.py   plaintext and encrypted LR (mini-batch SGD)
│       ├── svm.py          HOG features, linear SVM, encrypted decision function
│       └── metrics.py      ACC / AUC / MSE, timing, result files
└── tests/                  automated checks (pytest)
```

Do not run the files inside `src/rbe/` directly: `types.py` would then hide Python's built-in `types`
module. Always run the scripts in `scripts/`.

## 2. Setup (once)

Requirements: Python 3.12, 3.13 or 3.14 (not 3.15 yet), Git, VS Code with the Python extension, Poetry 2.x.

`pyproject.toml` must contain (besides the dependencies added with `poetry add`):

```toml
requires-python = ">=3.12,<3.15"

[tool.poetry]
packages = [
    { include = "rbe", from = "src" },
    { include = "rbe_ppml", from = "src" },
]
```

Dependencies: `poetry add numpy pandas scikit-learn scikit-image matplotlib tqdm` and
`poetry add --group dev pytest`. After changing `pyproject.toml`, run `poetry install`.

## 3. Running the experiments

From the project folder (VS Code: Terminal > New Terminal):

```powershell
poetry run pytest                                            # automated checks, a few seconds
poetry run python scripts/01_download_data.py                # datasets -> data/raw/ (CIFAR-10: 163 MB)
poetry run python scripts/02_check_rbe.py                    # RBE sanity checks
poetry run python scripts/03_plot_sigmoid_approximation.py   # Fig. 2 -> results/figures/
poetry run python scripts/04_run_logistic_regression.py      # Tables II-III, about 1-2 min
poetry run python scripts/05_run_svm_cifar10.py              # Table IV, about 2-3 min
```

Options: `04_... --dataset breast_cancer` (one dataset), `--quick` on scripts 04 and 05 (short smoke test,
results in `results/quick/`), `05_... --plain-model` (only the images are encrypted, w and b stay in plaintext).

Each experiment writes a JSON file (all numbers) and a Markdown file (tables in the layout of the paper) to
`results/`. Record a run with `git add results` and `git commit -m "..."`.

## 4. Reference run

Environment: Linux, 1 CPU core, Python 3.12.3, numpy 2.4.4, scikit-learn 1.8.0, scikit-image 0.26.0.
Runtimes depend on the processor; the paper used an Intel i5-6300MQ with Python 3.14.0.

Table III - LR metrics on the test set (114 and 61 samples), 80 epochs:

| Dataset | Model | ACC | AUC | MSE |
|---|---|---|---|---|
| Breast cancer (569, 30) | unencrypted, exact sigmoid | 93.86% | 99.90% | 0.048594 |
| | plaintext, Chebyshev sigmoid | 93.86% | 99.83% | 0.057510 |
| | **encrypted (RBE)** | **93.86%** | **99.83%** | **0.057510** |
| | paper: unencrypted / encrypted | 94.74% / 92.11% | 98.84% / 98.40% | 0.077921 / 0.089929 |
| Heart disease (303, 13) | unencrypted, exact sigmoid | 85.25% | 92.86% | 0.111970 |
| | plaintext, Chebyshev sigmoid | 81.97% | 92.42% | 0.116532 |
| | **encrypted (RBE)** | **81.97%** | **92.42%** | **0.116532** |
| | paper: unencrypted / encrypted | 90.16% / 88.52% | 96.53% / 96.32% | 0.091091 / 0.098702 |

The encrypted model reproduces the plaintext model with the same polynomial sigmoid: its parameters differ
by at most 1.7e-4 (fixed-point rounding at 2^-16) and all test predictions are identical. The difference
between the unencrypted and the encrypted columns is therefore due to the polynomial approximation of the
sigmoid, not to the RBE computation.

Table II - LR runtime (s):

| Metric | Breast cancer | Heart disease | Paper (BC / HD) |
|---|---|---|---|
| baseline training | 0.02 | 0.01 | 0.04 / 0.05 |
| dataset encryption (train + test + labels) | 10.29 | 2.42 | 0.12 / 0.02 |
| encrypted training, cloud computation | 31.09 | 6.11 | - |
| refresh, key owner | 17.04 | 3.74 | - |
| encrypted training, total | 48.20 | 9.87 | 18.84 / 9.70 |
| encrypted inference, whole test set | 0.05 | 0.01 | - |
| total encrypted pipeline | 58.56 | 12.32 | 18.96 / 9.72 |

Table IV - linear SVM on CIFAR-10, airplane vs automobile (10,000 training / 2,000 test images,
HOG 324 -> PCA 64 features, explained variance 0.810; w and b encrypted):

| | Plaintext ACC | Training time (s) | Encrypted ACC | Inference latency (s) |
|---|---|---|---|---|
| this run | 88.95% | 2.375 | 88.95% | 1.65 |
| paper | 89.1% | 2.185 | 88.9% | 1.62 |

Inference latency = homomorphic evaluation of the decision function for all 2,000 encrypted test vectors.
Additional measurements: encryption of the test features (128,000 values) 77.00 s,
decryption of the scores 0.53 s, agreement between encrypted and plaintext predictions
100.00%, largest difference between decrypted and plaintext scores 1.3e-04.

## 5. Correspondence with the paper

| Paper | Code |
|---|---|
| RBE over Z_n, l = 3, regulator k^-l (III-A) | `config.RBEConfig`, `crypto.RBEScheme` (subclass of `ToyFHEncryptor`) |
| fixed-point encoding with a scaling factor (III-A) | `crypto.CloudContext.encode`, `crypto.EncryptedNumber` |
| encrypted dataset Enc(Z) (N x F x l) and encrypted labels (III-B) | `DataOwner.encrypt_matrix`, `DataOwner.encrypt_vector` |
| degree-3 Chebyshev sigmoid, Horner's method (III-B, Fig. 2) | `sigmoid.ChebyshevSigmoid` |
| encrypted SGD: ciphertext-weight products, homomorphic aggregation (III-B) | `logistic_regression.EncryptedLogisticRegression` |
| HOG -> PCA (64) -> linear SVM trained in plaintext (III-C) | `svm.hog_features`, `svm.train_linear_svm` |
| encrypted decision function w Enc(z) + b (III-C) | `svm.encrypted_decision_function` |

Parameters not stated in the paper (all in `config.py`): fixed-point precision Delta = 2^16; Chebyshev fit on
z in [-8, 8] (scaled input u = z/8 in [-1, 1]); min-max normalisation fitted on the training split; median
imputation of the 6 missing heart-disease values; stratified 80/20 split (seed 42), consistent with the paper's
ACC values (94.74% = 108/114, 90.16% = 55/61); mini-batch size 64; learning rate 0.2; zero initial weights;
CIFAR-10 classes airplane vs automobile (10,000 training / 2,000 test images, as in the paper); HOG with
9 orientations, 8x8-pixel cells, 2x2-cell blocks on grayscale images; `SVC(kernel="linear", C=1)`.

## 6. Differences from the paper and their reasons

1. **Refresh during encrypted training.** A fresh ciphertext holds its value at scale Delta. One SGD update
   multiplies data and weights (Delta^2), evaluates the cubic sigmoid (Delta^8), multiplies by the data again
   (Delta^9) and by the step size (Delta^11). RBE has no homomorphic operation that divides by Delta (the
   plaintext space is the integers modulo n), and without it the exponent would reach about 41 after the second
   update and about 131 after the third, beyond any practical modulus. After every update the key owner
   therefore decrypts the parameters, rounds them to precision Delta and re-encrypts them
   (`DataOwner.refresh`). The cloud never sees plaintext data or parameters; the cost (one decryption and one
   encryption per parameter and update) is reported separately. The paper's statement that the parameters are
   decrypted only after convergence is not reproduced.
2. **Key generation.** `ToyFHEncryptor._generate_encrypted_one` runs a loop of up to (n-1)/scale RBE
   multiplications, which never finishes for a modulus of realistic size (and does not accumulate the product).
   `RBEScheme` computes the intended value pk^(x+2) with square-and-multiply. `src/rbe/` is not modified.
3. **Multiplication.** `ToyFHEncryptor.multiply` divides by the scale with a modular inverse, which is correct
   only when the integer product is divisible by the scale (1.5 x 2.5 is correct, 1.23 x 4.56 is not). The code
   uses the RBE product `utils.modular_multiply_sets` without that division and tracks the scale exponent of each
   ciphertext (`EncryptedNumber.exp`); the division happens after decryption, which is exact while
   |value| x Delta^exp < n/2.
4. **Modulus.** The regular RBE construction uses n = pq. The generator test in `toy_fhe.py` factors n - 1 by trial
   division, which is only fast when n - 1 has small prime factors. The code uses the prime n = 103 x 2^250 + 1
   (n - 1 = 2^250 x 103); its 257 bits hold the largest scale used (Delta^11 = 2^176) with margin. These
   parameters are chosen for functional evaluation (accuracy, runtime), not for security.

## 7. Observations on `toy_fhe.py` relevant to confidentiality

These do not change the accuracy or runtime results.

- `encrypt(m)` returns m times one of 100 fixed vectors (powers of the public key). Hence `encrypt(0)` is
  always `[0, 0, 0]`, so every encrypted label y = 0 and every feature equal to 0 is visible, and two
  ciphertexts built from the same vector reveal the ratio of their plaintexts.
- With the original key generation (encrypted one = pk^3), these 100 vectors can be computed from the public key
  and the regulator, which the cloud needs for multiplication. In a test, the plaintexts 3.14, -2.5, 0.07 and
  123.45 were recovered exactly from their ciphertexts without the secret key.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'rbe'` or `'rbe_ppml'` | check the `packages` lines in `pyproject.toml`, then `poetry install` |
| strange errors mentioning `types` | a file inside `src/rbe/` was run directly; run the scripts instead |
| `poetry.lock` is out of date warning | `poetry lock`, then `poetry install` |
| CIFAR-10 download slow or interrupted | run script 01 again (it resumes), or download the archive manually into `data/raw/` |
| package installation fails on Python 3.15 | install Python 3.13 or 3.14, then `poetry env use (py -3.13 -c "import sys; print(sys.executable)")` and `poetry install` |
| an experiment takes too long | first try `--quick` |
