# RBE-based privacy-preserving machine learning

Python implementation of Section III ("Privacy-Preserving Based on RBE Technique") of the paper
*RBE-based privacy-preserving machine learning model* (Mansour Fathi, Laouid Abdelkader):

- logistic regression (LR) trained and evaluated on RBE-encrypted data: breast cancer and heart disease - linear-SVM inference on RBE-encrypted CIFAR-10 features ;
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
│   ├── 03_plot_sigmoid_approximation.py    
│   ├── 04_run_logistic_regression.py       
│   └── 05_run_svm_cifar10.py               
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



```powershell
poetry run pytest                                            # automated checks, a few seconds
poetry run python scripts/01_download_data.py                # datasets -> data/raw/ (CIFAR-10: 163 MB)
poetry run python scripts/02_check_rbe.py                    # RBE sanity checks
poetry run python scripts/03_plot_sigmoid_approximation.py   # Fig. 2 -> results/figures/
poetry run python scripts/04_run_logistic_regression.py      
poetry run python scripts/05_run_svm_cifar10.py              
```

Options: `04_... --dataset breast_cancer` (one dataset), `--quick` on scripts 04 and 05 (short smoke test,
results in `results/quick/`), `05_... --plain-model` 

Each experiment writes a JSON file (all numbers) and a Markdown file (tables in the layout of the paper) to
`results/`.