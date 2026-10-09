# Datasets

`poetry run python scripts/01_download_data.py` downloads all datasets into `data/raw/`:

```
data/raw/
├── breast_cancer/
│   ├── wdbc.data                  569 rows: ID, diagnosis (M/B), 30 features
│   └── wdbc.names
├── heart_disease/
│   ├── processed.cleveland.data   303 rows: 13 features + num (0-4); '?' marks a missing value
│   └── heart-disease.names
├── cifar-10-python.tar.gz         163 MB archive (MD5 c58f30108f718f92721af3b95e74349a)
└── cifar-10-batches-py/           data_batch_1 ... data_batch_5, test_batch, batches.meta
```

Sources:

- Breast cancer [13]: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip
- Heart disease [14]: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- CIFAR-10 [8]: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

If you downloaded a file yourself, put it at the path shown above and run the script again:
it skips what is already present and only extracts/checks the rest. An interrupted CIFAR-10
download resumes when the script is run again.

`data/raw/` is excluded from git (see `.gitignore`).
