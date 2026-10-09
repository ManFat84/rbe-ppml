"""Step 2 - download the three datasets into data/raw/ and print their sizes."""
import argparse

import numpy as np

from rbe_ppml import data

JOBS = {
    "breast_cancer": data.download_breast_cancer,
    "heart_disease": data.download_heart_disease,
    "cifar10": data.download_cifar10,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(JOBS), help="download a single dataset")
    parser.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = parser.parse_args()

    for name, download in JOBS.items():
        if args.only and name != args.only:
            continue
        print(f"== {name}")
        download(force=args.force)

    print("\nSummary")
    if not args.only or args.only == "breast_cancer":
        X, y = data.load_breast_cancer()
        print(f"  breast cancer : {X.shape[0]} samples x {X.shape[1]} features, {y.sum()} malignant")
    if not args.only or args.only == "heart_disease":
        X, y = data.load_heart_disease()
        print(f"  heart disease : {X.shape[0]} samples x {X.shape[1]} features, {y.sum()} with disease, "
              f"{int(np.isnan(X).sum())} missing values")
    if not args.only or args.only == "cifar10":
        X_tr, y_tr, X_te, y_te = data.load_cifar10_binary((0, 1))
        print(f"  CIFAR-10      : airplane vs automobile -> {len(y_tr)} train / {len(y_te)} test images")
    print(f"Files are in {data.RAW_DIR}")


if __name__ == "__main__":
    main()
