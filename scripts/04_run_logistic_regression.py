"""Step 5 - logistic regression on breast cancer and heart disease (Tables II and III).

For each dataset, three models are trained with identical SGD settings:
  unencrypted  plaintext, exact sigmoid                      (paper: "Unencrypted LR")
  polynomial   plaintext, Chebyshev sigmoid                  (control: effect of the approximation alone)
  encrypted    RBE-encrypted data, labels and parameters     (paper: "Encrypted LR")
Results are written to results/lr_results.json and results/lr_results.md.
"""
import argparse
from dataclasses import asdict, replace

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from rbe_ppml import data
from rbe_ppml.config import LR, RBE, RESULTS_DIR
from rbe_ppml.crypto import DataOwner
from rbe_ppml.logistic_regression import EncryptedLogisticRegression, PlainLogisticRegression, n_updates
from rbe_ppml.metrics import Timer, classification_metrics, markdown_table, save_json
from rbe_ppml.sigmoid import ChebyshevSigmoid, sigmoid

LOADERS = {"breast_cancer": data.load_breast_cancer, "heart_disease": data.load_heart_disease}


def prepare(X, y, cfg):
    """Stratified split, median imputation and min-max normalisation, all fitted on the training part only."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.split_seed)
    medians = np.nanmedian(X_tr, axis=0)
    X_tr = np.where(np.isnan(X_tr), medians, X_tr)
    X_te = np.where(np.isnan(X_te), medians, X_te)
    scaler = MinMaxScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


def run_dataset(name, cfg):
    X, y = LOADERS[name]()
    X_tr, X_te, y_tr, y_te = prepare(X, y, cfg)
    print(f"\n=== {name}: {X.shape[0]} samples x {X.shape[1]} features -> train {len(y_tr)}, test {len(y_te)}")
    poly = ChebyshevSigmoid(cfg.sigmoid_degree, cfg.sigmoid_range)

    with Timer() as t_base:
        base = PlainLogisticRegression(cfg, sigmoid).fit(X_tr, y_tr)
    ctrl = PlainLogisticRegression(cfg, poly).fit(X_tr, y_tr)

    owner = DataOwner(RBE)  # data owner: key generation
    with Timer() as t_data:  # data owner: encrypt the dataset
        X_tr_enc = owner.encrypt_matrix(X_tr, desc="encrypt train")
        y_tr_enc = owner.encrypt_vector(y_tr)
        X_te_enc = owner.encrypt_matrix(X_te, desc="encrypt test")
    with Timer() as t_train:  # cloud computes, data owner refreshes
        model = EncryptedLogisticRegression(cfg, owner.ctx, poly).fit(X_tr_enc, y_tr_enc, owner)
    with Timer() as t_inf:  # cloud: encrypted inference on the encrypted test set
        scores_enc = model.predict_scores(X_te_enc)
    with Timer() as t_dec:  # data owner: decrypt the predictions
        scores = owner.decrypt_vector(scores_enc)

    w_enc, b_enc = owner.decrypt_vector(model.W), owner.decrypt(model.b)
    ctrl_scores = ctrl.predict_scores(X_te)
    times = {
        "key_generation": owner.keygen_seconds,
        "baseline_training": t_base.seconds,
        "dataset_encryption": t_data.seconds,
        "encrypted_training_cloud": model.cloud_seconds,
        "refresh_key_owner": owner.refresh_seconds,
        "encrypted_training_total": t_train.seconds,
        "encrypted_inference_test_set": t_inf.seconds,
        "decryption_test_set": t_dec.seconds,
    }
    times["total_encrypted_pipeline"] = (times["key_generation"] + t_data.seconds + t_train.seconds
                                         + t_inf.seconds + t_dec.seconds)
    times["encryption_overhead"] = times["total_encrypted_pipeline"] - t_base.seconds
    return {
        "dataset": name,
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "epochs": cfg.epochs,
        "sgd_updates": n_updates(len(y_tr), cfg),
        "refreshes": owner.refresh_count,
        "max_scale_exponent": model.max_scale_exp,
        "unencrypted": classification_metrics(y_te, base.predict_scores(X_te)),
        "polynomial_plaintext": classification_metrics(y_te, ctrl_scores),
        "encrypted": classification_metrics(y_te, scores),
        "max_abs_param_diff_encrypted_vs_polynomial": float(max(np.max(np.abs(w_enc - ctrl.w)), abs(b_enc - ctrl.b))),
        "prediction_agreement_encrypted_vs_polynomial": float(np.mean((scores >= 0.5) == (ctrl_scores >= 0.5))),
        "time_s": times,
    }


def write_report(results, cfg, out_dir):
    pct = lambda v: f"{100 * v:.2f}%"
    t3 = [[r["dataset"], f"({r['shape'][0]}, {r['shape'][1]})", r["epochs"],
           *[f(r[m][k]) for m in ("unencrypted", "polynomial_plaintext", "encrypted")
             for k, f in (("acc", pct), ("auc", pct), ("mse", lambda v: f"{v:.6f}"))]] for r in results]
    table3 = markdown_table(["Dataset", "(#N, #F)", "#Iter",
                             "Unenc. ACC", "Unenc. AUC", "Unenc. MSE",
                             "Poly ACC", "Poly AUC", "Poly MSE",
                             "Enc. ACC", "Enc. AUC", "Enc. MSE"], t3)
    rows = [("baseline training time", "baseline_training"),
            ("dataset encryption time", "dataset_encryption"),
            ("encrypted training time (cloud)", "encrypted_training_cloud"),
            ("refresh time (key owner)", "refresh_key_owner"),
            ("encrypted training time (total)", "encrypted_training_total"),
            ("encrypted inference, test set", "encrypted_inference_test_set"),
            ("key generation + decryption", None),
            ("total encrypted pipeline", "total_encrypted_pipeline"),
            ("encryption overhead", "encryption_overhead")]
    t2 = []
    for label, key in rows:
        vals = []
        for r in results:
            t = r["time_s"]
            v = t["key_generation"] + t["decryption_test_set"] if key is None else t[key]
            vals.append(f"{v:.2f}")
        t2.append([label, *vals])
    table2 = markdown_table(["Metric (s)", *[r["dataset"] for r in results]], t2)
    checks = markdown_table(
        ["Dataset", "SGD updates", "refreshes", "max scale exponent", "max |param diff| enc. vs poly", "prediction agreement"],
        [[r["dataset"], r["sgd_updates"], r["refreshes"], r["max_scale_exponent"],
          f"{r['max_abs_param_diff_encrypted_vs_polynomial']:.2e}",
          pct(r["prediction_agreement_encrypted_vs_polynomial"])] for r in results])
    text = (f"# Logistic regression results\n\nSettings: {asdict(cfg)}\n\n## Table III (metrics on the test set)\n\n"
            f"{table3}\n\n## Table II (runtime)\n\n{table2}\n\n## Consistency checks\n\n{checks}\n")
    (out_dir / "lr_results.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["breast_cancer", "heart_disease", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="2 epochs only (smoke test); results in results/quick/")
    args = parser.parse_args()
    cfg = replace(LR, epochs=2) if args.quick else LR
    out_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    names = list(LOADERS) if args.dataset == "all" else [args.dataset]
    results = [run_dataset(name, cfg) for name in names]
    save_json({"settings": {"lr": asdict(cfg), "rbe": {**asdict(RBE), "modulus": str(RBE.modulus)}},
               "results": results}, out_dir / "lr_results.json")
    write_report(results, cfg, out_dir)
    print(f"saved {out_dir / 'lr_results.json'} and {out_dir / 'lr_results.md'}")


if __name__ == "__main__":
    main()
