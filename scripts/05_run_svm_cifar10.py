"""Step 6 - linear SVM on CIFAR-10 with encrypted inference (Table IV).

Pipeline: HOG -> PCA (64 features) -> linear-kernel SVM trained in plaintext ->
test features (and, by default, w and b) encrypted with RBE -> decision function
computed on ciphertexts -> decrypted scores -> accuracy.
Results are written to results/svm_results.json and results/svm_results.md.
"""
import argparse
from dataclasses import asdict

import numpy as np
from sklearn.decomposition import PCA

from rbe_ppml import data
from rbe_ppml.config import RBE, RESULTS_DIR, SVM
from rbe_ppml.crypto import DataOwner
from rbe_ppml.metrics import Timer, markdown_table, save_json
from rbe_ppml.svm import encrypted_decision_function, hog_features, train_linear_svm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="encrypt only 200 test images (smoke test)")
    parser.add_argument("--plain-model", action="store_true", help="keep w and b in plaintext (only the images are encrypted)")
    args = parser.parse_args()
    encrypt_model = SVM.encrypt_model and not args.plain_model
    out_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR

    c0, c1 = SVM.classes
    X_tr_img, y_tr, X_te_img, y_te = data.load_cifar10_binary(SVM.classes)
    if args.quick:
        X_te_img, y_te = X_te_img[:200], y_te[:200]
    print(f"CIFAR-10 {data.CIFAR10_CLASSES[c0]} (0) vs {data.CIFAR10_CLASSES[c1]} (1): "
          f"train {len(y_tr)}, test {len(y_te)}")

    with Timer() as t_feat:
        hog_args = (SVM.hog_orientations, SVM.hog_pixels_per_cell, SVM.hog_cells_per_block)
        H_tr, H_te = hog_features(X_tr_img, *hog_args), hog_features(X_te_img, *hog_args)
        pca = PCA(n_components=SVM.pca_components, random_state=0).fit(H_tr)
        Z_tr, Z_te = pca.transform(H_tr), pca.transform(H_te)
    explained = float(pca.explained_variance_ratio_.sum())
    hog_dim = int(H_tr.shape[1])
    del X_tr_img, X_te_img, H_tr, H_te  # free memory before the encryption step
    print(f"features: HOG {hog_dim} -> PCA {Z_tr.shape[1]} (explained variance {explained:.3f}), "
          f"{t_feat.seconds:.1f} s")

    w, b, t_train = train_linear_svm(Z_tr, y_tr, SVM.C)
    plain_scores = Z_te @ w + b
    plain_acc = float(np.mean((plain_scores > 0).astype(int) == y_te))
    print(f"plaintext SVM: training {t_train:.3f} s, test accuracy {100 * plain_acc:.2f}%")

    owner = DataOwner(RBE)
    with Timer() as t_enc:
        Z_te_enc = owner.encrypt_matrix(Z_te, desc="encrypt test features")
    with Timer() as t_model:
        w_m, b_m = (owner.encrypt_vector(w), owner.encrypt(b)) if encrypt_model else (w, b)
    with Timer() as t_inf:
        scores_enc = encrypted_decision_function(owner.ctx, Z_te_enc, w_m, b_m)
    with Timer() as t_dec:
        scores = owner.decrypt_vector(scores_enc)
    enc_pred = (scores > 0).astype(int)
    enc_acc = float(np.mean(enc_pred == y_te))

    res = {
        "classes": [data.CIFAR10_CLASSES[c0], data.CIFAR10_CLASSES[c1]],
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "hog_dim": hog_dim, "pca_dim": int(Z_tr.shape[1]), "pca_explained_variance": explained,
        "encrypted_model": encrypt_model,
        "plaintext_acc": plain_acc, "encrypted_acc": enc_acc,
        "prediction_agreement": float(np.mean(enc_pred == (plain_scores > 0))),
        "max_abs_score_diff": float(np.max(np.abs(scores - plain_scores))),
        "time_s": {
            "features_hog_pca": t_feat.seconds,
            "svm_training": t_train,
            "key_generation": owner.keygen_seconds,
            "test_feature_encryption": t_enc.seconds,
            "model_encryption": t_model.seconds,
            "encrypted_inference": t_inf.seconds,
            "decryption": t_dec.seconds,
        },
    }
    pct = lambda v: f"{100 * v:.2f}%"
    table4 = markdown_table(["Plaintext ACC", "Training Time (s)", "Encrypted ACC", "Inference Latency (s)"],
                            [[pct(plain_acc), f"{t_train:.3f}", pct(enc_acc), f"{t_inf.seconds:.2f}"]])
    extra = markdown_table(["test feature encryption (s)", "decryption (s)", "prediction agreement", "max |score diff|"],
                           [[f"{t_enc.seconds:.2f}", f"{t_dec.seconds:.2f}", pct(res["prediction_agreement"]),
                             f"{res['max_abs_score_diff']:.2e}"]])
    text = (f"# SVM results (CIFAR-10, {res['classes'][0]} vs {res['classes'][1]})\n\n"
            f"Settings: {asdict(SVM)}, encrypted model: {encrypt_model}, test images: {len(y_te)}\n\n"
            f"## Table IV\n\n{table4}\n\n## Additional measurements\n\n{extra}\n")
    save_json({"settings": {"svm": asdict(SVM), "rbe": {**asdict(RBE), "modulus": str(RBE.modulus)}},
               "results": res}, out_dir / "svm_results.json")
    (out_dir / "svm_results.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"saved {out_dir / 'svm_results.json'} and {out_dir / 'svm_results.md'}")


if __name__ == "__main__":
    main()
