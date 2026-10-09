"""Step 4 - Fig. 2: sigmoid vs its degree-3 Chebyshev approximation."""
import matplotlib

matplotlib.use("Agg")  # write the figure to a file without opening a window
import matplotlib.pyplot as plt
import numpy as np

from rbe_ppml.config import FIGURES_DIR, LR
from rbe_ppml.sigmoid import ChebyshevSigmoid, sigmoid


def main():
    S = LR.sigmoid_range
    poly = ChebyshevSigmoid(LR.sigmoid_degree, S)
    u = np.linspace(-1.0, 1.0, 801)
    z = S * u
    err = np.abs(poly(z) - sigmoid(z))
    print("coefficients a0..a3 of p(z):", ", ".join(f"{c:+.6f}" for c in poly.coeffs))
    print(f"max |sigmoid(z) - p(z)| for z in [-{S:g}, {S:g}]: {err.max():.4f} (at z = {z[err.argmax()]:+.2f})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(u, sigmoid(z), label="sigmoid")
    ax.plot(u, poly(z), "--", label=f"Chebyshev approx. (deg={LR.sigmoid_degree})")
    ax.set_xlabel(f"scaled input u = z / {S:g},   z = w^T x + b")
    ax.set_ylabel("output")
    ax.set_title("Sigmoid vs polynomial approximation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_sigmoid_chebyshev.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
