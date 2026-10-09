"""Step 3 - check the RBE implementation and the fixed-point layer before running experiments."""
import random
import time

from rbe.toy_fhe import ToyFHEncryptor
from rbe_ppml.config import LR, RBE
from rbe_ppml.crypto import DataOwner
from rbe_ppml.sigmoid import ChebyshevSigmoid


def main():
    print(f"RBE parameters: l = {RBE.ell}, n = 103*2^250+1 ({RBE.modulus.bit_length()} bits), Delta = 2^{RBE.frac_bits}")
    owner = DataOwner(RBE)
    print(f"[1] key generation: {owner.keygen_seconds:.3f} s")

    a, b = 1.23, -4.56
    ea, eb = owner.encrypt(a), owner.encrypt(b)
    poly = ChebyshevSigmoid(LR.sigmoid_degree, LR.sigmoid_range)
    print(f"[2] Enc({a}) = {ea}   (3 integers of about 77 digits)")
    checks = [
        ("a + b", ea + eb, a + b),
        ("a - b", ea - eb, a - b),
        ("a * b", ea * eb, a * b),
        ("a * b + a", ea * eb + ea, a * b + a),
        ("0.75 * a", ea * 0.75, 0.75 * a),
        ("poly_sigmoid(a)", poly.encrypted(ea), float(poly(a))),
    ]
    for name, enc, expected in checks:
        print(f"    {name:<16} decrypted {owner.decrypt(enc):+.6f}  expected {expected:+.6f}  (scale exponent {enc.exp})")

    random.seed(0)
    toy = ToyFHEncryptor(n=1000003, ln=3, scale=100)
    for x, y in [(1.5, 2.5), (1.23, 4.56)]:
        got = toy.decrypt(toy.multiply(toy.encrypt(x), toy.encrypt(y)))
        print(f"[3] original ToyFHEncryptor.multiply: {x} * {y} -> {got}   (expected {x * y:.4f})")
    print("    multiply() is correct only when the integer product is divisible by the scale,")
    print("    so rbe_ppml tracks the scale exponent and divides after decryption instead.")

    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        owner.ctx.ct_mul(ea.ct, eb.ct)
    t_mul = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(200):
        owner.encrypt(0.5)
    t_enc = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        owner.decrypt(ea)
    t_dec = (time.perf_counter() - t0) / 200
    print(f"[4] one RBE multiplication {t_mul * 1e6:.1f} us | one encryption {t_enc * 1e3:.2f} ms | "
          f"one decryption {t_dec * 1e6:.1f} us")
    print(f"[5] Enc(0) = {owner.encrypt(0.0).ct}: toy_fhe.encrypt multiplies a fixed vector by m, so 0 is not hidden")


if __name__ == "__main__":
    main()
