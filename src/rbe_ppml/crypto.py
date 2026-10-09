"""Fixed-point encrypted arithmetic on top of the original RBE implementation.

Roles (Fig. 1 of the paper)
---------------------------
DataOwner     holds the RBE secret key: generates keys, encrypts data and model
              parameters, decrypts results and, during training, refreshes
              ciphertexts.
CloudContext  the public material sent to the cloud: modulus n, length l,
              public key pk and regulator. It is enough to add and multiply
              ciphertexts, but not to decrypt.

Fixed-point encoding
--------------------
RBE computes on integers modulo n. A real number v is represented by the
integer round(v * Delta**e), with Delta = 2**frac_bits and e the "scale
exponent" of the ciphertext. Fresh encryptions have e = 1.
  * adding two ciphertexts needs equal exponents; the smaller one is raised
    exactly by multiplying it with the integer Delta**k;
  * multiplying two ciphertexts adds their exponents;
  * decryption returns integer / Delta**e.
This is exact as long as |v| * Delta**e < n / 2.

ToyFHEncryptor.multiply() is not used because it divides by the scale with a
modular inverse, which is correct only when the integer product is divisible
by the scale (1.5 * 2.5 works, 1.23 * 4.56 does not).
"""
from __future__ import annotations

import numbers
import random
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from tqdm import tqdm

from rbe import utils
from rbe.toy_fhe import ToyFHEncryptor
from rbe.types import Ciphertext

from .config import RBEConfig


def rbe_power(c: Ciphertext, e: int, reg: int, n: int) -> Ciphertext:
    """c (.) c (.) ... (.) c with e >= 1 factors, by square-and-multiply."""
    if e < 1:
        raise ValueError("exponent must be >= 1")
    result, base = None, c
    while e:
        if e & 1:
            result = base if result is None else utils.modular_multiply_sets(result, base, reg, n)
        e >>= 1
        if e:
            base = utils.modular_multiply_sets(base, base, reg, n)
    return result


class RBEScheme(ToyFHEncryptor):
    """The original ToyFHEncryptor with a fast computation of the encrypted one.

    toy_fhe.py draws a trapdoor x in [0, (n-1)//scale] and then runs a loop of x
    RBE multiplications (whose body does not accumulate the product). For a
    modulus of realistic size this loop never finishes. Here the intended value
    e(1)^(x+2) = pk^(x+2) is computed with square-and-multiply, in O(log x)
    multiplications. Key generation, encrypt() and decrypt() are inherited
    unchanged from toy_fhe.py.
    """

    def _generate_encrypted_one(self) -> Ciphertext:
        x = random.randint(0, (self.n - 1) // self.scale)
        return rbe_power(self._pk, x + 2, self._reg, self.n)

    @property
    def public_key(self) -> tuple[int, ...]:
        return tuple(self._pk)

    @property
    def regulator(self) -> int:
        return self._reg


@dataclass(frozen=True)
class CloudContext:
    """Public RBE material plus fixed-point settings (what the cloud receives)."""

    n: int
    ell: int
    public_key: tuple[int, ...]
    regulator: int
    frac_bits: int
    const_exp: int
    max_exp: int

    def encode(self, value: float, exp: int) -> int:
        """Integer representing `value` at scale Delta**exp."""
        return round(float(value) * float(2 ** (self.frac_bits * exp)))

    def delta_pow(self, k: int) -> int:
        return 1 << (self.frac_bits * k)

    def check_exp(self, exp: int) -> None:
        if exp > self.max_exp:
            raise OverflowError(
                f"scale exponent {exp} exceeds what the modulus can hold (max {self.max_exp}); "
                "refresh the ciphertext first"
            )

    # Raw ciphertext operations (lists of l integers modulo n) ----------------
    def ct_add(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        return [(x + y) % self.n for x, y in zip(a, b)]

    def ct_neg(self, a: Ciphertext) -> Ciphertext:
        return [(-x) % self.n for x in a]

    def ct_scalar(self, a: Ciphertext, k: int) -> Ciphertext:
        return [(x * k) % self.n for x in a]

    def ct_mul(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        """RBE homomorphic multiplication (utils.modular_multiply_sets, without scale)."""
        return utils.modular_multiply_sets(a, b, self.regulator, self.n)

    def ct_plain(self, m: int) -> Ciphertext:
        """Ciphertext of the integer m from the public key: Dec(m * pk) = m * sum(r_i) = m."""
        return [(p * m) % self.n for p in self.public_key]

    def dot(self, a: Sequence[EncryptedNumber], b: Sequence[EncryptedNumber]) -> EncryptedNumber:
        """Encrypted inner product sum_j a_j * b_j (one RBE multiplication per term)."""
        exp = a[0].exp + b[0].exp
        self.check_exp(exp)
        n, reg = self.n, self.regulator
        acc = [0] * self.ell
        for x, y in zip(a, b, strict=True):
            if x.exp + y.exp != exp:
                raise ValueError("dot() expects all terms to have the same scale exponent")
            prod = utils.modular_multiply_sets(x.ct, y.ct, reg, n)
            acc = [(s + t) % n for s, t in zip(acc, prod)]
        return EncryptedNumber(acc, exp, self)


class EncryptedNumber:
    """An RBE ciphertext together with its fixed-point scale exponent.

    Supports +, -, * with other EncryptedNumbers and with plaintext numbers.
    """

    __slots__ = ("ct", "exp", "ctx")

    def __init__(self, ct: Ciphertext, exp: int, ctx: CloudContext):
        self.ct = ct
        self.exp = exp
        self.ctx = ctx

    def __repr__(self) -> str:
        return f"EncryptedNumber(exp={self.exp}, ct=[{str(self.ct[0])[:12]}..., ...])"

    def raised_to(self, exp: int) -> EncryptedNumber:
        """The same value at a larger scale exponent (exact: multiply by Delta**k)."""
        if exp == self.exp:
            return self
        if exp < self.exp:
            raise ValueError("the scale exponent cannot be lowered homomorphically (use a refresh)")
        self.ctx.check_exp(exp)
        return EncryptedNumber(self.ctx.ct_scalar(self.ct, self.ctx.delta_pow(exp - self.exp)), exp, self.ctx)

    def __add__(self, other):
        ctx = self.ctx
        if isinstance(other, EncryptedNumber):
            exp = max(self.exp, other.exp)
            return EncryptedNumber(ctx.ct_add(self.raised_to(exp).ct, other.raised_to(exp).ct), exp, ctx)
        if isinstance(other, numbers.Real):  # plaintext constant
            return EncryptedNumber(ctx.ct_add(self.ct, ctx.ct_plain(ctx.encode(other, self.exp))), self.exp, ctx)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> EncryptedNumber:
        return EncryptedNumber(self.ctx.ct_neg(self.ct), self.exp, self.ctx)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        ctx = self.ctx
        if isinstance(other, EncryptedNumber):  # ciphertext x ciphertext
            exp = self.exp + other.exp
            ctx.check_exp(exp)
            return EncryptedNumber(ctx.ct_mul(self.ct, other.ct), exp, ctx)
        if isinstance(other, numbers.Integral):  # exact integer: no rescaling
            return EncryptedNumber(ctx.ct_scalar(self.ct, int(other)), self.exp, ctx)
        if isinstance(other, numbers.Real):  # real constant, encoded with Delta**const_exp
            exp = self.exp + ctx.const_exp
            ctx.check_exp(exp)
            return EncryptedNumber(ctx.ct_scalar(self.ct, ctx.encode(other, ctx.const_exp)), exp, ctx)
        return NotImplemented

    __rmul__ = __mul__


class DataOwner:
    """Holds the RBE secret key: key generation, encryption, decryption, refresh."""

    def __init__(self, cfg: RBEConfig):
        self.cfg = cfg
        random.seed(cfg.seed)  # the RBE code uses Python's `random`; seeding makes runs reproducible
        t0 = time.perf_counter()
        self.scheme = RBEScheme(n=cfg.modulus, ln=cfg.ell, scale=cfg.scale)
        self.keygen_seconds = time.perf_counter() - t0
        margin_bits = 32
        self.ctx = CloudContext(
            n=cfg.modulus,
            ell=cfg.ell,
            public_key=self.scheme.public_key,
            regulator=self.scheme.regulator,
            frac_bits=cfg.frac_bits,
            const_exp=cfg.const_exp,
            max_exp=(cfg.modulus.bit_length() - 1 - margin_bits) // cfg.frac_bits,
        )
        self.refresh_count = 0
        self.refresh_seconds = 0.0

    # Encryption (ToyFHEncryptor.encrypt, scale exponent 1) -----------------
    def encrypt(self, value: float) -> EncryptedNumber:
        return EncryptedNumber(self.scheme.encrypt(float(value)), 1, self.ctx)

    def encrypt_vector(self, values) -> list[EncryptedNumber]:
        return [self.encrypt(v) for v in np.asarray(values, dtype=float).ravel()]

    def encrypt_matrix(self, X, desc: str = "encrypting") -> list[list[EncryptedNumber]]:
        X = np.asarray(X, dtype=float)
        return [[self.encrypt(v) for v in row] for row in tqdm(X, desc=desc, unit="row")]

    # Decryption (ToyFHEncryptor.decrypt returns integer / Delta) ------------
    def decrypt(self, enc: EncryptedNumber) -> float:
        return self.scheme.decrypt(enc.ct) / float(self.ctx.delta_pow(enc.exp - 1))

    def decrypt_vector(self, encs: Sequence[EncryptedNumber]) -> np.ndarray:
        return np.array([self.decrypt(e) for e in encs])

    # Refresh (key-owner side) -----------------------------------------------
    def refresh(self, enc: EncryptedNumber) -> EncryptedNumber:
        """Decrypt, round to the precision Delta and re-encrypt (scale exponent back to 1)."""
        t0 = time.perf_counter()
        fresh = self.encrypt(self.decrypt(enc))
        self.refresh_seconds += time.perf_counter() - t0
        self.refresh_count += 1
        return fresh
