"""Chebyshev approximation of the sigmoid, evaluated with Horner's method (Section III-B).

The polynomial approximates sigma(S*u) for the scaled input u = z/S in [-1, 1]
(Fig. 2), which is the same as approximating sigma(z) for z in [-S, S].
It is stored in power form p(z) = a0 + a1 z + ... + ad z^d.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class ChebyshevSigmoid:
    def __init__(self, degree: int = 3, input_range: float = 8.0):
        self.degree = degree
        self.input_range = input_range
        cheb = Chebyshev.interpolate(sigmoid, degree, domain=[-input_range, input_range])
        self.coeffs = cheb.convert(kind=Polynomial).coef  # a0, a1, ..., a_d (coefficients of z^k)

    def __call__(self, z):
        """Plaintext evaluation (Horner)."""
        z = np.asarray(z, dtype=float)
        acc = np.full_like(z, self.coeffs[-1])
        for c in self.coeffs[-2::-1]:
            acc = acc * z + c
        return acc

    def encrypted(self, z):
        """Homomorphic evaluation (Horner) of p(z) for an EncryptedNumber z.

        Uses degree-1 ciphertext x ciphertext multiplications and one
        plaintext x ciphertext multiplication.
        """
        acc = z * float(self.coeffs[-1])
        for c in self.coeffs[-2:0:-1]:
            acc = (acc + float(c)) * z
        return acc + float(self.coeffs[0])
