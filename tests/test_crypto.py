"""The fixed-point layer (rbe_ppml.crypto) gives the same results as float arithmetic."""
import numpy as np
import pytest

from rbe_ppml.config import RBEConfig
from rbe_ppml.crypto import DataOwner
from rbe_ppml.sigmoid import ChebyshevSigmoid

STEP = 2.0**-16


@pytest.fixture(scope="module")
def owner():
    return DataOwner(RBEConfig())


def test_roundtrip(owner):
    for v in (0.0, 1.0, -3.25, 0.123456, 1234.5):
        assert abs(owner.decrypt(owner.encrypt(v)) - v) <= STEP


def test_arithmetic_tracks_the_scale(owner):
    a, b = 1.23, -4.56
    ea, eb = owner.encrypt(a), owner.encrypt(b)
    prod = ea * eb
    assert prod.exp == 2
    assert abs(owner.decrypt(prod) - a * b) < 1e-3
    assert abs(owner.decrypt(ea + eb) - (a + b)) < 1e-4
    assert abs(owner.decrypt(prod + ea) - (a * b + a)) < 1e-3
    assert abs(owner.decrypt(ea * 0.75) - 0.75 * a) < 1e-4
    assert abs(owner.decrypt(ea - 2.5) - (a - 2.5)) < 1e-4
    assert abs(owner.decrypt(3 * ea) - 3 * a) < 1e-4


def test_dot_product(owner):
    rng = np.random.default_rng(0)
    x, w = rng.normal(size=10), rng.normal(size=10)
    assert abs(owner.decrypt(owner.ctx.dot(owner.encrypt_vector(w), owner.encrypt_vector(x))) - w @ x) < 1e-3


def test_encrypted_polynomial_sigmoid(owner):
    poly = ChebyshevSigmoid(3, 8.0)
    one = owner.encrypt(1.0)
    for z in (-6.0, -1.0, 0.0, 0.3, 2.5, 7.0):
        ez = owner.encrypt(z) * one  # scale exponent 2, like a logit <w, x> + b
        assert abs(owner.decrypt(poly.encrypted(ez)) - float(poly(z))) < 1e-3


def test_refresh_resets_the_scale(owner):
    e = owner.encrypt(0.5) * owner.encrypt(0.25) * owner.encrypt(2.0)
    assert e.exp == 3
    r = owner.refresh(e)
    assert r.exp == 1
    assert abs(owner.decrypt(r) - 0.25) <= STEP
