"""The original RBE files (src/rbe) work as expected for small parameters."""
import random

from rbe.toy_fhe import ToyFHEncryptor


def make():
    random.seed(0)
    return ToyFHEncryptor(n=1000003, ln=3, scale=100)


def test_encrypt_decrypt():
    s = make()
    for m in (0.0, 1.23, -4.56, 99.99):
        assert abs(s.decrypt(s.encrypt(m)) - m) < 1e-9


def test_addition():
    s = make()
    assert abs(s.decrypt(s.add(s.encrypt(1.23), s.encrypt(4.56))) - 5.79) < 1e-9


def test_multiplication_when_product_divisible_by_scale():
    s = make()
    assert abs(s.decrypt(s.multiply(s.encrypt(1.5), s.encrypt(2.5))) - 3.75) < 1e-9
