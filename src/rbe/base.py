from __future__ import annotations
from abc import ABC, abstractmethod
from rbe.types import Ciphertext, Plaintext


class HomomorphicScheme(ABC):
    """Abstract interface every homomorphic scheme should implement."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def keygen(self):
        """Generate all necessary keys and parameters (private key, public key, etc.)."""
        pass

    @abstractmethod
    def encrypt(self, m: Plaintext) -> Ciphertext:
        """Encrypt a message m and return ciphertext."""
        ...
        pass

    @abstractmethod
    def decrypt(self, c: Ciphertext) -> Plaintext:
        """Decrypt ciphertext and return plaintext."""
        ...
        pass

    @abstractmethod
    def add(self, c1: Ciphertext, c2: Ciphertext) -> Ciphertext:
        """Homomorphic addition of two ciphertexts."""
        ...
        pass

    @abstractmethod
    def multiply(self, c1: Ciphertext, c2: Ciphertext) -> Ciphertext:
        """Homomorphic multiplication of two ciphertexts."""
        ...
        pass
