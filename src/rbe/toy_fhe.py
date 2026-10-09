import random
from typing import List
from .base import HomomorphicScheme
from rbe import utils
from rbe.types import Ciphertext, Plaintext


class ToyFHEncryptor(HomomorphicScheme):
    def __init__(self, n: int, ln: int, scale: int, *args, **kwargs) -> None:
        """
        Initialize with modulus n and parameter ln.

        Args:
            n (int): Modulus (Composite n = p * q or large prime recommended).
            ln (int): Length of modular set used in the scheme.
            scale (int): Scale to convert floats to integers.
        """
        super().__init__(*args, **kwargs)

        if not isinstance(scale, int):
            raise TypeError("The scale must be an integer.")

        self.n = n  # Modulus (p*q)
        self.ln = ln  # Length of sets
        self.scale = scale
        self._k = None  # Base value for the set [k, k^2, ..., k^ln]
        self._sk = None
        self._pk = None
        self._enc_one = None

        self.keygen()

    def __repr__(self) -> str:
        return f"<ToyFHEncryptor n={self.n} ln={self.ln} scale={self.scale}>"

    def keygen(self) -> None:
        """
        Generates:
            - Generator k: Valid generator mod n
            - Regulator reg: Computed k^{-ln} mod n
            - Private key: List of powers of k modulo n
            - Public key: r*k^i mod n for each private key element i
        """
        self._k = utils.select_random_generator(self.n)
        while utils.extended_gcd(self._k, self.n)[0] != 1:
            self._k = utils.select_random_generator(self.n)

        self._reg = utils.modular_inverse(pow(self._k, self.ln, self.n), self.n)
        self._sk = self._generate_private_key()
        self._pk = self._generate_public_key()
        self._enc_one = self._generate_encrypted_one()

    def _generate_private_key(self) -> List[int]:
        """
        Generates a private key as a list of powers of k modulo n:
        [k^1 mod n, k^2 mod n, ..., k^ln mod n]

        Returns:
            List[int]: A list containing [k, k^2, k^3, ..., k^ln] with each value taken modulo n.
        """
        return [pow(self._k, i, self.n) for i in range(1, self.ln + 1)]

    def _generate_public_key(self) -> List[int]:
        """
        Generates a public key using the private key and a modular set summing to 1 mod n.

        Formula:
            >>> public_key[i] = (private_key[i] * ones[i]) mod n
            where 'ones' is a random modular set of length ln summing to 1 mod n.
        """
        # Generate modular set summing to 1 mod n, i.e. e(1)
        ones = utils.generate_modular_set(self.ln, self.n, 1)
        return [(self._sk[i] * ones[i]) % self.n for i in range(self.ln)]

    def _generate_encrypted_one(self) -> Ciphertext:
        """
        Calculates a set of encrypted ones using a trapdoor
        """
        base = utils.modular_multiply_sets(self._pk, self._pk, self._reg, self.n)

        # Selecting a trapdoor x for randomness increase, i.e. e(1)^x
        x = random.randint(0, (self.n - 1) // self.scale)

        result = base
        for _ in range(x):
            result = utils.modular_multiply_sets(self._pk, base, self._reg, self.n)

        return result

    def encrypt(self, m: Plaintext) -> Ciphertext:
        """
        Encrypts a plaintext integer into a ciphertext.

        Returns:
            Ciphertext (List[int]): list of length ln
        """
        if self._pk is None:
            raise RuntimeError("Public key not generated yet.")

        m, _ = utils.encode_real(m, self.scale)

        base = utils.modular_multiply_sets(self._pk, self._enc_one, self._reg, self.n)
        x = random.randint(1, 100)  # Random exponent for additional randomness
        for i in range(x):
            base = utils.modular_multiply_sets(self._pk, base, self._reg, self.n)

        c = [base[i] * m % self.n for i in range(self.ln)]

        return c

    def decrypt(self, c: Ciphertext) -> Plaintext:
        """
        Decrypt a ciphertext into a plaintext.

        Returns:
            Plaintext (int | float): plaintext result (sum of decrypted set)
        """
        if self._sk is None:
            raise RuntimeError("Private key not generated yet.")

        tmp = [0] * self.ln
        m = 0
        for i in range(self.ln):
            inv = utils.modular_inverse(pow(self._k, i + 1, self.n), self.n)
            tmp[i] = (inv * c[i]) % self.n
            m = (m + tmp[i]) % self.n

        m = utils.decode_real(m, self.scale, self.n)

        return m

    def add(self, c1: Ciphertext, c2: Ciphertext) -> Ciphertext:
        """
        Performs addition of two ciphertexts.
        """
        return [(c1[i] + c2[i]) % self.n for i in range(self.ln)]

    def multiply(self, c1: Ciphertext, c2: Ciphertext) -> Ciphertext:
        """
        Performs multiplication of two ciphertexts.
        """
        return utils.modular_multiply_sets(c1, c2, self._reg, self.n, self.scale)
