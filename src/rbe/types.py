from typing import List, TypeAlias
import numpy as np

Ciphertext: TypeAlias = List[int]
Plaintext: TypeAlias = int | float

EncryptedImage: TypeAlias = np.ndarray
PlainImage: TypeAlias = np.ndarray

ImageArray: TypeAlias = np.ndarray
RGBImage: TypeAlias = np.ndarray
