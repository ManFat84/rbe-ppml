import random
from typing import List, Tuple


def prime_factors(n: int) -> set:
    """
    Returns the set of prime factors of n.
    """
    factors = set()
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.add(i)
    if n > 1:
        factors.add(n)

    return factors


def is_generator(g: int, n: int) -> bool:
    """
    Checks if g is a generator of the group Z_n* (multiplicative group modulo n).

    Args:
        g (int): Candidate generator.
        n (int): Modulus of the group.

    Returns:
        bool: True if g is a generator, False otherwise.
    """

    # Group order is n - 1 for Z_n*
    group_order = n - 1

    # Factor the group order (we know n is prime, so we check prime divisors)
    factors = prime_factors(group_order)

    # g is a generator if g^(group_order / q) mod n != 1 for all q in factors
    for factor in factors:
        if pow(g, group_order // factor, n) == 1:
            return False

    return True


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Computes the greatest common divisor (GCD) of two integers a and b,
    along with the coefficients x and y such that: a*x + b*y = gcd(a, b).
    This is known as the Extended Euclidean Algorithm.

    Args:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        Tuple[int, int, int]: A tuple containing (gcd, x, y) where
        gcd is the greatest common divisor of a and b,
        x and y are the coefficients satisfying Bézout's identity.

    Example:
        >>> extended_gcd(30, 20)
        (10, 1, -1)
        # 30*1 + 20*(-1) = 10
    """
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return gcd, x, y


def modular_inverse(a: int, m: int) -> int:
    """
    Computes the modular inverse of a under modulus m,
    i.e., x such that (a * x) % m == 1.

    It uses the Extended Euclidean Algorithm.

    Args:
        a (int): The number to find the modular inverse of.
        m (int): The modulus.

    Returns:
        int: The modular inverse of a mod m if it exists, else None if the inverse does not exist.
    """
    # Calculate gcd and the coefficients using the extended Euclidean algorithm
    gcd, x, y = extended_gcd(a, m)

    # Modular inverse exists only if gcd is 1
    if gcd != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}")
    else:
        # x might be negative, so we take x % m to get the positive value
        return x % m


def select_random_generator(n: int) -> int:
    while True:
        g = random.randint(2, n - 2)
        if is_generator(g, n):
            return g


def generate_modular_set(ln: int, n: int, target_sum: int) -> List[int]:
    """
    Generates a list of length l of random values
    such that their sum modulo n equals target_sum.

    Args:
        target_sum (int): The target sum modulo n.

    Returns:
        List[int]: List of integers of length self.l satisfying the sum condition.
    """
    # Create a list of random values mod n
    values = [random.randint(0, n - 1) for _ in range(ln - 1)]

    # Calculate the sum of the current values modulo n
    current_sum_mod_n = sum(values) % n

    # Calculate the last value needed to satisfy the condition
    last_value = (target_sum - current_sum_mod_n) % n

    # Append the last value to the set
    values.append(last_value)

    return values


def modular_multiply_sets(set1: List[int], set2: List[int], reg: int, n: int, scale: int = None) -> List[int]:
    """
    Performs modular n multiplication of two sets and adjust values based on k^h.
    h being the highest exponent in the set (set1 and set2 are of size h).

    Args:
        set1 (list): The first set of values of the form [k, k^2, ..., k^l].
        set2 (list): The second set of values of the form [k, k^2, ..., k^l].
        reg (init): Regulator value.
        n (int): The modulus for the modular multiplication.
        scale (int): The scale to use in case of encrypting plaintexts.

    Returns:
        List[int]: A new set after modular multiplication and adjustment.
    """
    assert len(set1) == len(set2)

    ln = len(set1)
    result = [0] * ln
    for i in range(ln):
        for j in range(ln):
            # Perform element-wise multiplication modulo n
            element = (set1[i] * set2[j]) % n

            pos = 0
            # Check if the set length is reached
            if i + j >= ln - 1:
                # Adjust the value by multiplying it by reg
                element = (element * reg) % n
                pos = i + j - ln + 1
            else:
                pos = i + j + 1

            # Insert the value into the result set in the right position
            result[pos] = (result[pos] + element) % n

    # Case when dealing with floats
    if scale:
        result = [(modular_inverse(scale, n) * x) % n for x in result]

    return result


def pq_modular_multiply_sets(
    set1: List[int], set2: List[int], reg: List[Tuple[int, int]], ln: int, n: int, scale: int = None
) -> List[int]:
    """
    Multiply each element of set1 by each element of set2 modulo n (where n = p * q),
    and return the modular result along with the combined exponent power.

    Parameters:
        set1 (list): First list of numbers, each corresponds to k^(i+1) mod n.
        set2 (list): Second list of numbers, each corresponds to k^(j+1) mod n.
        n (int): Modulus.

    Returns:
        list: List of tuples (value_mod_n, exponent_sum).
    """
    results = []
    for i in range(len(set1)):
        for j in range(len(set2)):
            value_mod_n = (set1[i] * set2[j]) % n
            exponent_sum = (i + 1) + (j + 1)  # because i=0 means k^1
            results.append((value_mod_n, exponent_sum))
    new_results = []
    while results:
        value, exp = results.pop()
        if exp > ln:
            # Reduction of exponents > ln using regulators.
            # Ensure progress (reg_exp > 0) to avoid infinite loops.
            if all(r_exp <= 0 for _, r_exp in reg):
                new_results.append((value, exp))
                continue

            for reg_value, reg_exp in reg:
                if reg_exp > 0:
                    new_v, new_e = (value * reg_value) % n, exp - reg_exp
                    if new_e > ln:
                        results.append((new_v, new_e))
                    else:
                        new_results.append((new_v, new_e))
                else:
                    new_results.append(((value * reg_value) % n, exp))
        else:
            new_results.append((value, exp))

    # Reduction pass for terms with exponent ln (consistent with original logic)
    final_pairs = []
    for value, exp in new_results:
        if exp == ln:
            for reg_value, reg_exp in reg:
                final_pairs.append(((value * reg_value) % n, exp - reg_exp))
        else:
            final_pairs.append((value, exp))

    merged = merge_same_exp(final_pairs, n, ln)
    if scale:
        inv_scale = modular_inverse(scale, n)
        return [(inv_scale * x) % n for x in merged]
    return merged


def merge_same_exp(pairs: List[tuple], n: int, ln: int) -> List[int]:
    """
    Merge elements with the same exponent by summing their values.

    Args:
        pairs (list[tuple]): List of (value, exponent) tuples.
        n (int, optional): Modulus to apply to summed values. If None, no modulus.

    Returns:
        list[tuple]: List of merged (value, exponent) tuples.
    """
    merged = {}

    for value, exp in pairs:
        if exp in merged:
            merged[exp] += value
        else:
            merged[exp] = value

    # Merge coefficients by exponent and return a fixed-length list for powers 1..ln.
    # Decryption expects elements for all positions in range(ln).
    return [merged.get(exp, 0) % n for exp in range(1, ln + 1)]


def encode_real(x: float, scale: int) -> tuple[int, float]:
    """Scales a real number to an integer for encryption."""
    return int(round(x * scale)), scale


def decode_real(x_int: int, scale: int, n: int = None) -> float:
    """Scales back integer to real after decryption. Handles centered lifting if n is provided."""
    if n is not None and x_int > n // 2:
        x_int -= n

    return x_int / scale
