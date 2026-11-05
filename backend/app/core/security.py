# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Password hashing with scrypt from the standard library.

No bcrypt/argon2 dependency on purpose: ``hashlib.scrypt`` is memory-hard, part of
CPython, and the only thing the project needs. The stored form is self-describing
(``scrypt$n$r$p$salt$hash``) so parameters can be raised later without a
migration: old hashes keep verifying with the parameters they were written with.
"""
from __future__ import annotations

import hashlib
import secrets

ALGORITHM = "scrypt"
# 2^14 * 8 * 128 bytes = 16 MiB per hash: high enough to make a GPU dictionary
# attack expensive, low enough that the API can hash on login without a stall.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32
_MAX_MEM = 64 * 1024 * 1024


def hash_password(password: str) -> str:
    """Hash a password with a fresh salt."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        dklen=KEY_BYTES, maxmem=_MAX_MEM,
    )
    return (f"{ALGORITHM}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}"
            f"${salt.hex()}${derived.hex()}")


def verify_password(password: str, stored: str) -> bool:
    """True when ``password`` matches ``stored``. Never raises on bad input.

    A malformed hash returns False rather than an exception: a corrupted row
    must lock one account, not crash the login endpoint.
    """
    try:
        algorithm, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        expected = bytes.fromhex(hash_hex)
        derived = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p),
            dklen=len(expected), maxmem=_MAX_MEM,
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(derived, expected)
