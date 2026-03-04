"""Content-hashing utilities."""

import hashlib


def _hash(*parts) -> str:
    """Return the SHA-256 hex digest of pipe-joined string representations."""
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode()).hexdigest()
