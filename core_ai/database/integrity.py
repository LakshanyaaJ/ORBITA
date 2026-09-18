"""
ORBITA Data Integrity Module
============================
Provides SHA-256 hashing and verification for structured experiment payloads
and telemetry records, guaranteeing end-to-end data integrity between the
edge compute device and the ground system.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Union


def canonical_json(data: Union[Dict[str, Any], list, Any]) -> str:
    """
    Serialize data to deterministic, canonical JSON format:
    - Keys sorted alphabetically
    - Compact delimiters (no trailing whitespace)
    - UTF-8 encoding
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calculate_checksum(data: Union[Dict[str, Any], list, str, bytes]) -> str:
    """
    Compute standard SHA-256 hex digest for any data structure.
    If a dict or list is supplied, it is first canonicalized.
    """
    if isinstance(data, (dict, list)):
        encoded = canonical_json(data).encode("utf-8")
    elif isinstance(data, str):
        encoded = data.encode("utf-8")
    elif isinstance(data, bytes):
        encoded = data
    else:
        encoded = str(data).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def verify_payload_checksum(payload: Dict[str, Any], expected_checksum: str) -> bool:
    """
    Verify whether the SHA-256 digest of payload matches the expected checksum.
    If the payload contains a nested 'checksum' field, it is excluded prior to
    computing the verification digest.
    """
    if not expected_checksum:
        return False

    # Copy and remove self-referential checksum if present
    verify_data = dict(payload)
    if "checksum" in verify_data:
        del verify_data["checksum"]

    calculated = calculate_checksum(verify_data)
    return calculated.lower() == expected_checksum.lower()
