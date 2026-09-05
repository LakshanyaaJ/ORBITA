"""
ORBITA SSL Certificate Helper
=============================
Generates self-signed certificates on the fly to enable HTTPS Secure Context
for mobile browsers (Android Chrome, iOS Safari) streaming camera video over LAN.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from core_ai.video.phone_stream_receiver import get_lan_ip

logger = logging.getLogger(__name__)


def ensure_ssl_certificates(
    config_dir: Path = Path("config"),
    cert_filename: str = "cert.pem",
    key_filename: str = "key.pem",
) -> Tuple[str, str]:
    """
    Ensure valid self-signed SSL cert and private key exist for localhost and LAN IP.
    Returns (cert_path_str, key_path_str).
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    cert_path = config_dir / cert_filename
    key_path = config_dir / key_filename

    # If both files exist, return them
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    logger.info("Generating self-signed SSL certificate for Secure Context mobile streaming...")

    lan_ip = get_lan_ip()

    # Generate 2048-bit RSA private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ORBITA Jetson System"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"ORBITA-CAM-{lan_ip}"),
    ])

    # Build SAN entries
    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        if lan_ip and lan_ip != "127.0.0.1":
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
    except Exception as e:
        logger.warning("Could not add LAN IP to SAN: %s", e)

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=730))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )

    # Write key
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write cert
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info("SSL Certificate successfully created: %s, %s", cert_path, key_path)
    return str(cert_path), str(key_path)
