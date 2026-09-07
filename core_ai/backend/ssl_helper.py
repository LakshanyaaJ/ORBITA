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

from core_ai.video.phone_stream_receiver import get_lan_ip, get_all_lan_ips

logger = logging.getLogger(__name__)


def _cert_covers_ips(cert_path: Path, required_ips: list[str]) -> bool:
    """Check if the existing certificate has all required IPs in SubjectAlternativeName."""
    try:
        cert_data = cert_path.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_data)
        san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        san = san_ext.value
        existing_ips = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        for req_ip in required_ips:
            if req_ip not in existing_ips:
                return False
        return True
    except Exception:
        return False


def ensure_ssl_certificates(
    config_dir: Path = Path("config"),
    cert_filename: str = "cert.pem",
    key_filename: str = "key.pem",
) -> Tuple[str, str]:
    """
    Ensure valid self-signed SSL cert and private key exist for localhost and all LAN IPs.
    If the current IP changes, automatically regenerates the certificate so mobile browsers match.
    Returns (cert_path_str, key_path_str).
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    cert_path = config_dir / cert_filename
    key_path = config_dir / key_filename

    lan_ip = get_lan_ip()
    all_interfaces = get_all_lan_ips()
    all_ips = [iface["ip"] for iface in all_interfaces if iface["ip"] != "127.0.0.1"]
    if lan_ip and lan_ip != "127.0.0.1" and lan_ip not in all_ips:
        all_ips.append(lan_ip)

    # If both files exist and cover all current IPs, reuse them
    if cert_path.exists() and key_path.exists() and _cert_covers_ips(cert_path, all_ips):
        return str(cert_path), str(key_path)

    logger.info("Generating/Updating self-signed SSL certificate for Secure Context mobile streaming...")

    # Generate 2048-bit RSA private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ORBITA Jetson System"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"ORBITA-CAM-{lan_ip}"),
    ])

    # Build SAN entries (localhost + 127.0.0.1 + all active LAN IPs)
    san_list: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]

    for ip_str in all_ips:
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(ip_str)))
        except Exception as e:
            logger.warning("Could not add IP %s to SAN: %s", ip_str, e)

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

    logger.info("SSL Certificate successfully updated for IPs %s: %s, %s", all_ips, cert_path, key_path)
    return str(cert_path), str(key_path)
