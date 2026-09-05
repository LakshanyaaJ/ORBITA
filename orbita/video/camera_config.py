"""
ORBITA Camera Configuration & Validation
========================================
Supports configuration and URL validation for:
  - Jetson / Local Cameras (device index integer or device path)
  - Phone IP Cameras (IP address, port, path, or complete URL)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class IPCameraConfig:
    """Configuration for an IP camera stream."""
    url: str
    ip: str = ""
    port: int = 8080
    path: str = "/video"
    timeout_sec: float = 4.0
    buffer_size: int = 1
    reconnect_interval_sec: float = 3.0
    max_reconnect_attempts: int = 10
    target_fps: int = 30
    width: int = 640
    height: int = 480


def get_default_ip_camera_url() -> str:
    """Read default IP camera URL from environment or return sensible default."""
    return os.getenv("IP_CAMERA_URL", "")


def validate_and_format_camera_url(
    url: Optional[str] = None,
    ip: Optional[str] = None,
    port: Optional[int | str] = None,
    path: Optional[str] = None,
) -> tuple[bool, str, str]:
    """
    Validates and formats an IP camera URL.

    Returns:
        (is_valid: bool, formatted_url: str, error_message: str)
    """
    # 1. Prefer full URL if supplied
    if url and url.strip():
        clean_url = url.strip()
        parsed = urlparse(clean_url)
        if not parsed.scheme:
            # Assume http if no scheme
            clean_url = f"http://{clean_url}"
            parsed = urlparse(clean_url)

        if parsed.scheme not in ("http", "https", "rtsp", "mjpeg"):
            return False, "", f"Unsupported stream protocol: {parsed.scheme}. Use http, https, or rtsp."

        if not parsed.netloc:
            return False, "", "Invalid stream URL: Missing host address or IP."

        return True, clean_url, ""

    # 2. Build from IP, port, path
    if not ip or not ip.strip():
        return False, "", "Please enter a Phone IP Address."

    clean_ip = ip.strip()
    
    # If the user pasted a full URL or http/https into the IP box, treat it as a full URL
    if clean_ip.startswith("http://") or clean_ip.startswith("https://") or clean_ip.startswith("rtsp://"):
        return validate_and_format_camera_url(url=clean_ip)

    # Check if a path is included in the IP string (e.g. 192.168.1.105:8080/video or 192.168.1.105/video)
    if "/" in clean_ip:
        ip_part, path_part = clean_ip.split("/", 1)
        clean_ip = ip_part
        if not path or not path.strip():
            path = "/" + path_part

    # Check if a port is included in the IP string (e.g. 192.168.1.105:8080)
    if ":" in clean_ip:
        ip_part, port_part = clean_ip.split(":", 1)
        clean_ip = ip_part
        if port is None or not str(port).strip():
            port = port_part

    # Validate IP or hostname syntax
    # Matches IPv4 or local hostname
    ip_pattern = r"^([0-9]{1,3}\.){3}[0-9]{1,3}$"
    hostname_pattern = r"^[a-zA-Z0-9.-]+$"
    if not re.match(ip_pattern, clean_ip) and not re.match(hostname_pattern, clean_ip):
        return False, "", f"Invalid IP address format: '{clean_ip}'. Ensure it is a valid local IP like 192.168.1.105."

    # Validate port
    port_int = 8080
    if port is not None and str(port).strip():
        try:
            port_int = int(port)
            if port_int < 1 or port_int > 65535:
                return False, "", "Port must be between 1 and 65535."
        except ValueError:
            return False, "", f"Invalid port: '{port}'. Must be a number."

    # Clean path
    clean_path = path.strip() if path else "/video"
    if not clean_path.startswith("/"):
        clean_path = "/" + clean_path

    built_url = f"http://{clean_ip}:{port_int}{clean_path}"
    return True, built_url, ""
