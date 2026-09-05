"""ORBITA Backend Service Layer.

Exposes the FastAPI application and server entrypoint.
"""
from orbita.backend.api import create_app

__all__ = ["create_app"]
