"""ORBITA Server Entrypoint Facade.

Exposes the FastAPI application and server entrypoint.
"""
from core_ai.backend.api import create_app

__all__ = ["create_app"]
