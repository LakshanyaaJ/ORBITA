"""
ORBITA WebSocket Manager
========================
Manages broadcast to multiple connected dashboard clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Broadcast manager for multiple WebSocket clients."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket client connected. Total: %d", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send JSON data to all connected clients."""
        if not self._connections:
            return
        message = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)
