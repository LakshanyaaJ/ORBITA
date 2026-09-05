"""
ORBITA Offline TTS
==================
Non-blocking text-to-speech using pyttsx3 (offline, no cloud).

Architecture:
  Caller thread      Background TTS thread
       |                     |
       |-- speak(text) ----→ Queue
       |                     |
       |                     ← pyttsx3.runAndWait()
       |                     |
       |                     (next message)

Graceful degradation:
  If pyttsx3 is not installed or fails to init, TTS is silently disabled.
  The system continues running in "voice off" mode.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_SENTINEL = None  # Signals background thread to stop


class TTSEngine:
    """
    Offline Text-to-Speech engine with async speech queue.

    Usage:
        tts = TTSEngine(config)
        tts.speak("Step one: Open the main box.")
        tts.stop()
    """

    def __init__(self, config):
        self.enabled = config.enabled
        self._engine = None
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=10)
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._init_engine(config)

    def speak(self, text: str) -> None:
        """Queue text for speech. Non-blocking."""
        if not self.enabled or self._engine is None:
            logger.debug("TTS (disabled): %s", text)
            return
        # Clear pending messages if queue is full (don't block the AI pipeline)
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Drop oldest, add new
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                pass

    def stop(self) -> None:
        """Gracefully stop the TTS background thread."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(_SENTINEL)
            self._thread.join(timeout=2.0)

    def is_available(self) -> bool:
        return self.enabled and self._engine is not None

    # ----------------------------------------------------------------------- #
    # Internal
    # ----------------------------------------------------------------------- #
    def _init_engine(self, config) -> None:
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            engine.setProperty("rate", config.rate)
            engine.setProperty("volume", config.volume)

            # Set specific voice if configured
            if config.voice_id:
                engine.setProperty("voice", config.voice_id)

            self._engine = engine
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="orbita-tts"
            )
            self._thread.start()
            logger.info("TTS engine initialized (rate=%d, volume=%.2f).", config.rate, config.volume)
        except ImportError:
            logger.warning("pyttsx3 not installed — TTS disabled.")
            self.enabled = False
        except Exception as exc:
            logger.warning("TTS init failed: %s — TTS disabled.", exc)
            self.enabled = False

    def _worker(self) -> None:
        """Background thread: reads from queue and speaks."""
        while True:
            text = self._queue.get()
            if text is _SENTINEL:
                break
            try:
                if self._engine and text:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS speak error: %s", exc)
