"""
ORBITA Offline TTS
==================
Non-blocking text-to-speech with dedicated worker thread.
On Windows: Uses native SAPI5 COM with STA initialization on the worker thread.
Fallback: pyttsx3 initialized on the worker thread.
Default voice: Female (Microsoft Zira / Hazel or system female voice).
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Signals background thread to stop


class TTSEngine:
    """
    Offline Text-to-Speech engine with async speech queue.

    Usage:
        tts = TTSEngine(config)
        tts.speak("Step one: Open the main box.")
        tts.stop()
    """

    def __init__(self, config):
        self.enabled: bool = getattr(config, "enabled", True)
        self.rate: int = getattr(config, "rate", 175)
        self.volume: float = getattr(config, "volume", 0.95)
        self.voice_id: Optional[str] = getattr(config, "voice_id", "female")
        self._engine_type: str = "none"
        self._voice_name: str = "Default"
        self._voice_gender: str = "Female"
        self._is_ready: bool = False
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=10)
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="orbita-tts"
            )
            self._thread.start()

    def speak(self, text: str) -> None:
        """Queue text for speech. Non-blocking."""
        if not self.enabled or not text:
            return

        text = str(text).strip()
        if not text:
            return

        # Drop oldest if queue is full to prevent lagging behind real-time events
        try:
            self._queue.put_nowait(text)
        except queue.Full:
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
            try:
                self._queue.put(_SENTINEL)
                self._thread.join(timeout=1.5)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all pending messages in queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def is_available(self, timeout: float = 0.5) -> bool:
        if not self.enabled:
            return False
        if self._is_ready:
            return True
        if self._thread is not None and self._thread.is_alive():
            import time
            end_t = time.time() + timeout
            while time.time() < end_t and not self._is_ready:
                time.sleep(0.02)
        return self.enabled and self._is_ready

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self._is_ready,
            "engine": self._engine_type,
            "voice_name": self._voice_name,
            "voice_gender": self._voice_gender,
            "rate": self.rate,
            "volume": self.volume,
            "voice_id": self.voice_id,
            "queue_size": self._queue.qsize(),
        }

    # ----------------------------------------------------------------------- #
    # Internal Worker
    # ----------------------------------------------------------------------- #
    def _worker(self) -> None:
        """Dedicated background thread: initializes TTS engine in thread context and speaks."""
        is_windows = platform.system() == "Windows"
        sapi_voice = None
        pyttsx_engine = None
        com_initialized = False

        if is_windows:
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                com_initialized = True
                sapi = win32com.client.Dispatch("SAPI.SpVoice")
                # SAPI rate range is -10 to +10 (where 0 is ~200 wpm)
                sapi_rate = max(-10, min(10, int((self.rate - 200) / 10)))
                sapi.Rate = sapi_rate
                sapi.Volume = max(0, min(100, int(self.volume * 100)))

                # Voice selection: prioritize female voice (Zira or Hazel)
                want_female = not self.voice_id or self.voice_id.lower() in ("female", "woman", "zira", "hazel")
                voice_chosen = None

                if want_female:
                    zira_tokens = sapi.GetVoices("Name=Microsoft Zira Desktop")
                    if zira_tokens.Count > 0:
                        voice_chosen = zira_tokens.Item(0)
                    else:
                        female_tokens = sapi.GetVoices("Gender=Female")
                        if female_tokens.Count > 0:
                            voice_chosen = female_tokens.Item(0)
                elif self.voice_id:
                    tokens = sapi.GetVoices()
                    for i in range(tokens.Count):
                        tok = tokens.Item(i)
                        if self.voice_id.lower() in tok.GetDescription().lower():
                            voice_chosen = tok
                            break

                if voice_chosen is not None:
                    sapi.Voice = voice_chosen
                    self._voice_name = voice_chosen.GetDescription()
                else:
                    self._voice_name = sapi.Voice.GetDescription()

                is_female = any(k in self._voice_name.lower() for k in ("zira", "hazel", "female", "woman"))
                self._voice_gender = "Female" if is_female else "Male"

                sapi_voice = sapi
                self._engine_type = "sapi5"
                self._is_ready = True
                logger.info("Windows SAPI5 TTS engine initialized (voice=%s, gender=%s).", self._voice_name, self._voice_gender)
            except Exception as exc:
                logger.warning("Windows SAPI5 init failed (%s), falling back to pyttsx3", exc)

        if not self._is_ready:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)

                want_female = not self.voice_id or self.voice_id.lower() in ("female", "woman", "zira", "hazel")
                voices = engine.getProperty("voices")
                voice_chosen = None

                if want_female:
                    for v in voices:
                        name_l = v.name.lower()
                        if "zira" in name_l or "hazel" in name_l or getattr(v, "gender", "") == "female":
                            voice_chosen = v
                            break
                elif self.voice_id:
                    for v in voices:
                        if self.voice_id.lower() in v.name.lower():
                            voice_chosen = v
                            break

                if voice_chosen:
                    engine.setProperty("voice", voice_chosen.id)
                    self._voice_name = voice_chosen.name
                else:
                    self._voice_name = str(engine.getProperty("voice"))

                is_female = any(k in self._voice_name.lower() for k in ("zira", "hazel", "female", "woman"))
                self._voice_gender = "Female" if is_female else "Male"

                pyttsx_engine = engine
                self._engine_type = "pyttsx3"
                self._is_ready = True
                logger.info("pyttsx3 TTS engine initialized on worker thread (voice=%s, gender=%s).", self._voice_name, self._voice_gender)
            except Exception as exc:
                logger.warning("pyttsx3 init failed (%s) — TTS disabled.", exc)
                self.enabled = False
                self._is_ready = False

        while True:
            try:
                item = self._queue.get()
                if item is _SENTINEL:
                    break

                if not item or not self.enabled or not self._is_ready:
                    continue

                text = str(item).strip()
                if not text:
                    continue

                if sapi_voice is not None:
                    try:
                        sapi_voice.Speak(text, 0)
                    except Exception as sapi_err:
                        logger.warning("SAPI speak exception (%s), re-dispatching voice and retrying...", sapi_err)
                        try:
                            sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                            if voice_chosen is not None:
                                sapi_voice.Voice = voice_chosen
                            sapi_voice.Rate = sapi_rate
                            sapi_voice.Volume = max(0, min(100, int(self.volume * 100)))
                            sapi_voice.Speak(text, 0)
                        except Exception as retry_err:
                            logger.warning("SAPI retry failed: %s", retry_err)
                elif pyttsx_engine is not None:
                    pyttsx_engine.say(text)
                    pyttsx_engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS speak error: %s", exc)

        if com_initialized:
            try:
                sapi_voice = None
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
