"""
ORBITA Asynchronous Priority Voice Manager
===========================================
High-reliability, non-blocking voice alert manager for zero-cloud execution.

Architectural Workflow:
  FSM / Reasoning Event
           ↓
     VoiceManager.announce(text, priority)
           ↓
     Priority Queue (CRITICAL > WARNING > GUIDANCE > SUCCESS > INFO)
           ↓
     Deduplication & Cooldown Filter
           ↓
     Single Dedicated TTS Worker
           ↓
     Offline Audio Output (SAPI5 / pyttsx3)

Key Guarantees:
  - Non-blocking: Inference and video loops enqueue messages in <0.1ms and return immediately.
  - Priority Capping: Critical errors (wrong object/action) interrupt or preempt queued guidance.
  - Deduplication: Identical announcements within `cooldown_seconds` (default 3.0s) are suppressed.
  - Thread Safety: Single-threaded TTS execution prevents speech overlap and COM deadlocks.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class VoicePriority(IntEnum):
    """Voice announcement priority levels (lower integer = higher priority)."""
    CRITICAL = 1   # Wrong object, wrong action, emergency hazard
    WARNING = 2    # Step skipped, out of sequence
    GUIDANCE = 3   # Step prompt, procedural instruction
    SUCCESS = 4    # Step confirmed, action completed
    INFO = 5       # General system telemetry, experiment complete


@dataclass(order=True)
class VoiceItem:
    """Queued voice announcement item ordered by priority and timestamp."""
    priority: int
    timestamp: float
    text: str = field(compare=False)
    voice_type: str = field(compare=False, default="GUIDANCE")
    step_id: int = field(compare=False, default=0)
    event_id: str = field(compare=False, default="")
    callback: Optional[Callable[[str], None]] = field(compare=False, default=None)


class VoiceManager:
    """
    Central Asynchronous Priority Voice Engine for ORBITA.
    """

    def __init__(self, tts_engine: Any = None, cooldown_seconds: float = 3.0):
        self.tts_engine = tts_engine
        self.cooldown_seconds: float = cooldown_seconds

        self._queue: queue.PriorityQueue[VoiceItem] = queue.PriorityQueue(maxsize=20)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # State & Deduplication Tracking
        self._last_spoken_time: Dict[str, float] = {}  # text -> timestamp
        self._last_spoken_text: str = ""
        self._last_event_id: str = ""
        self._is_speaking: bool = False
        self._speaking_until: float = 0.0

        # Diagnostics metrics
        self._total_announced: int = 0
        self._total_spoken: int = 0
        self._total_suppressed: int = 0

        self._start_worker()

    def set_tts_engine(self, tts_engine: Any) -> None:
        """Bind or update the underlying offline TTS engine."""
        with self._lock:
            self.tts_engine = tts_engine

    def _start_worker(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="orbita-voice-manager"
        )
        self._thread.start()

    def announce(
        self,
        text: str,
        priority: VoicePriority = VoicePriority.GUIDANCE,
        voice_type: str = "GUIDANCE",
        step_id: int = 0,
        event_id: str = "",
        callback: Optional[Callable[[str], None]] = None,
        force: bool = False,
    ) -> bool:
        """
        Non-blocking voice event dispatch.
        Pushes voice items into priority queue and returns immediately.
        """
        if not text:
            return False

        clean_text = str(text).strip()
        if not clean_text:
            return False

        now = time.time()
        self._total_announced += 1

        # Deduplication & Cooldown Filter
        with self._lock:
            last_ts = self._last_spoken_time.get(clean_text, 0.0)
            elapsed = now - last_ts

            if not force and priority > VoicePriority.CRITICAL:
                if elapsed < self.cooldown_seconds:
                    self._total_suppressed += 1
                    logger.debug("VoiceManager: Suppressed duplicate '%s' (%.1fs < %.1fs cooldown)",
                                 clean_text, elapsed, self.cooldown_seconds)
                    return False

            item = VoiceItem(
                priority=int(priority),
                timestamp=now,
                text=clean_text,
                voice_type=voice_type,
                step_id=step_id,
                event_id=event_id,
                callback=callback,
            )

            # High-priority preempts lower-priority items currently in queue
            if priority <= VoicePriority.WARNING:
                self._purge_lower_priority_items(max_priority=int(priority))

            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                self._drop_lowest_priority_item()
                try:
                    self._queue.put_nowait(item)
                    return True
                except queue.Full:
                    return False

    def _purge_lower_priority_items(self, max_priority: int) -> None:
        """Remove lower-priority items (larger integer priority) from queue."""
        temp: List[VoiceItem] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item.priority <= max_priority:
                    temp.append(item)
                else:
                    self._total_suppressed += 1
            except queue.Empty:
                break
        for item in temp:
            self._queue.put_nowait(item)

    def _drop_lowest_priority_item(self) -> None:
        """Drop the lowest-priority item from queue if full."""
        temp: List[VoiceItem] = []
        while not self._queue.empty():
            try:
                temp.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if temp:
            temp.sort()
            dropped = temp.pop()  # largest integer priority = lowest priority
            self._total_suppressed += 1
            logger.debug("VoiceManager: Dropped lowest priority item '%s'", dropped.text)
            for item in temp:
                self._queue.put_nowait(item)

    def clear(self) -> None:
        """Clear all pending queued messages."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def stop(self) -> None:
        """Stop voice worker thread cleanly."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @property
    def is_speaking(self) -> bool:
        return time.time() < self._speaking_until

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def get_telemetry(self) -> Dict[str, Any]:
        """Return diagnostic metrics."""
        return {
            "queue_size": self.queue_size,
            "is_speaking": self.is_speaking,
            "last_spoken_text": self._last_spoken_text,
            "last_event_id": self._last_event_id,
            "total_announced": self._total_announced,
            "total_spoken": self._total_spoken,
            "total_suppressed": self._total_suppressed,
        }

    def _worker_loop(self) -> None:
        """Single background worker reading priority queue and speaking sequentially."""
        while not self._stop_event.is_set():
            try:
                item: VoiceItem = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.time()

            # Record spoken state
            with self._lock:
                self._last_spoken_time[item.text] = now
                self._last_spoken_text = item.text
                self._last_event_id = item.event_id
                words = len(item.text.split())
                est_dur = max(1.5, words * 0.38)
                self._speaking_until = now + est_dur
                self._total_spoken += 1

            logger.info("VoiceManager Speaking [P%d, %s]: \"%s\"", item.priority, item.voice_type, item.text)

            # Trigger optional callback (e.g. UI log or TTS engine)
            if item.callback:
                try:
                    item.callback(item.text)
                except Exception as cb_err:
                    logger.warning("VoiceManager callback error: %s", cb_err)

            if self.tts_engine and hasattr(self.tts_engine, "speak"):
                try:
                    self.tts_engine.speak(item.text)
                except Exception as tts_err:
                    logger.warning("VoiceManager TTS speak error: %s", tts_err)

            self._queue.task_done()
