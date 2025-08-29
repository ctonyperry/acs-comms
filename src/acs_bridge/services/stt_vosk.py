"""Vosk Speech-to-Text service implementation."""

import asyncio
import contextlib
import json
import logging
import queue
import threading

from ..audio.constants import STT_QUEUE_SIZE, STT_QUEUE_THRESHOLD
from .stt_base import BaseSTTService

logger = logging.getLogger(__name__)

# Optional Vosk imports
try:
    from vosk import KaldiRecognizer
    from vosk import Model as VoskModel
    VOSK_AVAILABLE = True
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
    VOSK_AVAILABLE = False


class VoskSTTService(BaseSTTService):
    """Vosk-based Speech-to-Text service."""

    def __init__(self, model_path: str | None, sample_rate: int = 16000):
        super().__init__()
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._model: VoskModel | None = None
        self._stt_queue: queue.Queue | None = None
        self._worker_thread: threading.Thread | None = None
        self._should_stop = threading.Event()

        # Try to load model
        if VOSK_AVAILABLE and model_path:
            try:
                self._model = VoskModel(model_path)
                logger.info(f"Vosk STT model loaded from: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load Vosk model: {e}")
                self._model = None
        else:
            if not model_path:
                logger.info("STT disabled: model path not provided")
            if not VOSK_AVAILABLE:
                logger.info("STT disabled: 'vosk' package not installed")

    @property
    def is_available(self) -> bool:
        """Check if Vosk STT is available."""
        return VOSK_AVAILABLE and self._model is not None

    async def start_processing(self, transcript_queue: asyncio.Queue[str]) -> None:
        """Start STT processing with transcript output queue."""
        if not self.is_available:
            logger.warning("Cannot start STT: Vosk not available or model not loaded")
            return

        if self._is_running:
            logger.warning("STT processing already running")
            return

        self._transcript_queue = transcript_queue
        self._stt_queue = queue.Queue(maxsize=STT_QUEUE_SIZE)
        self._should_stop.clear()

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._stt_worker,
            daemon=True
        )
        self._worker_thread.start()
        self._is_running = True
        logger.info("STT worker started")

    async def process_audio_chunk(self, audio_data: bytes) -> None:
        """Process audio chunk for speech recognition."""
        if not self._is_running or not self._stt_queue:
            return

        # Add to queue if not full (to prevent blocking)
        try:
            if self._stt_queue.qsize() < STT_QUEUE_THRESHOLD:
                self._stt_queue.put_nowait(audio_data)
        except queue.Full:
            # Drop audio if queue is full
            pass
        except Exception as e:
            logger.warning(f"Error queuing audio data: {e}")

    async def stop_processing(self) -> None:
        """Stop STT processing and cleanup."""
        if not self._is_running:
            return

        logger.info("Stopping STT processing")
        self._should_stop.set()

        # Signal worker to stop
        if self._stt_queue:
            with contextlib.suppress(Exception):
                self._stt_queue.put_nowait(None)

        # Wait for worker thread
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            if self._worker_thread.is_alive():
                logger.warning("STT worker thread did not stop gracefully")

        self._is_running = False
        self._transcript_queue = None
        self._stt_queue = None
        self._worker_thread = None
        logger.info("STT processing stopped")

    def _stt_worker(self) -> None:
        """Worker thread for STT processing."""
        if not self._model or not self._stt_queue:
            return

        try:
            rec = KaldiRecognizer(self._model, self.sample_rate)
            rec.SetWords(True)

            logger.info("STT worker thread started")

            while not self._should_stop.is_set():
                try:
                    # Get audio data with timeout
                    data = self._stt_queue.get(timeout=1.0)
                    if data is None:
                        break

                    # Process with Vosk
                    if rec.AcceptWaveform(data):
                        # Final result
                        result = json.loads(rec.Result())
                        text = (result.get("text") or "").strip()
                        if text:
                            logger.info(f"STT final: {text}")
                            self._emit_transcript(text)
                    else:
                        # Partial result
                        partial_result = json.loads(rec.PartialResult())
                        partial_text = (partial_result.get("partial") or "").strip()
                        if partial_text:
                            logger.debug(f"STT partial: {partial_text}")

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in STT worker: {e}")
                    break

        except Exception as e:
            logger.error(f"STT worker thread error: {e}")
        finally:
            logger.info("STT worker thread finished")

    def _emit_transcript(self, text: str) -> None:
        """Emit transcript to output queue."""
        if self._transcript_queue:
            try:
                self._transcript_queue.put_nowait(text)
            except asyncio.QueueFull:
                logger.warning("Transcript queue full, dropping transcript")
            except Exception as e:
                logger.error(f"Error emitting transcript: {e}")
