"""Media streaming service for microphone and phone audio."""

import asyncio
import base64
import contextlib
import datetime
import json
import logging
import wave

import sounddevice as sd
from fastapi import WebSocket

from ..audio.constants import CHANNELS, FRAME_SAMPLES, SAMPLE_RATE, TTS_QUEUE_SIZE
from ..audio.utils import stream_wav_over_ws
from ..models.state import CallState
from .stt_base import STTService
from .tts_base import TTSService

logger = logging.getLogger(__name__)


class MediaStreamer:
    """Manages bidirectional audio streaming between microphone and phone."""

    def __init__(self, stt_service: STTService, tts_service: TTSService):
        """Initialize media streamer.
        
        Args:
            stt_service: Speech-to-Text service
            tts_service: Text-to-Speech service
        """
        self.stt_service = stt_service
        self.tts_service = tts_service

        # Audio streaming state
        self._mic_stream: sd.InputStream | None = None
        self._tx_queue: asyncio.Queue | None = None
        self._tx_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None
        self._transcript_queue: asyncio.Queue | None = None

        # Recording
        self._wav_writer: wave.Wave_write | None = None
        self._recording_path: str | None = None

    async def start_streaming(self, websocket: WebSocket, call_state: CallState) -> None:
        """Start bidirectional audio streaming.
        
        Args:
            websocket: WebSocket connection for audio
            call_state: Current call state
        """
        if self._mic_stream is not None:
            logger.warning("Media streaming already active")
            return

        logger.info("Starting media streaming")

        # Start recording
        await self._start_recording()

        # Start STT if available
        if self.stt_service.is_available:
            self._transcript_queue = asyncio.Queue(maxsize=TTS_QUEUE_SIZE)
            await self.stt_service.start_processing(self._transcript_queue)

            # Start TTS consumer task
            self._tts_task = asyncio.create_task(
                self._tts_consumer(websocket, call_state)
            )

        # Start microphone streaming
        await self._start_microphone_streaming(websocket, call_state)

        logger.info("Media streaming started successfully")

    async def stop_streaming(self) -> None:
        """Stop all audio streaming and cleanup resources."""
        logger.info("Stopping media streaming")

        # Stop microphone
        await self._stop_microphone_streaming()

        # Stop TTS consumer
        if self._tts_task:
            self._tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tts_task
            self._tts_task = None

        # Stop STT
        if self.stt_service.is_available:
            await self.stt_service.stop_processing()

        # Stop recording
        await self._stop_recording()

        # Clear queues
        self._transcript_queue = None

        logger.info("Media streaming stopped")

    async def process_incoming_audio(self, audio_data: bytes) -> None:
        """Process incoming audio from the phone.
        
        Args:
            audio_data: Raw audio data from WebSocket
        """
        # Record to WAV
        if self._wav_writer:
            self._wav_writer.writeframes(audio_data)

        # Send to STT if available
        if self.stt_service.is_available:
            await self.stt_service.process_audio_chunk(audio_data)

    async def play_audio_file(self, file_path: str, websocket: WebSocket, call_state: CallState) -> None:
        """Play an audio file to the phone.
        
        Args:
            file_path: Path to audio file to play
            websocket: WebSocket connection
            call_state: Current call state
        """
        logger.info(f"Playing audio file: {file_path}")

        # Mute microphone during playback
        was_muted = call_state.muted
        call_state.muted = True

        try:
            call_state.seq = await stream_wav_over_ws(
                websocket, file_path, seq_start=call_state.seq
            )
        finally:
            call_state.muted = was_muted

        logger.info("Audio file playback completed")

    async def _start_recording(self) -> None:
        """Start recording incoming audio to WAV file."""
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self._recording_path = f"caller-{timestamp}.wav"

        self._wav_writer = wave.open(self._recording_path, "wb")
        self._wav_writer.setnchannels(CHANNELS)
        self._wav_writer.setsampwidth(2)  # 16-bit
        self._wav_writer.setframerate(SAMPLE_RATE)

        logger.info(f"Recording started: {self._recording_path}")

    async def _stop_recording(self) -> None:
        """Stop recording and close WAV file."""
        if self._wav_writer:
            self._wav_writer.close()
            self._wav_writer = None
            logger.info(f"Recording saved: {self._recording_path}")

    async def _start_microphone_streaming(self, websocket: WebSocket, call_state: CallState) -> None:
        """Start streaming microphone audio to phone."""
        self._tx_queue = asyncio.Queue(maxsize=200)

        def mic_callback(indata, frames, time_info, status):
            """Microphone input callback."""
            if call_state.muted:
                return

            try:
                if not self._tx_queue.full():
                    self._tx_queue.put_nowait(indata.copy().tobytes())
            except Exception as e:
                logger.warning(f"Error in mic callback: {e}")

        try:
            self._mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=mic_callback
            )
            self._mic_stream.start()
            logger.info("Microphone streaming started")

        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self._mic_stream = None

        # Start TX sender task
        self._tx_task = asyncio.create_task(
            self._tx_sender(websocket, call_state)
        )

    async def _stop_microphone_streaming(self) -> None:
        """Stop microphone streaming."""
        # Stop TX task
        if self._tx_task:
            self._tx_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tx_task
            self._tx_task = None

        # Stop microphone
        if self._mic_stream:
            with contextlib.suppress(Exception):
                self._mic_stream.stop()
                self._mic_stream.close()
            self._mic_stream = None
            logger.info("Microphone streaming stopped")

        self._tx_queue = None

    async def _tx_sender(self, websocket: WebSocket, call_state: CallState) -> None:
        """Send microphone audio to WebSocket."""
        try:
            while True:
                if not self._tx_queue:
                    break

                audio_buffer = await self._tx_queue.get()
                b64_data = base64.b64encode(audio_buffer).decode("ascii")

                sequence_number = call_state.increment_seq()

                payload = {
                    "kind": "audioData",
                    "audioData": {
                        "data": b64_data,
                        "sequenceNumber": sequence_number
                    }
                }

                await websocket.send_text(json.dumps(payload))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TX sender error: {e}")

    async def _tts_consumer(self, websocket: WebSocket, call_state: CallState) -> None:
        """Consume STT transcripts and speak them back via TTS."""
        try:
            while True:
                if not self._transcript_queue:
                    break

                # Get transcript from STT
                transcript = await self._transcript_queue.get()
                if transcript is None:
                    break

                logger.info(f"TTS speaking transcript: {transcript!r}")

                # Mute microphone during TTS playback
                was_muted = call_state.muted
                call_state.muted = True

                try:
                    # Synthesize speech
                    wav_path = await self.tts_service.synthesize(
                        text=transcript,
                        voice_id=None,
                        rate=180
                    )

                    # Play synthesized audio
                    call_state.seq = await stream_wav_over_ws(
                        websocket, str(wav_path), seq_start=call_state.seq
                    )

                except Exception as e:
                    logger.error(f"TTS playback failed: {e}")
                finally:
                    call_state.muted = was_muted

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TTS consumer error: {e}")
