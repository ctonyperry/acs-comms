"""pyttsx3 Text-to-Speech service implementation."""

import asyncio
import contextlib
import hashlib
import logging
from pathlib import Path

from ..audio.utils import ensure_16k_mono_wav
from ..models.schemas import VoiceInfo
from .tts_base import BaseTTSService

logger = logging.getLogger(__name__)

# Optional pyttsx3 imports
try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False


class Pyttsx3TTSService(BaseTTSService):
    """pyttsx3-based Text-to-Speech service."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or "tts_cache")
        self.cache_dir.mkdir(exist_ok=True)

    @property
    def is_available(self) -> bool:
        """Check if pyttsx3 TTS is available."""
        return TTS_AVAILABLE

    async def synthesize(self, text: str, voice_id: str | None = None, rate: int = 180) -> Path:
        """Synthesize text to speech and return path to normalized WAV file.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use (optional)
            rate: Speech rate (default 180)

        Returns:
            Path to synthesized and normalized WAV file

        Raises:
            RuntimeError: If pyttsx3 is not available
            ValueError: If text is empty
        """
        if not self.is_available:
            raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")

        if not text.strip():
            raise ValueError("Empty text")

        # Generate cache key
        cache_key = hashlib.sha1(f"{voice_id or ''}|{rate}|{text}".encode()).hexdigest()[:16]
        raw_wav = self.cache_dir / f"{cache_key}_raw.wav"

        logger.info(f"TTS: synthesizing text -> {raw_wav}")

        # Synthesize using pyttsx3 in thread
        await asyncio.to_thread(self._synthesize_sync, text, voice_id, rate, raw_wav)

        # Normalize to 16kHz mono
        normalized_path = ensure_16k_mono_wav(str(raw_wav))
        logger.info(f"TTS: normalized -> {normalized_path}")

        return Path(normalized_path)

    def _synthesize_sync(
        self, text: str, voice_id: str | None, rate: int, output_path: Path
    ) -> None:
        """Synchronous synthesis in worker thread."""
        try:
            engine = pyttsx3.init()

            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", int(rate))

            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

            logger.info("TTS: synthesis complete")

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
        finally:
            with contextlib.suppress(Exception):
                engine.stop()

    async def list_voices(self) -> list[VoiceInfo]:
        """List available voices using COM-safe worker thread.

        Returns:
            List of available voice information

        Raises:
            RuntimeError: If pyttsx3 is not available
        """
        if not self.is_available:
            raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")

        voices_data = await asyncio.to_thread(self._list_voices_sync)
        
        # Convert raw voice data to VoiceInfo objects, handling list languages
        voice_list = []
        for voice_data in voices_data:
            # Handle languages field - convert list to string if needed
            lang = voice_data.get("lang")
            if isinstance(lang, list) and lang:
                lang = lang[0]  # Take first language
            elif not isinstance(lang, str):
                lang = None
                
            voice_info = VoiceInfo(
                id=voice_data["id"],
                name=voice_data["name"],
                lang=lang
            )
            voice_list.append(voice_info)
            
        return voice_list

    def _list_voices_sync(self) -> list[dict]:
        """Synchronous voice listing with COM initialization."""
        pythoncom = None

        # Try to initialize COM for Windows
        try:
            import pythoncom  # from pywin32

            pythoncom.CoInitialize()
        except ImportError:
            pythoncom = None
        except Exception as e:
            logger.warning(f"COM initialization failed: {e}")
            pythoncom = None

        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")

            voice_list = []
            for voice in voices:
                voice_info = {
                    "id": voice.id,
                    "name": getattr(voice, "name", ""),
                    "lang": getattr(voice, "languages", None),
                }
                voice_list.append(voice_info)

            logger.info(f"Found {len(voice_list)} TTS voices")
            return voice_list

        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            raise
        finally:
            # Cleanup
            with contextlib.suppress(Exception):
                engine.stop()

            if pythoncom:
                with contextlib.suppress(Exception):
                    pythoncom.CoUninitialize()
