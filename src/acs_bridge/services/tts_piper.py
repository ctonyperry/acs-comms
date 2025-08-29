"""Piper Text-to-Speech service implementation.

This module provides a TTS service using the Piper neural TTS engine,
which offers high-quality, fast speech synthesis. It supports configurable
voice parameters and falls back to pyttsx3 if Piper is not available.
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from ..audio.textnorm import normalize, preprocess_for_tts
from ..audio.utils import ensure_16k_mono_wav, crossfade_wav_files, get_wav_duration
from ..models.schemas import VoiceInfo
from .tts_base import BaseTTSService

logger = logging.getLogger(__name__)


def check_piper_available() -> bool:
    """Check if Piper TTS is available on the system."""
    return shutil.which("piper") is not None


class PiperTTSService(BaseTTSService):
    """Piper-based Text-to-Speech service with high-quality neural synthesis."""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 voice_path: Optional[str] = None,
                 length_scale: float = 1.08,
                 noise_scale: float = 0.65, 
                 noise_w: float = 0.80,
                 sentence_silence: float = 0.25):
        """Initialize Piper TTS service.
        
        Args:
            cache_dir: Directory for caching synthesized audio
            voice_path: Path to Piper voice model (.onnx file)
            length_scale: Speech speed multiplier (1.0 = normal, >1.0 = slower)
            noise_scale: Variability in speech (0.0-1.0, higher = more variable)
            noise_w: Variance in speech timing (0.0-1.0)
            sentence_silence: Pause between sentences in seconds
        """
        self.cache_dir = Path(cache_dir or "tts_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Piper configuration
        self.voice_path = voice_path or os.getenv("PIPER_VOICE_PATH", "./voices/en-us-high.onnx")
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence
        
        # Check for Piper availability
        self._piper_available = check_piper_available()
        if self._piper_available and self.voice_path:
            if not os.path.isfile(self.voice_path):
                logger.warning(f"Piper voice file not found: {self.voice_path}")
                self._piper_available = False
    
    @property
    def is_available(self) -> bool:
        """Check if Piper TTS is available."""
        return self._piper_available and bool(self.voice_path) and os.path.isfile(self.voice_path)
    
    async def synthesize(self, text: str, voice_id: Optional[str] = None, rate: int = 180) -> Path:
        """Synthesize text to speech using Piper with sentence-level processing.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID (ignored for Piper, uses configured voice)
            rate: Speech rate (ignored for Piper, uses length_scale)
            
        Returns:
            Path to synthesized and normalized WAV file
            
        Raises:
            RuntimeError: If Piper is not available
            ValueError: If text is empty
        """
        if not self.is_available:
            raise RuntimeError(
                f"Piper TTS not available. "
                f"Check that 'piper' is installed and voice file exists: {self.voice_path}"
            )
            
        if not text.strip():
            raise ValueError("Empty text")
            
        # Generate cache key based on text and settings
        cache_key = hashlib.sha1(
            f"{text}|{self.voice_path}|{self.length_scale}|{self.noise_scale}|"
            f"{self.noise_w}|{self.sentence_silence}".encode("utf-8")
        ).hexdigest()[:16]
        
        cached_wav = self.cache_dir / f"{cache_key}_piper.wav"
        
        # Return cached result if available
        if cached_wav.exists():
            logger.info(f"Using cached Piper TTS: {cached_wav}")
            return cached_wav
        
        logger.info(f"Synthesizing with Piper: {text[:50]}...")
        
        # Preprocess and normalize text into sentences
        processed_text = preprocess_for_tts(text)
        sentences = normalize(processed_text)
        
        if not sentences:
            raise ValueError("No sentences found after text normalization")
        
        try:
            # Synthesize each sentence separately
            sentence_wavs = []
            temp_files = []
            
            for i, sentence in enumerate(sentences):
                temp_wav = tempfile.mktemp(suffix=f"_sentence_{i}.wav")
                temp_files.append(temp_wav)
                
                # Run Piper synthesis for this sentence
                await self._synthesize_sentence(sentence, temp_wav)
                
                # Normalize to 16kHz mono
                normalized_wav = ensure_16k_mono_wav(temp_wav)
                sentence_wavs.append(normalized_wav)
            
            # Crossfade sentences together if multiple sentences
            if len(sentence_wavs) == 1:
                final_wav = sentence_wavs[0]
                # Copy to cache location
                shutil.copy2(final_wav, str(cached_wav))
            else:
                # Crossfade with 30-60ms overlap for smooth transitions
                crossfade_ms = min(60, max(30, int(self.sentence_silence * 100)))
                final_wav = crossfade_wav_files(
                    sentence_wavs, 
                    crossfade_ms=crossfade_ms, 
                    output_path=str(cached_wav)
                )
            
            logger.info(f"Piper synthesis complete: {final_wav}")
            return Path(final_wav)
            
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            raise
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                with contextlib.suppress(Exception):
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
            
            # Clean up intermediate sentence files (except final result)
            for wav_file in sentence_wavs:
                if wav_file != final_wav and os.path.exists(wav_file):
                    with contextlib.suppress(Exception):
                        os.unlink(wav_file)
    
    async def _synthesize_sentence(self, sentence: str, output_path: str) -> None:
        """Synthesize a single sentence using Piper CLI.
        
        Args:
            sentence: Sentence text to synthesize
            output_path: Output WAV file path
        """
        # Build Piper command
        cmd = [
            "piper",
            "--model", self.voice_path,
            "--output_file", output_path,
            "--length_scale", str(self.length_scale),
            "--noise_scale", str(self.noise_scale),
            "--noise_w", str(self.noise_w),
        ]
        
        # Add sentence silence if not the default
        if self.sentence_silence != 0.25:
            cmd.extend(["--sentence_silence", str(self.sentence_silence)])
        
        logger.debug(f"Running Piper: {' '.join(cmd)}")
        
        # Run Piper synthesis
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=sentence.encode("utf-8"))
            
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                raise RuntimeError(f"Piper process failed (code {process.returncode}): {error_msg}")
                
            if not os.path.exists(output_path):
                raise RuntimeError(f"Piper did not create output file: {output_path}")
                
        except Exception as e:
            logger.error(f"Piper subprocess error: {e}")
            raise
    
    async def list_voices(self) -> List[VoiceInfo]:
        """List available Piper voices.
        
        For Piper, we return information about the configured voice model.
        
        Returns:
            List with single VoiceInfo for the configured voice
        """
        if not self.is_available:
            return []
        
        # Extract voice name from path
        voice_name = Path(self.voice_path).stem if self.voice_path else "piper-default"
        
        return [
            VoiceInfo(
                id="piper-voice",
                name=f"Piper Voice ({voice_name})",
                lang="en-US"  # Could be made configurable
            )
        ]