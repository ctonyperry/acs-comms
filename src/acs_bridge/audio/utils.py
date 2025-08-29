"""Audio utility functions for WAV processing and streaming."""

import os
import asyncio
import base64
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Union

from fastapi import WebSocket

from .constants import SAMPLE_RATE, FRAME_MS


def ensure_16k_mono_wav(path: Union[str, Path]) -> str:
    """Return a path to a 16 kHz mono s16le WAV. Convert with ffmpeg if needed.
    
    Args:
        path: Input WAV file path
        
    Returns:
        Path to 16 kHz mono WAV file
        
    Raises:
        RuntimeError: If file is not correct format and ffmpeg is not available
    """
    path_str = str(path)
    
    try:
        with wave.open(path_str, "rb") as w:
            ch = w.getnchannels()
            sr = w.getframerate()
            sw = w.getsampwidth()
        if ch == 1 and sr == 16000 and sw == 2:
            return path_str
    except Exception:
        pass

    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path_str} is not 16k mono s16le and ffmpeg is not installed to convert it.")

    base, _ = os.path.splitext(path_str)
    out = f"{base}_16k_mono.wav"
    print(f"Audio: normalizing WAV with ffmpeg -> {out}")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", path_str, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        check=True
    )
    return out


async def stream_wav_over_ws(ws: WebSocket, path: str, seq_start: int = 0, frame_ms: int = FRAME_MS) -> int:
    """Stream mono 16 kHz 16-bit PCM WAV to the caller with monotonic pacing.
    
    Args:
        ws: WebSocket connection
        path: Path to WAV file
        seq_start: Starting sequence number
        frame_ms: Frame duration in milliseconds
        
    Returns:
        Next sequence number
    """
    samples_per_frame = int(SAMPLE_RATE * frame_ms / 1000)
    bytes_per_frame = samples_per_frame * 2  # int16 mono

    seq = seq_start
    loop = asyncio.get_running_loop()
    next_t = loop.time()
    interval = frame_ms / 1000.0

    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2, \
            "WAV must be mono, 16 kHz, 16-bit"
        while True:
            chunk = w.readframes(samples_per_frame)
            if not chunk:
                break
            if len(chunk) < bytes_per_frame:
                chunk += b"\x00" * (bytes_per_frame - len(chunk))

            await ws.send_text(json.dumps({
                "kind": "audioData",
                "audioData": {
                    "data": base64.b64encode(chunk).decode("ascii"),
                    "sequenceNumber": seq
                }
            }))
            seq += 1
            next_t += interval
            await asyncio.sleep(max(0.0, next_t - loop.time()))
    return seq