"""Audio utility functions for WAV processing and streaming."""

import os
import asyncio
import base64
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Union, List, Optional

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


def crossfade_wav_files(wav_files: List[Union[str, Path]], 
                       crossfade_ms: int = 50,
                       output_path: Optional[Union[str, Path]] = None) -> str:
    """Crossfade multiple WAV files together with smooth transitions.
    
    Args:
        wav_files: List of WAV file paths to concatenate
        crossfade_ms: Crossfade duration in milliseconds (30-60ms recommended)
        output_path: Output file path (if None, generates temp file)
        
    Returns:
        Path to the concatenated WAV file
        
    Raises:
        ValueError: If less than 2 files provided or files have different formats
        RuntimeError: If ffmpeg is not available
    """
    if len(wav_files) < 2:
        # If only one file, just return it (or copy if output_path specified)
        if len(wav_files) == 1:
            if output_path:
                import shutil
                shutil.copy2(str(wav_files[0]), str(output_path))
                return str(output_path)
            return str(wav_files[0])
        raise ValueError("At least one WAV file required")
    
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for crossfading but is not installed")
    
    # Generate output path if not provided
    if output_path is None:
        import tempfile
        output_path = tempfile.mktemp(suffix="_crossfaded.wav")
    else:
        output_path = str(output_path)
    
    # Validate all input files have same format (16kHz mono)
    for wav_file in wav_files:
        try:
            with wave.open(str(wav_file), "rb") as w:
                if w.getnchannels() != 1 or w.getframerate() != 16000 or w.getsampwidth() != 2:
                    raise ValueError(f"All WAV files must be 16kHz mono s16le format. "
                                   f"File {wav_file} is {w.getframerate()}Hz, "
                                   f"{w.getnchannels()} channels, {w.getsampwidth()*8}-bit")
        except Exception as e:
            raise ValueError(f"Error reading {wav_file}: {e}")
    
    # Convert crossfade time to seconds for ffmpeg
    crossfade_sec = crossfade_ms / 1000.0
    
    # Build ffmpeg filter complex for crossfading
    # For each pair of files, we create a crossfade filter
    filter_parts = []
    input_args = []
    
    # Add all input files
    for i, wav_file in enumerate(wav_files):
        input_args.extend(["-i", str(wav_file)])
    
    # Build the filter chain for crossfading
    if len(wav_files) == 2:
        # Simple case: crossfade two files
        filter_complex = f"[0][1]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[out]"
    else:
        # Multiple files: chain crossfades
        filter_complex = ""
        for i in range(len(wav_files) - 1):
            if i == 0:
                # First crossfade
                filter_complex += f"[{i}][{i+1}]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[cf{i}];"
            elif i == len(wav_files) - 2:
                # Last crossfade
                filter_complex += f"[cf{i-1}][{i+1}]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[out]"
            else:
                # Middle crossfades
                filter_complex += f"[cf{i-1}][{i+1}]acrossfade=d={crossfade_sec}:c1=tri:c2=tri[cf{i}];"
    
    # Run ffmpeg command
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error"
    ] + input_args + [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        output_path
    ]
    
    print(f"Audio: crossfading {len(wav_files)} files -> {output_path}")
    subprocess.run(cmd, check=True)
    
    return output_path


def get_wav_duration(wav_path: Union[str, Path]) -> float:
    """Get duration of a WAV file in seconds.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Duration in seconds
    """
    with wave.open(str(wav_path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / rate