# app.py
# ACS Local Audio Bridge with offline STT (Vosk) and TTS (pyttsx3)
# Single port: /events (Event Grid) + /ws (media) + /api/* controls

import os
import asyncio
import base64
import json
import datetime
import wave
import threading
import queue
import shutil
import subprocess
import contextlib
from typing import Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, Body, HTTPException
from fastapi.responses import JSONResponse

from azure.communication.callautomation import (
    CallAutomationClient, MediaStreamingOptions,
    StreamingTransportType, MediaStreamingContentType,
    MediaStreamingAudioChannelType, AudioFormat
)

# -------- Optional offline STT (Vosk) --------
try:
    from vosk import Model as VoskModel, KaldiRecognizer  # pip install vosk
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False
    VoskModel = None  # type: ignore
    KaldiRecognizer = None  # type: ignore

# -------- Optional offline TTS (pyttsx3/SAPI) --------
try:
    import pyttsx3  # pip install pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    pyttsx3 = None  # type: ignore

# ---------- Env / constants ----------
load_dotenv()
ACS = os.environ["ACS_CONNECTION_STRING"]                          # endpoint=...;accesskey=...
PUBLIC_BASE = os.environ["PUBLIC_BASE"]                            # e.g. https://<ngrok>.ngrok-free.app
STT_MODEL_PATH = os.getenv("STT_MODEL_PATH", "").strip()           # e.g. ./models/vosk-model-small-en-us-0.15

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# ---------- Global state ----------
STATE = {
    "ws": None,                 # active call websocket
    "seq": 0,                   # sequence number for outbound audioData
    "muted": False,             # gate mic -> phone
    "call_connection_id": None  # for hangup
}

# ---------- App / client ----------
app = FastAPI()
client = CallAutomationClient.from_connection_string(ACS)

# ---------- STT init ----------
VOSK_MODEL: Optional["VoskModel"] = None
if VOSK_AVAILABLE and STT_MODEL_PATH:
    try:
        VOSK_MODEL = VoskModel(STT_MODEL_PATH)
        print(f"Vosk STT model loaded from: {STT_MODEL_PATH}")
    except Exception as e:
        print("Failed to load Vosk model:", e)
        VOSK_MODEL = None
else:
    if not STT_MODEL_PATH:
        print("STT disabled: set STT_MODEL_PATH in .env to enable Vosk.")
    if not VOSK_AVAILABLE:
        print("STT disabled: 'vosk' package not installed.")

# ---------- STT worker (push finals into out_q) ----------
def start_stt_worker(model: "VoskModel", sample_rate: int,
                     in_q: "queue.Queue[bytes]", out_q: "queue.Queue[str]"):
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)
    while True:
        data = in_q.get()
        if data is None:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            txt = (r.get("text") or "").strip()
            if txt:
                print("STT:", txt)
                with contextlib.suppress(Exception):
                    out_q.put_nowait(txt)
        else:
            pr = json.loads(rec.PartialResult())
            p = (pr.get("partial") or "").strip()
            if p:
                print("STT(partial):", p)

# ---------- Media helpers ----------
async def play_wav_to_phone(ws: WebSocket, path: str, seq_start: int = 0, frame_ms: int = FRAME_MS) -> int:
    """Stream mono 16 kHz 16-bit PCM WAV to the caller with monotonic pacing."""
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

# ---------- WAV normalization + TTS synth ----------
def ensure_16k_mono_wav(path: str) -> str:
    """Return a path to a 16 kHz mono s16le WAV. Convert with ffmpeg if needed."""
    try:
        with wave.open(path, "rb") as w:
            ch = w.getnchannels(); sr = w.getframerate(); sw = w.getsampwidth()
        if ch == 1 and sr == 16000 and sw == 2:
            return path
    except Exception:
        pass

    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path} is not 16k mono s16le and ffmpeg is not installed to convert it.")

    base, _ = os.path.splitext(path)
    out = f"{base}_16k_mono.wav"
    print(f"TTS/PLAY: normalizing WAV with ffmpeg -> {out}")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        check=True
    )
    return out

def _synthesize_tts_to_wav_sync(text: str, voice_id: Optional[str], rate: int) -> str:
    if not TTS_AVAILABLE:
        raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")
    if not text.strip():
        raise ValueError("Empty text")

    import hashlib
    h = hashlib.sha1(f"{voice_id or ''}|{rate}|{text}".encode("utf-8")).hexdigest()[:16]
    tmp_dir = os.path.join(os.getcwd(), "tts_cache")
    os.makedirs(tmp_dir, exist_ok=True)
    raw_wav = os.path.join(tmp_dir, f"{h}_raw.wav")

    print(f"TTS: synthesizing -> {raw_wav}")
    eng = pyttsx3.init()
    if voice_id:
        eng.setProperty("voice", voice_id)
    eng.setProperty("rate", int(rate))
    eng.save_to_file(text, raw_wav)
    eng.runAndWait()
    print("TTS: synthesis complete")

    out_wav = ensure_16k_mono_wav(raw_wav)
    print(f"TTS: ready -> {out_wav}")
    return out_wav

async def synthesize_tts_to_wav(text: str, voice_id: Optional[str], rate: int) -> str:
    return await asyncio.to_thread(_synthesize_tts_to_wav_sync, text, voice_id, rate)

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"ok": True, "ws_active": STATE["ws"] is not None, "muted": STATE["muted"], "seq": STATE["seq"]}

@app.post("/events")
async def events(request: Request):
    body = await request.json()

    # Event Grid handshake
    if isinstance(body, list) and body and body[0].get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
        code = body[0]["data"]["validationCode"]
        print("EventGrid validation:", code)
        return JSONResponse({"validationResponse": code})

    # ACS events
    if isinstance(body, list):
        for evt in body:
            etype = evt.get("eventType")
            data = evt.get("data", {})
            if etype == "Microsoft.Communication.IncomingCall":
                incoming_call_context = data["incomingCallContext"]

                ws_url = f"{PUBLIC_BASE.replace('https://','wss://')}/ws"
                cb_url = f"{PUBLIC_BASE}/events"
                print("DEBUG PUBLIC_BASE:", repr(PUBLIC_BASE))
                print("DEBUG WS URL:", ws_url)
                print("DEBUG CALLBACK URL:", cb_url)

                media = MediaStreamingOptions(
                    transport_url=ws_url,
                    transport_type=StreamingTransportType.WEBSOCKET,
                    content_type=MediaStreamingContentType.AUDIO,
                    audio_channel_type=MediaStreamingAudioChannelType.MIXED,
                    start_media_streaming=True,
                    enable_bidirectional=True,
                    audio_format=AudioFormat.PCM16_K_MONO,
                )

                try:
                    res = client.answer_call(
                        incoming_call_context=incoming_call_context,
                        callback_url=cb_url,
                        media_streaming=media,
                    )
                    STATE["call_connection_id"] = res.call_connection.call_connection_id
                    print("Answered; call_connection_id:", STATE["call_connection_id"])
                except Exception as e:
                    import traceback
                    print("answer_call FAILED:", e)
                    traceback.print_exc()
            else:
                print("Event:", etype)
    return JSONResponse({"ok": True})

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    STATE["ws"] = ws
    print("WS CONNECTED")

    # --- Recording (caller -> wav) ---
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    wav_path = f"caller-{ts}.wav"
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
    print("REC →", wav_path)

    # --- STT (queues + worker) ---
    stt_q: "queue.Queue[bytes]" = queue.Queue(maxsize=400)
    tts_q: "queue.Queue[str]" = queue.Queue(maxsize=50)
    stt_thread: Optional[threading.Thread] = None
    if VOSK_MODEL:
        stt_thread = threading.Thread(
            target=start_stt_worker, args=(VOSK_MODEL, SAMPLE_RATE, stt_q, tts_q), daemon=True
        )
        stt_thread.start()
        print("STT worker started")

    # --- TX (mic -> phone) ---
    tx_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def on_mic(indata, frames, time_info, status):
        if STATE["muted"]:
            return None
        try:
            if not tx_queue.full():
                tx_queue.put_nowait(indata.copy().tobytes())
        except Exception:
            pass
        return None

    mic = None
    try:
        mic = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="int16", blocksize=FRAME_SAMPLES, callback=on_mic
        )
        mic.start()
        print("MIC started")
    except Exception as e:
        print("MIC ERROR:", e)

    async def tx_sender():
        try:
            while True:
                buf = await tx_queue.get()
                b64 = base64.b64encode(buf).decode("ascii")
                STATE["seq"] += 1
                payload = {"kind": "audioData", "audioData": {"data": b64, "sequenceNumber": STATE["seq"]}}
                await ws.send_text(json.dumps(payload))
        except Exception as e:
            print("TX sender stopped:", e)

    tx_task = asyncio.create_task(tx_sender())

    # --- TTS consumer (speak back final STT) ---
    async def tts_consumer():
        while True:
            txt = await asyncio.to_thread(tts_q.get)
            if txt is None:
                break
            print(f"TTS<-STT: {txt!r}")
            STATE["muted"] = True
            try:
                wav_path_tts = await synthesize_tts_to_wav(txt, voice_id=None, rate=180)
                wav_path_tts = ensure_16k_mono_wav(wav_path_tts)
                STATE["seq"] = await play_wav_to_phone(STATE["ws"], wav_path_tts, seq_start=STATE["seq"])
            finally:
                STATE["muted"] = False

    tts_task = asyncio.create_task(tts_consumer())

    # --- RX loop (phone -> record, STT feed, optional monitor) ---
    frames_in = 0
    try:
        while True:
            raw = await ws.receive_text()
            obj = json.loads(raw)
            kind = (obj.get("kind") or "").lower()

            if kind == "audiometadata":
                print("META:", obj)

            elif kind == "audiodata":
                ad = obj.get("audioData") or obj.get("audiodata") or {}
                b64 = ad.get("data")
                if not b64:
                    continue
                buf = base64.b64decode(b64)

                wf.writeframes(buf)

                if VOSK_MODEL:
                    with contextlib.suppress(Exception):
                        if stt_q.qsize() < 350:
                            stt_q.put_nowait(buf)

                # Optional monitor (HEADPHONES!)
                # pcm = np.frombuffer(buf, dtype=np.int16)
                # pcm = np.clip(pcm.astype(np.int32) * 2, -32768, 32767).astype(np.int16)
                # sd.play(pcm, SAMPLE_RATE, blocking=False)

                frames_in += 1
                if frames_in % 50 == 0:
                    print(f"inbound frames: {frames_in}")

    except Exception as e:
        print("WS CLOSED:", repr(e))
    finally:
        # stop STT + TTS consumer
        with contextlib.suppress(Exception):
            if VOSK_MODEL and stt_thread:
                stt_q.put(None)
                stt_thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            tts_q.put_nowait(None)
            tts_task.cancel()

        # cleanup audio + tasks
        with contextlib.suppress(Exception):
            wf.close(); print("REC saved:", wav_path)
        with contextlib.suppress(Exception):
            if mic: mic.stop(); mic.close()
        tx_task.cancel()

        # clear state
        STATE["ws"] = None
        STATE["call_connection_id"] = None
        print("CLEANUP DONE")

# ---------- Control Endpoints ----------
@app.post("/api/mute")
async def api_mute(on: bool = Body(embed=True)):
    if STATE["ws"] is None:
        raise HTTPException(409, "No active call")
    STATE["muted"] = bool(on)
    return {"muted": STATE["muted"]}

@app.post("/api/play")
async def api_play(file: str = Body(embed=True)):
    if STATE["ws"] is None:
        raise HTTPException(409, "No active call")
    if not file or not os.path.isfile(file):
        raise HTTPException(400, f"file not found: {file}")

    try:
        norm = ensure_16k_mono_wav(file)
    except Exception as e:
        raise HTTPException(500, f"Failed to prepare WAV: {e}")

    print(f"PLAY: streaming {norm}")
    STATE["muted"] = True
    try:
        STATE["seq"] = await play_wav_to_phone(STATE["ws"], norm, seq_start=STATE["seq"])
    finally:
        STATE["muted"] = False
    return {"played": norm, "seq": STATE["seq"]}

# safe voice listing on a worker thread (COM init)
def _list_voices_sync():
    try:
        import pythoncom  # from pywin32
        pythoncom.CoInitialize()
    except Exception:
        pythoncom = None
    try:
        eng = pyttsx3.init()
        vs = eng.getProperty("voices")
        out = [{"id": v.id, "name": getattr(v, "name", ""), "lang": getattr(v, "languages", None)} for v in vs]
        with contextlib.suppress(Exception):
            eng.stop()
        return out
    finally:
        if 'pythoncom' in locals() and pythoncom:
            with contextlib.suppress(Exception):
                pythoncom.CoUninitialize()

@app.get("/api/voices")
async def api_voices():
    if not TTS_AVAILABLE:
        raise HTTPException(501, "pyttsx3 not installed (pip install pyttsx3)")
    try:
        voices = await asyncio.to_thread(_list_voices_sync)
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(500, f"pyttsx3 voice enumeration failed: {e}")

@app.post("/api/say")
async def api_say(payload: dict = Body(...)):
    if STATE["ws"] is None:
        raise HTTPException(409, "No active call")
    if not TTS_AVAILABLE:
        raise HTTPException(501, "pyttsx3 not installed (pip install pyttsx3)")

    text = (payload.get("text") or "").strip()
    voice = payload.get("voice")  # voice id from /api/voices
    rate = int(payload.get("rate", 180))
    if not text:
        raise HTTPException(400, "field 'text' is required")

    try:
        wav_path = await synthesize_tts_to_wav(text, voice, rate)
        wav_path = ensure_16k_mono_wav(wav_path)
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")

    print(f"TTS: streaming {wav_path}")
    STATE["muted"] = True
    try:
        STATE["seq"] = await play_wav_to_phone(STATE["ws"], wav_path, seq_start=STATE["seq"])
    finally:
        STATE["muted"] = False
    return {"ok": True, "played": wav_path, "seq": STATE["seq"]}

@app.post("/api/hangup")
async def api_hangup():
    cid = STATE.get("call_connection_id")
    if not cid:
        raise HTTPException(409, "No active call")
    try:
        client.get_call_connection(cid).hang_up(is_for_everyone=True)
    except Exception as e:
        raise HTTPException(500, f"hangup failed: {e}")
    return {"hung_up": True}
