import os, json, base64
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import asyncio, json, base64
import numpy as np
import sounddevice as sd
from fastapi import WebSocket
import wave, datetime


from azure.communication.callautomation import (
    CallAutomationClient, MediaStreamingOptions,
    StreamingTransportType, MediaStreamingContentType,
    MediaStreamingAudioChannelType, AudioFormat
)

load_dotenv()
ACS = os.environ["ACS_CONNECTION_STRING"]
PUBLIC_BASE = os.environ["PUBLIC_BASE"]  # e.g., https://<ngrok>.ngrok-free.app

app = FastAPI()
client = CallAutomationClient.from_connection_string(ACS)

# Audio Stuff
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/events")
async def events(request: Request):
    body = await request.json()

    # 1) Handle Event Grid subscription validation (first-time handshake)
    if isinstance(body, list) and body and body[0].get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
        code = body[0]["data"]["validationCode"]
        print("EventGrid validation:", code)
        return JSONResponse({"validationResponse": code})

    # 2) Handle ACS call events
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
                    client.answer_call(
                        incoming_call_context=incoming_call_context,
                        callback_url=cb_url,
                        media_streaming=media,
                    )
                    print("Answered call + enabled streaming")
                except Exception as e:
                    import traceback
                    print("answer_call FAILED:", e)
                    traceback.print_exc()

            # Log other events for now
            else:
                print("Event:", etype)
    return JSONResponse({"ok": True})

import asyncio, base64, json, wave

async def play_wav_to_phone(ws, path: str, seq_start: int = 0, frame_ms: int = 20):
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
                # pad last partial frame to keep cadence exact
                chunk += b"\x00" * (bytes_per_frame - len(chunk))

            payload = {
                "kind": "audioData",
                "audioData": {"data": base64.b64encode(chunk).decode("ascii"),
                              "sequenceNumber": seq}
            }
            await ws.send_text(json.dumps(payload))
            seq += 1

            # precise pacing
            next_t += interval
            await asyncio.sleep(max(0.0, next_t - loop.time()))
    return seq

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("WS CONNECTED")

    # --- recording setup (caller -> wav) ---
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    wav_path = f"caller-{ts}.wav"
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
    print("REC →", wav_path)

    # --- TX setup (mic -> phone) ---
    seq = 0
    tx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    muted = False  # set True while playing WAV to avoid talking over it

    def on_mic(indata, frames, time_info, status):
        if muted:
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
            dtype="int16", blocksize=FRAME_SAMPLES,
            callback=on_mic
        )
        mic.start()
        print("MIC started")
    except Exception as e:
        print("MIC ERROR:", e)

    async def tx_sender():
        nonlocal seq
        try:
            while True:
                buf = await tx_queue.get()
                b64 = base64.b64encode(buf).decode("ascii")
                seq += 1
                payload = {"kind": "audioData", "audioData": {"data": b64, "sequenceNumber": seq}}
                await ws.send_text(json.dumps(payload))
        except Exception as e:
            print("TX sender stopped:", e)

    tx_task = asyncio.create_task(tx_sender())

    # --- Optional: play a greeting WAV, then unmute mic ---
    try:
        muted = True
        seq = await play_wav_to_phone(ws, "greeting_16k_mono.wav", seq_start=seq)
    finally:
        muted = False

    # --- RX loop (phone -> PC speakers + wav) ---
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
                wf.writeframes(buf)  # record

                # optional monitor to speakers (wear headphones!)
                # pcm = np.frombuffer(buf, dtype=np.int16)
                # pcm = np.clip(pcm.astype(np.int32) * 2, -32768, 32767).astype(np.int16)  # +6 dB
                # sd.play(pcm, SAMPLE_RATE, blocking=False)

                frames_in += 1
                if frames_in % 50 == 0:
                    print(f"inbound frames: {frames_in}")

            else:
                # keepalives/other events
                pass

    except Exception as e:
        print("WS CLOSED:", repr(e))
    finally:
        try: wf.close(); print("REC saved:", wav_path)
        except: pass
        if mic:
            try: mic.stop(); mic.close()
            except: pass
        tx_task.cancel()
        print("CLEANUP DONE")