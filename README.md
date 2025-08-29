# ACS Local Audio Bridge (FastAPI + WebSocket, 1-port)

Bidirectional phone audio over Azure Communication Services (ACS) to a local Python app.
- Answers inbound PSTN calls to your ACS number
- Streams phone audio to your PC (optional local monitor + WAV recording)
- Streams your mic audio back to the caller
- Simple control endpoints: mute/unmute, play WAV, hang up

> Single port: `/events` (Event Grid) + `/ws` (media) both on `:8080`.

---

## Features

- **Answer calls** with ACS Call Automation
- **WebSocket media**: PCM 16 kHz mono, bidirectional
- **Mic → Phone** streaming with sequence numbering
- **Phone → PC** (optional speaker monitor) + **WAV recording**
- **Controls**:
  - `POST /api/mute { "on": true|false }`
  - `POST /api/play { "file": "greeting_16k_mono.wav" }`
  - `POST /api/hangup`

---

## Prerequisites

- **Python** 3.10–3.11
- **Azure Communication Services** resource + **phone number**
- **ngrok** (or equivalent HTTPS/WSS tunnel)
- Windows: enable **Microphone** permission for desktop apps

---

## Quick Start

```bash
# 1) create venv
python -m venv .venv
.\.venv\Scripts\activate

# 2) install
pip install -r requirements.txt
# (or)
pip install fastapi uvicorn[standard] python-dotenv sounddevice numpy \
            azure-communication-callautomation

# 3) env
copy .env.example .env
# edit .env:
# ACS_CONNECTION_STRING=endpoint=https://<your-acs>.communication.azure.com/;accesskey=<key>
# PUBLIC_BASE=https://<your-ngrok>.ngrok-free.app

# 4) run server
uvicorn app:app --reload --host 0.0.0.0 --port 8080

# 5) expose
ngrok http http://localhost:8080
# use the https URL as PUBLIC_BASE in .env
