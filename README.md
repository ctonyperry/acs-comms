# ACS Bridge - Azure Communication Services Audio Bridge

Bidirectional phone audio over Azure Communication Services (ACS) with a clean, modular FastAPI architecture.

- **Answers inbound PSTN calls** to your ACS number  
- **Streams phone audio** to your PC (with WAV recording)
- **Streams your mic audio** back to the caller
- **Offline STT/TTS** support with Vosk and pyttsx3
- **Control APIs** for mute/unmute, play WAV, hang up, and voice synthesis

> **Single port architecture**: `/events` (Event Grid) + `/ws` (media) + `/api/*` (controls) all on `:8080`

---

## Features

- **Call Management**: Answer calls with ACS Call Automation
- **Bidirectional Audio**: WebSocket media streaming (PCM 16 kHz mono)  
- **Real-time Processing**: Mic → Phone streaming with sequence numbering
- **Recording**: Phone → PC audio saved as WAV files
- **Speech-to-Text**: Offline STT using Vosk (optional)
- **Text-to-Speech**: Offline TTS using pyttsx3/SAPI (optional)
- **Control APIs**:
  - `POST /api/mute` - Mute/unmute microphone
  - `POST /api/play` - Play WAV file to caller
  - `POST /api/say` - Synthesize and speak text
  - `GET /api/voices` - List available TTS voices
  - `POST /api/hangup` - Hang up active call
  - `GET /api/health` - Application health status

---

## Project Structure

```
acs-comms/
├─ src/acs_bridge/           # Main application package
│  ├─ main.py                # FastAPI app factory + entry point
│  ├─ settings.py            # Pydantic settings (env vars)
│  ├─ logging_config.py      # Structured logging configuration
│  ├─ deps.py                # Dependency injection
│  ├─ routers/               # FastAPI route handlers
│  │  ├─ events.py           # /events route (Event Grid)
│  │  ├─ media_ws.py         # /ws route (WebSocket media)
│  │  └─ controls.py         # /api/* routes (controls)
│  ├─ services/              # Business logic services
│  │  ├─ acs_client.py       # ACS client wrapper
│  │  ├─ media_streamer.py   # Audio streaming management
│  │  ├─ stt_vosk.py         # Vosk STT implementation
│  │  └─ tts_pyttsx3.py      # pyttsx3 TTS implementation
│  ├─ audio/                 # Audio processing utilities
│  │  ├─ utils.py            # WAV normalization, streaming
│  │  └─ constants.py        # Audio format constants
│  └─ models/                # Data models and schemas
│     ├─ state.py            # Call state management
│     └─ schemas.py          # Pydantic API models
├─ tests/                    # Unit tests
├─ requirements.txt          # Python dependencies
├─ pyproject.toml           # Project configuration
├─ Makefile                 # Development tasks (Linux/macOS)
├─ tasks.bat                # Development tasks (Windows)
└─ run.py                   # Local development entry point
```

---

## Prerequisites

- **Python** 3.10+ 
- **Azure Communication Services** resource with phone number
- **ngrok** or similar tunnel for webhooks
- **Windows**: Microphone permissions for desktop apps
- **Optional**: Vosk model for STT, pyttsx3 for TTS

---

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repository-url>
cd acs-comms

# Create virtual environment  
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install STT/TTS support
pip install vosk pyttsx3
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings:
# ACS_CONNECTION_STRING=endpoint=https://your-acs.communication.azure.com/;accesskey=...
# PUBLIC_BASE=https://your-tunnel.ngrok-free.app  
# STT_MODEL_PATH=./models/vosk-model-small-en-us-0.15  # Optional
```

### 3. Run Application

```bash
# Using run.py (recommended for development)
python run.py

# Or using uvicorn directly
uvicorn src.acs_bridge.main:app --reload --host 0.0.0.0 --port 8080

# Or using make
make run
```

### 4. Expose with ngrok

```bash
# In another terminal
ngrok http 8080

# Copy the https URL to PUBLIC_BASE in .env
```

---

## Development

### Available Commands

```bash
# Install dependencies
make install          # Production only
make install-dev       # Development tools  
make install-all       # Everything including STT/TTS

# Code quality
make fmt              # Format with black
make lint             # Lint with ruff  
make type             # Type check with mypy
make test             # Run pytest tests
make all-checks       # Run all quality checks

# Running
make run              # Start development server
make clean            # Clean build artifacts
```

### Windows Users

Use `tasks.bat` instead of `make`:

```cmd
tasks.bat install-all
tasks.bat fmt
tasks.bat test  
tasks.bat run
```

---

## API Reference

### Event Grid Webhook
- `POST /events` - Handles ACS events and Event Grid validation

### Media WebSocket  
- `WS /ws` - Bidirectional audio streaming (PCM 16kHz mono)

### Control APIs
- `POST /api/mute` - `{"on": true}` - Mute/unmute microphone
- `POST /api/play` - `{"file": "path.wav"}` - Play audio file  
- `POST /api/say` - `{"text": "hello", "voice": "id", "rate": 180}` - Text-to-speech
- `GET /api/voices` - List available TTS voices
- `POST /api/hangup` - Hang up active call
- `GET /api/health` - Application status

---

## Architecture

The application follows clean architecture principles:

- **Separation of Concerns**: Routers, services, models clearly separated
- **Dependency Injection**: Services wired through deps.py
- **Protocol Interfaces**: STT/TTS services use Protocol for testability  
- **Type Safety**: Full type hints with mypy validation
- **Error Handling**: Structured logging and graceful degradation
- **Testing**: Unit tests for core functionality

### Key Services

- **MediaStreamer**: Manages bidirectional audio streaming
- **ACSClient**: Wraps Azure Communication Services SDK
- **VoskSTTService**: Speech-to-text using Vosk (offline)
- **Pyttsx3TTSService**: Text-to-speech using pyttsx3 (offline)

---

## Troubleshooting

### Common Issues

1. **Module not found errors**: Install dependencies with `make install-all`
2. **Audio device errors**: Check microphone permissions
3. **Webhook failures**: Verify ngrok tunnel and PUBLIC_BASE setting
4. **STT not working**: Install Vosk and download a model
5. **TTS not working**: Install pyttsx3 and check available voices

### Logs

The application uses structured logging. Check console output for:
- Connection status
- Audio processing events  
- STT transcriptions
- TTS synthesis progress
- Error details

---
