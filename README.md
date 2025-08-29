# ACS Bridge - Azure Communication Services Audio Bridge

Bidirectional phone audio over Azure Communication Services (ACS) with a clean, modular FastAPI architecture.

- **Answers inbound PSTN calls** to your ACS number  
- **Streams phone audio** to your PC (with WAV recording)
- **Streams your mic audio** back to the caller
- **High-Quality TTS** with Piper neural synthesis and pyttsx3 fallback
- **Offline STT** support with Vosk models
- **LLM Integration** with Ollama for intelligent call handling
- **Control APIs** for mute/unmute, play WAV, hang up, and voice synthesis

> **Single port architecture**: `/events` (Event Grid) + `/ws` (media) + `/api/*` (controls) all on `:8080`

---

## Features

- **Call Management**: Answer calls with ACS Call Automation
- **Bidirectional Audio**: WebSocket media streaming (PCM 16 kHz mono)  
- **Real-time Processing**: Mic → Phone streaming with sequence numbering
- **Recording**: Phone → PC audio saved as WAV files
- **Speech-to-Text**: Offline STT using Vosk (optional)
- **Text-to-Speech**: High-quality neural TTS using Piper with pyttsx3 fallback
- **LLM Integration**: Intelligent call handling with Ollama and configurable personas
- **Guardrails**: Content filtering and safety controls for LLM interactions
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
│  │  ├─ tts_piper.py        # Piper neural TTS implementation
│  │  ├─ tts_pyttsx3.py      # pyttsx3 TTS fallback
│  │  └─ tts_composite.py    # Composite TTS (Piper + fallback)
│  └─ audio/                 # Audio processing utilities
│     ├─ utils.py            # WAV normalization, streaming
│     ├─ textnorm.py         # Text normalization for TTS
│     └─ constants.py        # Audio format constants
├─ src/phone_agent/          # LLM-powered call automation
│  ├─ services/              # LLM and guardrails services
│  │  ├─ llm_base.py         # LLM service protocol
│  │  ├─ llm_ollama.py       # Ollama LLM implementation
│  │  └─ guardrails.py       # Safety and content filtering
│  └─ README.md              # Phone agent documentation
├─ config/                   # Configuration files
│  └─ persona.yaml           # LLM persona configuration
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
- **System Dependencies**: 
  - PortAudio (required for audio processing)
  - Piper TTS binary (for high-quality neural TTS)
- **Optional Dependencies**:
  - Vosk model for STT
  - Ollama for LLM integration
- **Windows**: Microphone permissions for desktop apps

---

## System Dependencies Setup

### PortAudio (Required)

PortAudio is required for audio processing. Install it using your system package manager:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
PortAudio is typically bundled with the sounddevice Python package. If you encounter issues:
1. Install Microsoft Visual C++ Redistributable
2. Or install PortAudio manually from [portaudio.com](http://portaudio.com/)

### Piper TTS (Recommended for High-Quality Speech)

Piper provides neural text-to-speech with significantly higher quality than pyttsx3.

**Install Piper Binary:**

**Linux (x86_64):**
```bash
# Download and install Piper
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
sudo cp piper/piper /usr/local/bin/
sudo chmod +x /usr/local/bin/piper
```

**macOS:**
```bash
# Using Homebrew
brew install piper-tts

# Or download manually
wget https://github.com/rhasspy/piper/releases/latest/download/piper_macos_x64.tar.gz
tar -xzf piper_macos_x64.tar.gz
sudo cp piper/piper /usr/local/bin/
```

**Windows:**
```bash
# Download from releases page
# https://github.com/rhasspy/piper/releases/latest
# Extract piper.exe to a directory in your PATH
```

**Download Voice Models:**

Create a voices directory and download models:
```bash
mkdir -p voices
cd voices

# High-quality English (US) voice (recommended)
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json

# Alternative: Medium quality (smaller file)
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

**Test Piper Installation:**
```bash
echo "Hello, this is a test." | piper -m voices/en_US-lessac-high.onnx -f test.wav
# Should create test.wav file
```

### Vosk STT Models (Optional)

For speech-to-text functionality, download a Vosk model:

```bash
# Create models directory
mkdir -p models

# Download small English model (40MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models/

# Or large English model (1.8GB, better accuracy)
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip -d models/
```

### Ollama LLM (Optional)

For intelligent call handling with LLM integration:

**Install Ollama:**
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

**Download a Model:**
```bash
# Small, fast model (1.3GB)
ollama pull llama3.2:1b

# Or larger, more capable model (4.7GB)
ollama pull llama3.2:3b
```

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

# Install optional dependencies
# For speech-to-text:
pip install vosk

# For enhanced TTS (pyttsx3 fallback):
pip install pyttsx3
# Windows also needs: pip install pywin32

# For LLM integration:
pip install ollama pyyaml

# Or install all optional dependencies at once:
pip install -e ".[stt,tts,llm]"
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings:
# ACS_CONNECTION_STRING=endpoint=https://your-acs.communication.azure.com/;accesskey=...
# PUBLIC_BASE=https://your-tunnel.ngrok-free.app  

# Optional: STT model path
# STT_MODEL_PATH=./models/vosk-model-small-en-us-0.15

# Optional: Piper TTS settings (for high-quality neural speech)
# PIPER_VOICE_PATH=./voices/en_US-lessac-high.onnx
# PIPER_LENGTH_SCALE=1.08
# PIPER_NOISE_SCALE=0.65
# PIPER_NOISE_W=0.80
# PIPER_SENTENCE_SILENCE=0.25

# Optional: Ollama LLM settings
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b
# OLLAMA_TIMEOUT=30.0
# OLLAMA_TEMPERATURE=0.4
# PERSONA_CONFIG_PATH=./config/persona.yaml
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
make install-all       # Everything including STT/TTS/LLM

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

## LLM Integration

The project includes optional LLM integration for intelligent call handling through the `phone_agent` module. This enables:

- **Automated Responses**: AI-generated responses to caller inquiries
- **Persona Management**: Configurable AI agent personality and behavior  
- **Content Safety**: Built-in guardrails and content filtering
- **Conversation Memory**: Context-aware responses within calls

### Quick LLM Setup

1. **Install Ollama** (see System Dependencies section above)
2. **Pull a model**: `ollama pull llama3.2:1b`
3. **Configure environment** variables in `.env`
4. **Customize persona** in `config/persona.yaml`

### Usage Example

```python
from phone_agent import OllamaLLMService, Guardrails

# Initialize services
llm_service = OllamaLLMService()
guardrails = Guardrails.from_config("config/persona.yaml")

# Process user input safely
user_text = "How can you help me?"
filtered_input = guardrails.apply_input_filters(user_text)

# Generate AI response
response = await llm_service.generate([
    {"role": "system", "content": guardrails.build_system_prompt()},
    {"role": "user", "content": filtered_input}
])
```

For detailed LLM documentation, see [`src/phone_agent/README.md`](src/phone_agent/README.md).

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
- **PiperTTSService**: Neural text-to-speech using Piper (high-quality)
- **Pyttsx3TTSService**: Text-to-speech using pyttsx3 (fallback)
- **CompositeTTSService**: Intelligent TTS routing (Piper → pyttsx3 fallback)
- **OllamaLLMService**: LLM integration for intelligent call handling
- **Guardrails**: Content filtering and safety controls for LLM interactions

---

## Troubleshooting

### Common Issues

1. **Module not found errors**: Install dependencies with `make install-all`
2. **Audio device errors**: Check microphone permissions and PortAudio installation
3. **Webhook failures**: Verify ngrok tunnel and PUBLIC_BASE setting
4. **STT not working**: Install Vosk and download a model
5. **TTS not working**: Check Piper installation and voice model availability

### System Dependency Issues

**PortAudio Errors:**
```
OSError: PortAudio library not found
```
- **Linux**: `sudo apt-get install portaudio19-dev`
- **macOS**: `brew install portaudio`
- **Windows**: Install Visual C++ Redistributable

**Piper TTS Issues:**
```
Warning: Piper voice file not found: ./voices/en-us-high.onnx
```
- Verify Piper binary is installed: `which piper` (should return a path)
- Download voice models as shown in setup section
- Check `PIPER_VOICE_PATH` environment variable points to existing `.onnx` file
- Test Piper manually: `echo "test" | piper -m voices/en_US-lessac-high.onnx -f test.wav`

**Vosk STT Issues:**
```
Warning: STT_MODEL_PATH './models/vosk-model-small-en-us-0.15' does not exist
```
- Download a Vosk model as shown in setup section
- Verify model directory contains necessary files
- Check `STT_MODEL_PATH` points to the extracted model directory

**Ollama LLM Issues:**
```
Error: Ollama connection failed
```
- Verify Ollama is running: `ollama list`
- Check `OLLAMA_HOST` setting (default: http://localhost:11434)
- Pull required model: `ollama pull llama3.2:1b`
- Test connection: `curl http://localhost:11434/api/tags`

### Audio Quality Issues

**Poor TTS Quality:**
- Ensure Piper is installed and working (check logs for "Piper TTS available")
- If only pyttsx3 is available, consider installing Piper for better quality
- Adjust Piper parameters in `.env` file:
  - `PIPER_LENGTH_SCALE`: Speech speed (1.0 = normal, >1.0 = slower)
  - `PIPER_NOISE_SCALE`: Variability (0.0-1.0, higher = more variable)

**STT Recognition Issues:**
- Use larger Vosk model for better accuracy
- Check microphone permissions and audio levels
- Verify 16kHz mono audio format is supported

### Performance Issues

**High CPU/Memory Usage:**
- Use smaller models (llama3.2:1b instead of larger variants)
- Reduce `OLLAMA_MAX_TOKENS` setting
- Monitor system resources during calls

**Slow Response Times:**
- Ensure models are loaded: `ollama list`
- Check network connectivity for model downloads
- Consider using local models instead of API calls

### Logs

The application uses structured logging. Check console output for:
- System dependency availability status
- TTS service selection (Piper vs pyttsx3)
- Connection status and audio processing events  
- STT transcriptions and TTS synthesis progress
- LLM generation and guardrails filtering
- Error details with specific failure reasons

**Enable Debug Logging:**
Set environment variable: `LOG_LEVEL=DEBUG` for more detailed output.

---
