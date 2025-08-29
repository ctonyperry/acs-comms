"""Audio constants for ACS Bridge application."""

# Audio format constants
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# Queue sizes
STT_QUEUE_SIZE = 400
TTS_QUEUE_SIZE = 50
TX_QUEUE_SIZE = 200

# STT threshold
STT_QUEUE_THRESHOLD = 350
