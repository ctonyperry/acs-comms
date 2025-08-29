# Phone Agent LLM Integration

This module provides LLM-powered call automation capabilities with Ollama integration, persona management, and comprehensive guardrails.

## Features

- **LLM Integration**: Streaming and non-streaming text generation with Ollama
- **Persona System**: Configurable agent personality and behavior
- **Guardrails**: Input filtering and output validation for safety
- **Error Handling**: Robust error handling with backoff retry
- **Type Safety**: Full type annotations and Protocol definitions

## Quick Start

### 1. Install Dependencies

```bash
# Install LLM dependencies
pip install .[llm]

# Or install Ollama client directly  
pip install ollama pyyaml
```

### 2. Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2:1b
```

### 3. Configure Environment

Add to your `.env` file:

```env
# Ollama LLM settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=30.0
OLLAMA_MAX_RETRIES=3
OLLAMA_TEMPERATURE=0.4
OLLAMA_TOP_P=0.9
OLLAMA_MAX_TOKENS=512

# Persona configuration
PERSONA_CONFIG_PATH=./config/persona.yaml
```

### 4. Basic Usage

```python
import asyncio
from phone_agent import OllamaLLMService, Guardrails, assemble_messages

async def main():
    # Load persona configuration
    import yaml
    with open("config/persona.yaml") as f:
        persona_config = yaml.safe_load(f)
    
    # Initialize components
    guardrails = Guardrails(persona_config)
    llm_service = OllamaLLMService()
    
    # Generate system prompt
    system_prompt = guardrails.build_system_prompt()
    
    # Process user input with filtering
    user_input = "Hello, how can I help?"
    filtered_input = guardrails.apply_input_filters(user_input)
    
    # Assemble messages and generate response
    messages = assemble_messages(system_prompt, [], filtered_input)
    
    # Streaming generation
    async for chunk in await llm_service.generate(messages, stream=True):
        print(chunk, end="", flush=True)
    
    await llm_service.close()

asyncio.run(main())
```

## Architecture

### Core Components

1. **LLM Base (`llm_base.py`)**
   - Protocol definitions for LLM services
   - Message assembly utilities
   - Exception types

2. **Ollama Service (`llm_ollama.py`)**
   - Ollama integration with streaming support
   - Automatic model pulling and availability checking
   - Retry logic with exponential backoff

3. **Guardrails (`guardrails.py`)**
   - Input text filtering (blocklist, normalization)
   - Output validation (sensitive data, unauthorized tools)
   - System prompt generation

### Configuration

The persona configuration (`config/persona.yaml`) defines:

```yaml
name: "Calm Support Agent"
style: |
  You are concise, friendly, and helpful. You de-escalate tense situations.
constraints:
  - "Never give legal or medical advice."
  - "Never collect payment card numbers or SSNs."
guardrails:
  blocklist:
    - "credit card"
    - "social security number"
  allow_tools:
    - "transfer_call"
    - "create_ticket"
llm:
  temperature: 0.4
  top_p: 0.9
  max_tokens: 512
  stop: ["</s>"]
```

## API Reference

### LLM Interface

```python
class LLM(Protocol):
    async def generate(
        self,
        messages: list[Message],
        stream: bool = True,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str] | str:
        """Generate response from messages."""
        ...

    async def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Summarize the given text."""
        ...
```

### Guardrails API

```python
class Guardrails:
    def apply_input_filters(self, text: str) -> str:
        """Apply input filters to sanitize user text."""
        ...

    def is_output_allowed(self, text: str) -> tuple[bool, str]:
        """Check if output text is allowed by guardrails."""
        ...

    def build_system_prompt(self) -> str:
        """Build system prompt with persona and constraints."""
        ...
```

### Utilities

```python
def assemble_messages(
    persona_system: str, 
    history: list[Message], 
    user_text: str
) -> list[Message]:
    """Assemble messages for LLM generation."""
    ...
```

## Error Handling

The system includes comprehensive error handling:

- **LLMUnavailable**: Raised when LLM service is not available
- **Automatic Retry**: Exponential backoff for transient failures
- **Model Auto-Pull**: Automatically downloads missing models
- **Graceful Degradation**: Safe fallbacks when services are unavailable

## Demo

Run the interactive demo:

```bash
python demo_llm.py
```

This demonstrates:
- Guardrails filtering in action
- Text summarization
- Interactive conversation with streaming
- Error handling and availability checking

## Testing

Run the test suite:

```bash
# Run LLM-specific tests
pytest tests/test_llm_base.py tests/test_guardrails.py tests/test_ollama_service.py -v

# Run all tests
pytest tests/ -v
```

## Integration with ACS Bridge

The phone agent integrates with the existing ACS Bridge architecture:

1. Add LLM service to dependency injection (`deps.py`)
2. Create phone agent router for LLM endpoints
3. Integrate with media streaming for real-time responses
4. Use existing logging and configuration systems

Example integration:

```python
from phone_agent import OllamaLLMService, Guardrails

@lru_cache()
def get_llm_service(settings: Settings = None) -> OllamaLLMService:
    """Get LLM service instance."""
    if settings is None:
        settings = get_settings()
    
    return OllamaLLMService(
        model=settings.ollama_model,
        host=settings.ollama_host,
        temperature=settings.ollama_temperature,
    )

def get_llm_service_dependency() -> OllamaLLMService:
    """FastAPI dependency for LLM service."""
    return get_llm_service()
```

## Security Considerations

- Input sanitization prevents injection attacks
- Output validation blocks sensitive information leakage
- Tool usage is restricted to allowed list
- Configurable blocklists for domain-specific filtering
- No persistent storage of conversation data
- Rate limiting and timeout protection

## Performance

- Streaming responses for real-time interaction
- Configurable model sizes (1B-70B+ parameters)
- Connection pooling and keep-alive
- Efficient retry mechanisms
- Memory-efficient conversation history management