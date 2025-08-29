# Ollama LLM Integration Documentation

## 1. File Diffs & Function Signatures

### New Services

#### `src/acs_bridge/services/llm_base.py`
```python
class LLMService(Protocol):
    @property
    def is_available(self) -> bool: ...
    
    async def generate(
        self, 
        messages: list[LLMMessage], 
        stream: bool = True
    ) -> AsyncGenerator[str, None] | LLMResponse: ...
    
    async def summarize(self, text: str) -> str: ...

class BaseLLMService(ABC):
    def _prepare_system_message(
        self, 
        messages: list[LLMMessage], 
        persona_content: str
    ) -> list[LLMMessage]: ...
```

#### `src/acs_bridge/services/llm_ollama.py`
```python
class OllamaLLMService(BaseLLMService):
    def __init__(
        self, 
        config: OllamaConfig,
        persona_config_path: str | None = None,
        guardrails_service: GuardrailsService | None = None
    ): ...
    
    async def generate(
        self, 
        messages: list[LLMMessage], 
        stream: bool = True
    ) -> AsyncGenerator[str, None] | LLMResponse: ...
    
    async def summarize(self, text: str) -> str: ...
    
    def _format_messages_for_ollama(self, messages: list[LLMMessage]) -> str: ...
```

#### `src/acs_bridge/services/guardrails.py`
```python
class GuardrailsService:
    def __init__(self, config: GuardrailsConfig | None = None): ...
    
    def validate_input(self, content: str, role: str = "user") -> None: ...
    def validate_output(self, content: str) -> str: ...
    def is_safe_input(self, content: str, role: str = "user") -> bool: ...
    def is_safe_output(self, content: str) -> bool: ...
```

### Settings Extensions

#### `src/acs_bridge/settings.py`
```python
class Settings(BaseSettings):
    # Ollama LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9
    ollama_seed: int | None = None
    ollama_max_tokens: int = 2048
    ollama_stop: list[str] = ["\n\n", "Human:", "User:"]
    persona_config_path: str = "./config/persona.yaml"
```

### Configuration Files

#### `config/persona.yaml`
- Persona definition with role, traits, and guidelines
- Safety constraints and communication rules
- Technical and privacy constraints

## 2. Example System Prompt Composition (Persona Applied)

### Input Messages:
```python
messages = [
    LLMMessage(role="user", content="How can I improve my communication skills?")
]
```

### Persona Content from YAML:
```yaml
persona:
  role: "You are a professional AI assistant integrated with Azure Communication Services."
  traits: ["Professional and courteous", "Clear and concise communication"]
  guidelines: ["Always maintain a professional tone", "Provide clear and accurate information"]

constraints:
  safety_rules: ["Never provide personal information about users", "Respect user privacy"]
  communication_rules: ["Keep responses concise and relevant", "Use clear, professional language"]
```

### Final System Prompt:
```
You are a professional AI assistant integrated with Azure Communication Services.

Key personality traits:
- Professional and courteous
- Clear and concise communication

Behavioral guidelines:
- Always maintain a professional tone
- Provide clear and accurate information

CRITICAL SAFETY RULES (NEVER VIOLATE):
- Never provide personal information about users
- Respect user privacy and confidentiality

Communication rules:
- Keep responses concise and relevant
- Use clear, professional language
```

### Resulting Ollama Prompt Format:
```
System: You are a professional AI assistant integrated with Azure Communication Services.

Key personality traits:
- Professional and courteous
- Clear and concise communication

Behavioral guidelines:
- Always maintain a professional tone
- Provide clear and accurate information

CRITICAL SAFETY RULES (NEVER VIOLATE):
- Never provide personal information about users
- Respect user privacy and confidentiality

Communication rules:
- Keep responses concise and relevant
- Use clear, professional language