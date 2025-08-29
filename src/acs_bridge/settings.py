"""Application settings management using Pydantic."""

import os
from typing import Optional, List

from pydantic import Field, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
    # Required settings
    acs_connection_string: str = Field(
        ...,
        alias="ACS_CONNECTION_STRING",
        description="Azure Communication Services connection string"
    )
    
    public_base: str = Field(
        ..., 
        alias="PUBLIC_BASE",
        description="Public base URL for webhooks (e.g., ngrok tunnel)"
    )
    
    # Optional settings
    stt_model_path: Optional[str] = Field(
        default=None,
        alias="STT_MODEL_PATH", 
        description="Path to Vosk STT model directory"
    )
    
    # Piper TTS settings
    piper_voice_path: Optional[str] = Field(
        default="./voices/en-us-high.onnx",
        alias="PIPER_VOICE_PATH",
        description="Path to Piper voice model (.onnx file)"
    )
    
    piper_length_scale: float = Field(
        default=1.08,
        alias="PIPER_LENGTH_SCALE",
        description="Piper speech speed multiplier (1.0 = normal, >1.0 = slower)"
    )
    
    piper_noise_scale: float = Field(
        default=0.65,
        alias="PIPER_NOISE_SCALE", 
        description="Piper speech variability (0.0-1.0, higher = more variable)"
    )
    
    piper_noise_w: float = Field(
        default=0.80,
        alias="PIPER_NOISE_W",
        description="Piper variance in speech timing (0.0-1.0)"
    )
    
    piper_sentence_silence: float = Field(
        default=0.25,
        alias="PIPER_SENTENCE_SILENCE",
        description="Piper pause between sentences in seconds"
    )
    
    # Ollama LLM settings
    ollama_base_url: AnyHttpUrl = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Ollama server base URL"
    )
    
    ollama_model: str = Field(
        default="llama3.1:8b",
        alias="OLLAMA_MODEL",
        description="Ollama model name"
    )
    
    ollama_temperature: float = Field(
        default=0.4,
        alias="OLLAMA_TEMPERATURE",
        description="LLM temperature (0.0-1.0, higher = more random)"
    )
    
    ollama_top_p: float = Field(
        default=0.9,
        alias="OLLAMA_TOP_P", 
        description="LLM top-p sampling (0.0-1.0)"
    )
    
    ollama_seed: Optional[int] = Field(
        default=None,
        alias="OLLAMA_SEED",
        description="LLM random seed for reproducible outputs"
    )
    
    ollama_max_tokens: int = Field(
        default=512,
        alias="OLLAMA_MAX_TOKENS",
        description="Maximum tokens to generate"
    )
    
    ollama_stop: List[str] = Field(
        default_factory=list,
        alias="OLLAMA_STOP",
        description="Stop sequences for generation (comma-separated in env)"
    )
    
    # Persona configuration
    persona_path: str = Field(
        default="config/persona.yaml",
        alias="PERSONA_PATH",
        description="Path to persona configuration YAML file"
    )
    
    @field_validator("acs_connection_string")
    @classmethod
    def validate_acs_connection_string(cls, v: str) -> str:
        """Validate ACS connection string format."""
        if not v.startswith("endpoint=") or "accesskey=" not in v:
            raise ValueError("ACS_CONNECTION_STRING must be in format: endpoint=...;accesskey=...")
        return v
        
    @field_validator("public_base")
    @classmethod
    def validate_public_base(cls, v: str) -> str:
        """Validate public base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("PUBLIC_BASE must start with http:// or https://")
        return v.rstrip("/")
        
    @field_validator("stt_model_path", mode="before")
    @classmethod
    def validate_stt_model_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate STT model path exists if provided."""
        if v and v.strip():
            path = v.strip()
            if not os.path.isdir(path):
                print(f"Warning: STT_MODEL_PATH '{path}' does not exist - STT will be disabled")
                return None
            return path
        return None
        
    @field_validator("ollama_stop", mode="before")
    @classmethod
    def validate_ollama_stop(cls, v) -> List[str]:
        """Parse comma-separated stop sequences."""
        if isinstance(v, str):
            return parse_comma_separated_list(v)
        elif isinstance(v, list):
            return v
        return []


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


def parse_comma_separated_list(value: str) -> List[str]:
    """Parse comma-separated string into list of strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]