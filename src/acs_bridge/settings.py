"""Application settings management using Pydantic."""

import os
from typing import Optional

from pydantic import Field, field_validator
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


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()