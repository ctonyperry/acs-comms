"""Ollama LLM service implementation with streaming support."""

import json
import logging
from typing import Any, AsyncGenerator

import aiohttp
import yaml
from pydantic import BaseModel

from .guardrails import GuardrailsService
from .llm_base import BaseLLMService, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    """Ollama-specific configuration."""
    
    base_url: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int | None = None
    max_tokens: int = 2048
    stop: list[str] = []


class PersonaConfig(BaseModel):
    """Persona configuration from YAML."""
    
    persona: dict[str, Any]
    constraints: dict[str, Any]
    formatting: dict[str, Any] = {}
    privacy: dict[str, Any] = {}


class OllamaLLMService(BaseLLMService):
    """Ollama LLM service with streaming support and guardrails."""
    
    def __init__(
        self, 
        config: OllamaConfig,
        persona_config_path: str | None = None,
        guardrails_service: GuardrailsService | None = None
    ):
        """Initialize Ollama LLM service.
        
        Args:
            config: Ollama configuration
            persona_config_path: Path to persona YAML configuration
            guardrails_service: Guardrails service for safety
        """
        self.config = config
        self.guardrails = guardrails_service or GuardrailsService()
        self.persona_config = self._load_persona_config(persona_config_path)
        self._persona_system_message = self._build_persona_system_message()
        
    def _load_persona_config(self, config_path: str | None) -> PersonaConfig | None:
        """Load persona configuration from YAML file.
        
        Args:
            config_path: Path to persona YAML file
            
        Returns:
            PersonaConfig object or None if file not found/invalid
        """
        if not config_path:
            return None
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return PersonaConfig(**data)
        except Exception as e:
            logger.warning(f"Could not load persona config from {config_path}: {e}")
            return None
    
    def _build_persona_system_message(self) -> str:
        """Build system message from persona configuration.
        
        Returns:
            Formatted system message with persona and constraints
        """
        if not self.persona_config:
            return ""
            
        parts = []
        
        # Add persona description and role
        persona = self.persona_config.persona
        if "role" in persona:
            parts.append(persona["role"])
        
        # Add traits
        if "traits" in persona and persona["traits"]:
            traits_text = "Key personality traits:\n" + "\n".join(f"- {trait}" for trait in persona["traits"])
            parts.append(traits_text)
        
        # Add guidelines
        if "guidelines" in persona and persona["guidelines"]:
            guidelines_text = "Behavioral guidelines:\n" + "\n".join(f"- {guideline}" for guideline in persona["guidelines"])
            parts.append(guidelines_text)
        
        # Add safety constraints
        constraints = self.persona_config.constraints
        if "safety_rules" in constraints:
            safety_text = "CRITICAL SAFETY RULES (NEVER VIOLATE):\n" + "\n".join(f"- {rule}" for rule in constraints["safety_rules"])
            parts.append(safety_text)
        
        # Add communication rules
        if "communication_rules" in constraints:
            comm_text = "Communication rules:\n" + "\n".join(f"- {rule}" for rule in constraints["communication_rules"])
            parts.append(comm_text)
        
        return "\n\n".join(parts)
    
    @property
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        # For now, assume available if we have a config
        # Could add actual health check in production
        return bool(self.config.base_url and self.config.model)
    
    async def _make_ollama_request(self, payload: dict[str, Any], stream: bool = True) -> aiohttp.ClientResponse:
        """Make request to Ollama API.
        
        Args:
            payload: Request payload
            stream: Whether to stream the response
            
        Returns:
            HTTP response object
            
        Raises:
            aiohttp.ClientError: If request fails
        """
        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        
        async with aiohttp.ClientSession() as session:
            try:
                response = await session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response
            except Exception as e:
                logger.error(f"Ollama API request failed: {e}")
                raise
    
    def _format_messages_for_ollama(self, messages: list[LLMMessage]) -> str:
        """Format messages for Ollama prompt format.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Formatted prompt string
        """
        # Apply persona to system message
        enhanced_messages = self._prepare_system_message(messages, self._persona_system_message)
        
        # Convert to simple prompt format for Ollama
        prompt_parts = []
        
        for msg in enhanced_messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        # Add prompt for next response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def generate(
        self, 
        messages: list[LLMMessage], 
        stream: bool = True
    ) -> AsyncGenerator[str, None] | LLMResponse:
        """Generate text from messages with optional streaming.
        
        Args:
            messages: List of conversation messages
            stream: Whether to stream responses
            
        Returns:
            If stream=True: AsyncGenerator yielding text chunks
            If stream=False: Complete LLMResponse
            
        Raises:
            Exception: If generation fails
        """
        # Validate input messages with guardrails
        for msg in messages:
            self.guardrails.validate_input(msg.content, msg.role)
        
        # Format prompt
        prompt = self._format_messages_for_ollama(messages)
        
        # Prepare Ollama request payload
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            }
        }
        
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed
            
        if self.config.stop:
            payload["options"]["stop"] = self.config.stop
        
        if stream:
            return self._stream_generate(payload)
        else:
            return await self._complete_generate(payload)
    
    async def _stream_generate(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream text generation from Ollama.
        
        Args:
            payload: Request payload
            
        Yields:
            Text chunks from generation
        """
        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        full_response = ""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        if not line.strip():
                            continue
                            
                        try:
                            data = json.loads(line.decode('utf-8'))
                            
                            if "response" in data:
                                chunk = data["response"]
                                full_response += chunk
                                
                                # Only yield non-empty chunks
                                if chunk:  
                                    # Validate chunk with guardrails (non-strict for streaming)
                                    if self.guardrails.is_safe_output(chunk):
                                        yield chunk
                                    else:
                                        logger.warning("Blocked unsafe chunk in stream")
                                        # Could yield empty string or warning message
                                        continue
                            
                            # Check if generation is complete
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {line}")
                            continue
            
            # Final validation of complete response
            try:
                self.guardrails.validate_output(full_response)
            except Exception as e:
                logger.warning(f"Complete response failed guardrails: {e}")
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # Yield error message or re-raise depending on requirements
            yield f"[Error: Generation failed - {str(e)}]"
    
    async def _complete_generate(self, payload: dict[str, Any]) -> LLMResponse:
        """Complete text generation from Ollama.
        
        Args:
            payload: Request payload
            
        Returns:
            Complete LLMResponse object
        """
        payload["stream"] = False
        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            content = data.get("response", "")
            
            # Validate output with guardrails
            safe_content = self.guardrails.validate_output(content)
            
            return LLMResponse(
                content=safe_content,
                finish_reason=data.get("done_reason"),
                tokens_used=data.get("eval_count")
            )
            
        except Exception as e:
            logger.error(f"Complete generation failed: {e}")
            # Return error response
            return LLMResponse(
                content=f"[Error: Generation failed - {str(e)}]",
                finish_reason="error"
            )
    
    async def summarize(self, text: str) -> str:
        """Summarize the given text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        # Validate input
        self.guardrails.validate_input(text, "user")
        
        # Create summarization messages
        messages = [
            LLMMessage(
                role="system", 
                content="You are a helpful assistant that provides clear, concise summaries of text."
            ),
            LLMMessage(
                role="user", 
                content=f"Please provide a concise summary of the following text:\n\n{text}"
            )
        ]
        
        # Generate summary (non-streaming for simplicity)
        response = await self.generate(messages, stream=False)
        
        if isinstance(response, LLMResponse):
            return response.content
        else:
            # Should not happen, but handle gracefully
            logger.error("Unexpected response type from generate()")
            return "Summary generation failed."