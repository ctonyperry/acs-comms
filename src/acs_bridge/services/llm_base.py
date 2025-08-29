"""Large Language Model base interface definition."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Protocol

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """LLM message format."""
    
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """LLM response format."""
    
    content: str
    finish_reason: str | None = None
    tokens_used: int | None = None


class LLMService(Protocol):
    """Protocol for Large Language Model services."""

    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        ...

    async def generate(
        self, 
        messages: list[LLMMessage], 
        stream: bool = True
    ) -> AsyncGenerator[str, None] | LLMResponse:
        """Generate text from messages.
        
        Args:
            messages: List of conversation messages
            stream: Whether to stream responses
            
        Returns:
            If stream=True: AsyncGenerator yielding text chunks
            If stream=False: Complete LLMResponse
        """
        ...

    async def summarize(self, text: str) -> str:
        """Summarize the given text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        ...


class BaseLLMService(ABC):
    """Base class for LLM service implementations."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        pass

    @abstractmethod
    async def generate(
        self, 
        messages: list[LLMMessage], 
        stream: bool = True
    ) -> AsyncGenerator[str, None] | LLMResponse:
        """Generate text from messages."""
        pass

    @abstractmethod
    async def summarize(self, text: str) -> str:
        """Summarize the given text."""
        pass

    def _prepare_system_message(self, messages: list[LLMMessage], persona_content: str) -> list[LLMMessage]:
        """Prepend persona to system message or create new system message.
        
        Args:
            messages: Original messages
            persona_content: Persona and constraints to prepend
            
        Returns:
            Messages with persona prepended to system message
        """
        if not persona_content.strip():
            return messages
            
        # Find first system message and separate other messages
        system_message = None
        other_messages = []
        
        for msg in messages:
            if msg.role == "system" and system_message is None:
                system_message = msg  # Take only the first system message
            elif msg.role != "system":
                other_messages.append(msg)
        
        # Prepare enhanced system message
        if system_message:
            enhanced_content = f"{persona_content}\n\n{system_message.content}"
        else:
            enhanced_content = persona_content
            
        enhanced_system = LLMMessage(role="system", content=enhanced_content)
        return [enhanced_system] + other_messages