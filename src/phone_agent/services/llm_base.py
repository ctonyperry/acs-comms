"""Base LLM interface protocol definitions."""

from collections.abc import AsyncIterator
from typing import Protocol

# Type alias for message format
Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLMUnavailable(Exception):
    """Exception raised when LLM service is unavailable."""

    pass


class LLM(Protocol):
    """Protocol for Language Learning Model implementations."""

    async def generate(
        self,
        messages: list[Message],
        stream: bool = True,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str] | str:
        """Generate response from messages.

        Args:
            messages: List of message dictionaries with role and content
            stream: Whether to stream response (True) or return complete response (False)
            stop: Optional list of stop sequences to halt generation

        Returns:
            AsyncIterator[str] if stream=True (yields non-empty deltas)
            str if stream=False (complete response)

        Raises:
            LLMUnavailable: When LLM service is not available
        """
        ...

    async def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Summarize the given text.

        Args:
            text: Text to summarize
            max_tokens: Maximum tokens in summary

        Returns:
            Summarized text

        Raises:
            LLMUnavailable: When LLM service is not available
        """
        ...


def assemble_messages(persona_system: str, history: list[Message], user_text: str) -> list[Message]:
    """Assemble messages for LLM generation.

    Args:
        persona_system: System prompt with persona instructions
        history: Previous conversation messages
        user_text: Current user input

    Returns:
        Complete message list ready for LLM generation
    """
    return [
        {"role": "system", "content": persona_system},
        *history,
        {"role": "user", "content": user_text},
    ]
