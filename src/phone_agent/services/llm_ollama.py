"""Ollama LLM service implementation with streaming support."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from .llm_base import LLMUnavailable, Message

logger = logging.getLogger(__name__)

# Optional Ollama imports
try:
    import ollama
    from ollama import AsyncClient

    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    AsyncClient = None
    OLLAMA_AVAILABLE = False


class OllamaLLMService:
    """Ollama-based LLM service with streaming support."""

    def __init__(
        self,
        model: str = "llama3.2:1b",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> None:
        """Initialize Ollama LLM service.

        Args:
            model: Ollama model name to use
            host: Ollama server host URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            temperature: Sampling temperature (0.0-2.0)
            top_p: Top-p sampling parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self._client: AsyncClient | None = None
        self._is_available: bool | None = None

        if OLLAMA_AVAILABLE:
            self._client = AsyncClient(host=host, timeout=timeout)
            logger.info(f"Ollama LLM service initialized - Model: {model}, Host: {host}")
        else:
            logger.warning("Ollama not available - 'ollama' package not installed")

    @property
    def is_available(self) -> bool:
        """Check if Ollama LLM service is available."""
        if not OLLAMA_AVAILABLE or not self._client:
            return False

        if self._is_available is not None:
            return self._is_available

        # For testing purposes and when we can't async check,
        # assume available if client was created successfully
        if self._client is not None:
            self._is_available = True
            return True

        self._is_available = False
        return False

    async def _check_model_available(self) -> None:
        """Check if the specified model is available."""
        if not self._client:
            raise LLMUnavailable("Ollama client not initialized")

        try:
            models = await self._client.list()
            available_models = [model["name"] for model in models.get("models", [])]

            if self.model not in available_models:
                logger.warning(f"Model {self.model} not found. Available: {available_models}")
                # Try to pull the model
                logger.info(f"Attempting to pull model: {self.model}")
                await self._client.pull(self.model)
                logger.info(f"Successfully pulled model: {self.model}")

        except Exception as e:
            logger.error(f"Failed to check/pull model {self.model}: {e}")
            raise LLMUnavailable(f"Model {self.model} not available: {e}") from e

    async def _retry_with_backoff(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == self.max_retries:
                    logger.error(f"Operation failed after {self.max_retries + 1} attempts: {e}")
                    break

                delay = self.retry_delay * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise LLMUnavailable(f"Operation failed after retries: {last_exception}")

    async def generate(
        self,
        messages: list[Message],
        stream: bool = True,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str] | str:
        """Generate response from messages.

        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            stop: Optional stop sequences

        Returns:
            AsyncIterator[str] if stream=True, str if stream=False

        Raises:
            LLMUnavailable: When service is not available
        """
        if not self.is_available:
            raise LLMUnavailable("Ollama service not available")

        # Ensure model is available
        await self._check_model_available()

        # Prepare options
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
        }

        if stop:
            options["stop"] = stop

        logger.debug(f"Generating with model {self.model}, stream={stream}")

        if stream:
            return self._generate_stream(messages, options)
        else:
            return await self._generate_complete(messages, options)

    async def _generate_stream(
        self, messages: list[Message], options: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Generate streaming response."""

        async def _stream_operation():
            stream = await self._client.chat(
                model=self.model, messages=messages, stream=True, options=options
            )
            return stream

        try:
            stream = await self._retry_with_backoff(_stream_operation)

            async for chunk in stream:
                content = chunk.get("message", {}).get("content", "")

                # Only yield non-empty deltas as required
                if content:
                    yield content

                # Check if generation is done
                if chunk.get("done", False):
                    break

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise LLMUnavailable(f"Streaming failed: {e}") from e

    async def _generate_complete(self, messages: list[Message], options: dict[str, Any]) -> str:
        """Generate complete response."""

        async def _generate_operation():
            response = await self._client.chat(
                model=self.model, messages=messages, stream=False, options=options
            )
            return response

        try:
            response = await self._retry_with_backoff(_generate_operation)
            content = response.get("message", {}).get("content", "")

            if not content:
                logger.warning("Received empty response from Ollama")

            return content

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise LLMUnavailable(f"Generation failed: {e}") from e

    async def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Summarize the given text.

        Args:
            text: Text to summarize
            max_tokens: Maximum tokens in summary

        Returns:
            Summarized text
        """
        if not text.strip():
            return ""

        # Create summarization prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise summaries. Summarize the following text in a clear and brief manner.",
            },
            {"role": "user", "content": f"Please summarize this text:\n\n{text}"},
        ]

        # Use reduced max_tokens for summary
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens

        try:
            # Always use non-streaming for summaries
            summary = await self.generate(messages, stream=False)
            logger.debug(f"Generated summary of {len(text)} chars -> {len(summary)} chars")
            return summary
        finally:
            # Restore original max_tokens
            self.max_tokens = original_max_tokens

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            # Ollama client doesn't have explicit close method
            self._client = None
            logger.info("Ollama LLM service closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
