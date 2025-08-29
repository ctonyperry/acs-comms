"""Ollama LLM service implementation with streaming and retry logic."""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, List, Optional
from urllib.parse import urljoin

import aiohttp

from .llm_base import LLM, Message

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """Exception raised when LLM service is unavailable."""
    pass


class OllamaLLM:
    """Ollama LLM service with streaming support and retry logic."""
    
    def __init__(self, settings, session: Optional[aiohttp.ClientSession] = None) -> None:
        """Initialize Ollama LLM service.
        
        Args:
            settings: Application settings with OLLAMA_* configuration
            session: Optional aiohttp session (will create if not provided)
        """
        self.settings = settings
        self._session = session
        self._owned_session = session is None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def _close_session(self) -> None:
        """Close session if we own it."""
        if self._owned_session and self._session:
            await self._session.close()
            self._session = None
    
    async def generate(
        self,
        messages: List[Message],
        stream: bool = True,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str] | str:
        """Generate response from messages with streaming support.
        
        Args:
            messages: List of message dictionaries
            stream: If True, yield text chunks; else return full string
            stop: Optional stop sequences
            
        Returns:
            AsyncIterator[str] if streaming, str if not streaming
            
        Raises:
            LLMUnavailable: If service is unavailable after retries
        """
        if stream:
            return self._generate_stream(messages, stop)
        else:
            return await self._generate_non_stream(messages, stop)
    
    async def _generate_non_stream(
        self, 
        messages: List[Message], 
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate complete response without streaming."""
        full_response = ""
        async for chunk in self._generate_stream(messages, stop):
            full_response += chunk
        return full_response
    
    async def _generate_stream(
        self, 
        messages: List[Message], 
        stop: Optional[List[str]] = None
    ) -> AsyncIterator[str]:
        """Internal streaming generation with retry logic."""
        start_time = time.time()
        retries = 0
        max_retries = 3
        base_delay = 1.0
        
        # Convert stop list to format expected by Ollama
        stop_sequences = stop or self.settings.OLLAMA_STOP
        
        # Log generation start
        logger.info(
            "LLM generation started",
            extra={
                "evt": "llm_start",
                "model": self.settings.OLLAMA_MODEL,
                "temp": self.settings.OLLAMA_TEMPERATURE,
                "top_p": self.settings.OLLAMA_TOP_P,
                "max_tokens": self.settings.OLLAMA_MAX_TOKENS,
                "stop": stop_sequences,
            }
        )
        
        while retries <= max_retries:
            try:
                session = await self._get_session()
                url = urljoin(str(self.settings.OLLAMA_BASE_URL), "/api/generate")
                
                # Prepare request payload
                payload = {
                    "model": self.settings.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.settings.OLLAMA_TEMPERATURE,
                        "top_p": self.settings.OLLAMA_TOP_P,
                        "num_predict": self.settings.OLLAMA_MAX_TOKENS,
                    }
                }
                
                if self.settings.OLLAMA_SEED is not None:
                    payload["options"]["seed"] = self.settings.OLLAMA_SEED
                    
                if stop_sequences:
                    payload["options"]["stop"] = stop_sequences
                
                timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
                
                async with session.post(url, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        if response.status in (502, 503, 504):
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )
                        raise LLMUnavailable(f"Ollama API error: {response.status}")
                    
                    # Stream the response
                    collected_text = ""
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                            
                        try:
                            data = json.loads(line)
                            
                            if data.get("done", False):
                                # Generation complete
                                elapsed_ms = int((time.time() - start_time) * 1000)
                                logger.info(
                                    "LLM generation completed",
                                    extra={
                                        "evt": "llm_done",
                                        "model": self.settings.OLLAMA_MODEL,
                                        "elapsed_ms": elapsed_ms,
                                        "retries": retries,
                                    }
                                )
                                return
                                
                            if "response" in data:
                                chunk = data["response"]
                                if chunk:  # Only yield non-empty chunks
                                    collected_text += chunk
                                    
                                    # Check for stop sequences
                                    if stop_sequences:
                                        for stop_seq in stop_sequences:
                                            if stop_seq in collected_text:
                                                # Find the position and truncate
                                                stop_pos = collected_text.find(stop_seq)
                                                final_chunk = collected_text[:stop_pos]
                                                if final_chunk:
                                                    logger.info(
                                                        "LLM chunk yielded",
                                                        extra={
                                                            "evt": "llm_chunk", 
                                                            "chunk_len": len(final_chunk),
                                                            "stopped": True,
                                                        }
                                                    )
                                                    yield final_chunk
                                                return
                                    
                                    logger.debug(
                                        "LLM chunk yielded",
                                        extra={
                                            "evt": "llm_chunk",
                                            "chunk_len": len(chunk),
                                        }
                                    )
                                    yield chunk
                                    
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON line: {line}")
                            continue
                            
                # If we get here, stream ended normally
                return
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retries += 1
                if retries > max_retries:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    logger.error(
                        "LLM generation failed after retries",
                        extra={
                            "evt": "llm_error",
                            "error": str(e),
                            "retries": retries - 1,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
                    raise LLMUnavailable(f"Ollama API unavailable after {max_retries} retries: {e}")
                
                # Exponential backoff
                delay = base_delay * (2 ** (retries - 1))
                logger.warning(
                    f"LLM request failed, retrying in {delay}s",
                    extra={
                        "evt": "llm_retry",
                        "retry": retries,
                        "delay_s": delay,
                        "error": str(e),
                    }
                )
                await asyncio.sleep(delay)
                
    async def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens in summary
            
        Returns:
            Short summary suitable for call wrap-up notes
        """
        summary_prompt = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that creates concise summaries. "
                          "Create a brief, professional summary suitable for call notes."
            },
            {
                "role": "user",
                "content": f"Please summarize this conversation or text in {max_tokens} tokens or less:\n\n{text}"
            }
        ]
        
        # Use non-streaming generation for summaries
        return await self.generate(summary_prompt, stream=False)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup session."""
        await self._close_session()