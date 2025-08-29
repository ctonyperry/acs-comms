"""Tests for Ollama LLM streaming functionality."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import aiohttp

from src.phone_agent.services.llm_ollama import OllamaLLM, LLMUnavailable


class MockSettings:
    """Mock settings for testing."""
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_TEMPERATURE = 0.4
    OLLAMA_TOP_P = 0.9
    OLLAMA_SEED = None
    OLLAMA_MAX_TOKENS = 512
    OLLAMA_STOP = ["</s>"]


class MockResponse:
    """Mock aiohttp response for testing."""
    
    def __init__(self, status=200, content_lines=None):
        self.status = status
        self.request_info = MagicMock()
        self.history = []
        self._content_lines = content_lines or []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    @property
    def content(self):
        """Mock content stream."""
        mock_content = MagicMock()
        
        async def mock_iter(*args):
            for line in self._content_lines:
                yield line.encode('utf-8')
                
        mock_content.__aiter__ = mock_iter
        return mock_content


@pytest.mark.asyncio
class TestOllamaLLMStreaming:
    """Test cases for OllamaLLM streaming functionality."""
    
    async def test_stream_two_chunks_plus_done(self):
        """Test streaming with two chunks followed by done signal."""
        settings = MockSettings()
        
        # Mock response content: two chunks + done
        content_lines = [
            '{"response": "foo"}',
            '{"response": "bar"}', 
            '{"done": true}'
        ]
        
        mock_response = MockResponse(content_lines=content_lines)
        
        # Setup mock session that returns the mock response
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        
        llm = OllamaLLM(settings, session=mock_session)
        messages = [{"role": "user", "content": "test"}]
        
        # Get the async iterator and collect chunks
        response = await llm.generate(messages, stream=True)
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            
        assert chunks == ["foo", "bar"]
        
    async def test_stop_sequence_truncation(self):
        """Test that stop sequences properly truncate output."""
        settings = MockSettings()
        settings.OLLAMA_STOP = ["STOP"]
        
        # Mock response that should be truncated at STOP
        content_lines = [
            '{"response": "This is some text STOP and this should not appear"}',
            '{"done": true}'
        ]
        
        mock_response = MockResponse(content_lines=content_lines)
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        
        llm = OllamaLLM(settings, session=mock_session)
        messages = [{"role": "user", "content": "test"}]
        
        # Collect chunks
        chunks = []
        async for chunk in llm.generate(messages, stream=True):
            chunks.append(chunk)
            
        # Should only get text before STOP
        assert chunks == ["This is some text "]
        
    async def test_retry_on_502_error(self):
        """Test retry behavior on 502 error followed by success."""
        settings = MockSettings()
        
        # First call returns 502, second succeeds
        call_count = 0
        
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: 502 error
                mock_resp = MagicMock()
                mock_resp.status = 502
                mock_resp.request_info = MagicMock()
                mock_resp.history = []
                
                raise aiohttp.ClientResponseError(
                    request_info=mock_resp.request_info,
                    history=mock_resp.history,
                    status=502
                )
            else:
                # Second call: success
                content_lines = ['{"response": "success"}', '{"done": true}']
                return MockResponse(content_lines=content_lines)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        
        llm = OllamaLLM(settings, session=mock_session)
        messages = [{"role": "user", "content": "test"}]
        
        # Should succeed after retry
        chunks = []
        async for chunk in llm.generate(messages, stream=True):
            chunks.append(chunk)
            
        assert chunks == ["success"]
        assert call_count == 2  # Verify retry happened
        
    async def test_failure_after_max_retries(self):
        """Test LLMUnavailable exception after repeated failures."""
        settings = MockSettings()
        
        async def mock_post(*args, **kwargs):
            # Always return 502 error
            mock_resp = MagicMock()
            mock_resp.status = 502
            mock_resp.request_info = MagicMock()
            mock_resp.history = []
            
            raise aiohttp.ClientResponseError(
                request_info=mock_resp.request_info,
                history=mock_resp.history,
                status=502
            )
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        
        llm = OllamaLLM(settings, session=mock_session)
        messages = [{"role": "user", "content": "test"}]
        
        # Should raise LLMUnavailable after max retries
        with pytest.raises(LLMUnavailable, match="unavailable after 3 retries"):
            chunks = []
            async for chunk in llm.generate(messages, stream=True):
                chunks.append(chunk)
                
    async def test_empty_chunks_filtered(self):
        """Test that empty response chunks are filtered out."""
        settings = MockSettings()
        
        # Mock response with empty chunks mixed in
        content_lines = [
            '{"response": ""}',  # Empty chunk
            '{"response": "hello"}',
            '{"response": ""}',  # Empty chunk  
            '{"response": " world"}',
            '{"done": true}'
        ]
        
        mock_response = MockResponse(content_lines=content_lines)
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        
        llm = OllamaLLM(settings, session=mock_session)
        messages = [{"role": "user", "content": "test"}]
        
        # Collect chunks
        chunks = []
        async for chunk in llm.generate(messages, stream=True):
            chunks.append(chunk)
            
        # Should only get non-empty chunks
        assert chunks == ["hello", " world"]
        
    async def test_non_streaming_mode(self):
        """Test non-streaming mode returns complete response."""
        settings = MockSettings()
        
        content_lines = [
            '{"response": "Hello"}',
            '{"response": " world"}',
            '{"done": true}'
        ]
        
        mock_response = MockResponse(content_lines=content_lines)
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        
        llm = OllamaLLM(settings, session=mock_session)
        messages = [{"role": "user", "content": "test"}]
        
        # Non-streaming mode
        result = await llm.generate(messages, stream=False)
        
        assert result == "Hello world"
        assert isinstance(result, str)
        
    async def test_summarize_method(self):
        """Test summarize method uses correct prompt format."""
        settings = MockSettings()
        
        content_lines = [
            '{"response": "Brief summary here"}',
            '{"done": true}'
        ]
        
        mock_response = MockResponse(content_lines=content_lines)
        mock_session = AsyncMock()
        mock_session.post.return_value = mock_response
        
        llm = OllamaLLM(settings, session=mock_session)
        
        result = await llm.summarize("Long text to summarize", max_tokens=100)
        
        assert result == "Brief summary here"
        
        # Verify correct prompt structure was used
        call_args = mock_session.post.call_args
        payload = call_args[1]['json']
        messages = payload['messages']
        
        # Should have system message and user message
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'summary' in messages[0]['content'].lower()
        assert messages[1]['role'] == 'user'
        assert 'summarize' in messages[1]['content'].lower()
        
    async def test_context_manager(self):
        """Test async context manager properly cleans up session."""
        settings = MockSettings()
        
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        
        async with OllamaLLM(settings) as llm:
            # Should create its own session
            assert llm._owned_session is True
            
        # Session should be closed when exiting context
        # (Can't easily test the close call since we create our own session)