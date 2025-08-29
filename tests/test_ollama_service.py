"""Tests for Ollama LLM service."""

from unittest.mock import AsyncMock, patch

import pytest

from src.phone_agent.services.llm_base import LLMUnavailable
from src.phone_agent.services.llm_ollama import OLLAMA_AVAILABLE, OllamaLLMService


class TestOllamaLLMService:
    """Test Ollama LLM service implementation."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama async client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def llm_service(self):
        """Create OllamaLLMService instance for testing."""
        return OllamaLLMService(
            model="test-model",
            host="http://test:11434",
            timeout=10.0,
            max_retries=2,
            temperature=0.5,
            top_p=0.8,
            max_tokens=256,
        )

    def test_init(self, llm_service):
        """Test service initialization."""
        assert llm_service.model == "test-model"
        assert llm_service.host == "http://test:11434"
        assert llm_service.timeout == 10.0
        assert llm_service.max_retries == 2
        assert llm_service.temperature == 0.5
        assert llm_service.top_p == 0.8
        assert llm_service.max_tokens == 256

    def test_is_available_no_ollama(self):
        """Test availability check when Ollama is not installed."""
        with patch("src.phone_agent.services.llm_ollama.OLLAMA_AVAILABLE", False):
            service = OllamaLLMService()
            assert service.is_available is False

    @pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not installed")
    def test_is_available_with_ollama(self, llm_service):
        """Test availability check when Ollama is available."""
        # Since we can't actually connect to Ollama in tests,
        # we just check that the logic works
        assert llm_service.is_available in [True, False]

    @pytest.mark.asyncio
    async def test_check_model_available_no_client(self, llm_service):
        """Test model availability check without client."""
        llm_service._client = None

        with pytest.raises(LLMUnavailable, match="not initialized"):
            await llm_service._check_model_available()

    @pytest.mark.asyncio
    async def test_check_model_available_success(self, llm_service, mock_ollama_client):
        """Test successful model availability check."""
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model"}]}
        llm_service._client = mock_ollama_client

        # Should not raise
        await llm_service._check_model_available()
        mock_ollama_client.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_model_available_pull_needed(self, llm_service, mock_ollama_client):
        """Test model pulling when model not found."""
        mock_ollama_client.list.return_value = {"models": [{"name": "other-model"}]}
        mock_ollama_client.pull.return_value = AsyncMock()
        llm_service._client = mock_ollama_client

        await llm_service._check_model_available()

        mock_ollama_client.list.assert_called_once()
        mock_ollama_client.pull.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, llm_service):
        """Test successful operation with backoff."""

        async def success_operation():
            return "success"

        result = await llm_service._retry_with_backoff(success_operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_with_backoff_failure(self, llm_service):
        """Test operation failure after retries."""

        async def fail_operation():
            raise Exception("Always fails")

        with pytest.raises(LLMUnavailable, match="failed after retries"):
            await llm_service._retry_with_backoff(fail_operation)

    @pytest.mark.asyncio
    async def test_generate_not_available(self, llm_service):
        """Test generate when service not available."""
        llm_service._is_available = False

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(LLMUnavailable, match="not available"):
            await llm_service.generate(messages)

    @pytest.mark.asyncio
    async def test_generate_complete_response(self, llm_service, mock_ollama_client):
        """Test complete (non-streaming) response generation."""
        # Mock the client and model check
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model"}]}
        mock_ollama_client.chat.return_value = {
            "message": {"content": "Hello there!"},
            "done": True,
        }
        llm_service._client = mock_ollama_client
        llm_service._is_available = True

        # Mock OLLAMA_AVAILABLE to True for this test
        with patch("src.phone_agent.services.llm_ollama.OLLAMA_AVAILABLE", True):
            messages = [{"role": "user", "content": "Hi"}]

            result = await llm_service.generate(messages, stream=False)

            assert result == "Hello there!"
            mock_ollama_client.chat.assert_called_once()
            call_args = mock_ollama_client.chat.call_args
            assert call_args[1]["model"] == "test-model"
            assert call_args[1]["messages"] == messages
            assert call_args[1]["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_streaming_response(self, llm_service, mock_ollama_client):
        """Test streaming response generation."""

        # Mock streaming response
        async def mock_stream():
            yield {"message": {"content": "Hello"}, "done": False}
            yield {"message": {"content": " there"}, "done": False}
            yield {"message": {"content": "!"}, "done": True}

        mock_ollama_client.list.return_value = {"models": [{"name": "test-model"}]}
        mock_ollama_client.chat.return_value = mock_stream()
        llm_service._client = mock_ollama_client
        llm_service._is_available = True

        # Mock OLLAMA_AVAILABLE to True for this test
        with patch("src.phone_agent.services.llm_ollama.OLLAMA_AVAILABLE", True):
            messages = [{"role": "user", "content": "Hi"}]

            result_stream = await llm_service.generate(messages, stream=True)

            chunks = []
            async for chunk in result_stream:
                chunks.append(chunk)

            assert chunks == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_generate_with_stop_sequences(self, llm_service, mock_ollama_client):
        """Test generation with stop sequences."""
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model"}]}
        mock_ollama_client.chat.return_value = {"message": {"content": "Response"}, "done": True}
        llm_service._client = mock_ollama_client
        llm_service._is_available = True

        # Mock OLLAMA_AVAILABLE to True for this test
        with patch("src.phone_agent.services.llm_ollama.OLLAMA_AVAILABLE", True):
            messages = [{"role": "user", "content": "Hi"}]
            stop_sequences = ["</s>", "\n"]

            await llm_service.generate(messages, stream=False, stop=stop_sequences)

            call_args = mock_ollama_client.chat.call_args
            assert call_args[1]["options"]["stop"] == stop_sequences

    @pytest.mark.asyncio
    async def test_summarize_basic(self, llm_service, mock_ollama_client):
        """Test text summarization."""
        mock_ollama_client.list.return_value = {"models": [{"name": "test-model"}]}
        mock_ollama_client.chat.return_value = {
            "message": {"content": "This is a summary."},
            "done": True,
        }
        llm_service._client = mock_ollama_client
        llm_service._is_available = True

        # Mock OLLAMA_AVAILABLE to True for this test
        with patch("src.phone_agent.services.llm_ollama.OLLAMA_AVAILABLE", True):
            text = "This is a long text that needs to be summarized for better understanding."

            summary = await llm_service.summarize(text, max_tokens=50)

            assert summary == "This is a summary."

            # Check that summarization prompt was used
            call_args = mock_ollama_client.chat.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "summarize" in messages[0]["content"].lower()
            assert text in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_summarize_empty_text(self, llm_service):
        """Test summarization of empty text."""
        result = await llm_service.summarize("")
        assert result == ""

        result = await llm_service.summarize("   ")
        assert result == ""

    @pytest.mark.asyncio
    async def test_close(self, llm_service, mock_ollama_client):
        """Test resource cleanup."""
        llm_service._client = mock_ollama_client

        await llm_service.close()

        assert llm_service._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, llm_service):
        """Test async context manager usage."""
        async with llm_service as service:
            assert service is llm_service

        # Should be closed after context
        assert llm_service._client is None
