"""Tests for LLM base service."""

import pytest

from src.acs_bridge.services.llm_base import LLMMessage, LLMResponse, BaseLLMService


class MockLLMService(BaseLLMService):
    """Mock LLM service for testing base functionality."""
    
    def __init__(self, available: bool = True):
        self._available = available
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    async def generate(self, messages, stream=True):
        if stream:
            async def mock_stream():
                yield "Hello"
                yield " there!"
            return mock_stream()
        else:
            return LLMResponse(content="Hello there!", finish_reason="stop")
    
    async def summarize(self, text: str) -> str:
        return f"Summary of: {text[:50]}..."


class TestLLMMessage:
    """Test LLM message model."""
    
    def test_create_valid_message(self):
        """Test creating a valid LLM message."""
        msg = LLMMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
    
    def test_create_system_message(self):
        """Test creating a system message."""
        msg = LLMMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."


class TestLLMResponse:
    """Test LLM response model."""
    
    def test_create_basic_response(self):
        """Test creating a basic response."""
        response = LLMResponse(content="Hello!")
        assert response.content == "Hello!"
        assert response.finish_reason is None
        assert response.tokens_used is None
    
    def test_create_complete_response(self):
        """Test creating a complete response with all fields."""
        response = LLMResponse(
            content="Hello there!",
            finish_reason="stop",
            tokens_used=5
        )
        assert response.content == "Hello there!"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 5


class TestBaseLLMService:
    """Test base LLM service functionality."""
    
    def test_availability_check(self):
        """Test service availability check."""
        service = MockLLMService(available=True)
        assert service.is_available is True
        
        service = MockLLMService(available=False)
        assert service.is_available is False
    
    @pytest.mark.asyncio
    async def test_generate_streaming(self):
        """Test streaming text generation."""
        service = MockLLMService()
        messages = [LLMMessage(role="user", content="Hello")]
        
        result = await service.generate(messages, stream=True)
        
        # Collect all chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        assert chunks == ["Hello", " there!"]
    
    @pytest.mark.asyncio
    async def test_generate_complete(self):
        """Test complete text generation."""
        service = MockLLMService()
        messages = [LLMMessage(role="user", content="Hello")]
        
        result = await service.generate(messages, stream=False)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_summarize(self):
        """Test text summarization."""
        service = MockLLMService()
        
        result = await service.summarize("This is a long text that needs to be summarized")
        assert result.startswith("Summary of:")
        assert "This is a long text that needs to be summarized" in result
    
    def test_prepare_system_message_no_persona(self):
        """Test system message preparation with no persona."""
        service = MockLLMService()
        messages = [
            LLMMessage(role="user", content="Hello")
        ]
        
        result = service._prepare_system_message(messages, "")
        assert result == messages
    
    def test_prepare_system_message_with_persona_no_existing_system(self):
        """Test system message preparation with persona but no existing system message."""
        service = MockLLMService()
        messages = [
            LLMMessage(role="user", content="Hello")
        ]
        persona = "You are a helpful assistant."
        
        result = service._prepare_system_message(messages, persona)
        
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == persona
        assert result[1] == messages[0]
    
    def test_prepare_system_message_with_persona_existing_system(self):
        """Test system message preparation with persona and existing system message."""
        service = MockLLMService()
        messages = [
            LLMMessage(role="system", content="Original system message"),
            LLMMessage(role="user", content="Hello")
        ]
        persona = "You are a helpful assistant."
        
        result = service._prepare_system_message(messages, persona)
        
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == f"{persona}\n\n{messages[0].content}"
        assert result[1] == messages[1]
    
    def test_prepare_system_message_multiple_system_messages(self):
        """Test system message preparation with multiple system messages."""
        service = MockLLMService()
        messages = [
            LLMMessage(role="system", content="First system message"),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="system", content="Second system message"),
            LLMMessage(role="assistant", content="Hi there!")
        ]
        persona = "You are a helpful assistant."
        
        result = service._prepare_system_message(messages, persona)
        
        # Should only take the first system message and enhance it
        assert len(result) == 3  # Enhanced system + user + assistant
        assert result[0].role == "system"
        assert "You are a helpful assistant." in result[0].content
        assert "First system message" in result[0].content
        
        # Other non-system messages should be preserved
        assert result[1].role == "user"
        assert result[2].role == "assistant"