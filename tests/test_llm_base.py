"""Tests for LLM base protocol and utilities."""


import pytest

from src.phone_agent.services.llm_base import LLMUnavailable, Message, assemble_messages


class TestLLMBaseTypes:
    """Test LLM base type definitions."""

    def test_message_type(self):
        """Test Message type alias works correctly."""
        message: Message = {"role": "user", "content": "Hello"}
        assert message["role"] == "user"
        assert message["content"] == "Hello"

    def test_llm_unavailable_exception(self):
        """Test LLMUnavailable exception."""
        with pytest.raises(LLMUnavailable):
            raise LLMUnavailable("Service down")


class TestAssembleMessages:
    """Test message assembly helper function."""

    def test_assemble_messages_basic(self):
        """Test basic message assembly."""
        persona_system = "You are a helpful assistant."
        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        user_text = "How are you?"

        result = assemble_messages(persona_system, history, user_text)

        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]

        assert result == expected

    def test_assemble_messages_empty_history(self):
        """Test message assembly with empty history."""
        persona_system = "You are a bot."
        history: list[Message] = []
        user_text = "Test message"

        result = assemble_messages(persona_system, history, user_text)

        expected = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Test message"},
        ]

        assert result == expected

    def test_assemble_messages_empty_system(self):
        """Test message assembly with empty system prompt."""
        persona_system = ""
        history = [{"role": "user", "content": "Previous"}]
        user_text = "Current"

        result = assemble_messages(persona_system, history, user_text)

        expected = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Previous"},
            {"role": "user", "content": "Current"},
        ]

        assert result == expected

    def test_assemble_messages_preserves_order(self):
        """Test that message assembly preserves history order."""
        persona_system = "System"
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Response 2"},
        ]
        user_text = "Third"

        result = assemble_messages(persona_system, history, user_text)

        # Check that order is preserved
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "First"
        assert result[2]["content"] == "Response 1"
        assert result[3]["content"] == "Second"
        assert result[4]["content"] == "Response 2"
        assert result[5]["content"] == "Third"
