"""Integration tests for LLM services end-to-end functionality."""

import pytest
from unittest.mock import MagicMock

from src.phone_agent.services import (
    OllamaLLM, Guardrails, assemble_messages, llm_respond_text, load_persona_config
)


class MockSettings:
    """Mock settings for integration testing."""
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_TEMPERATURE = 0.4
    OLLAMA_TOP_P = 0.9
    OLLAMA_SEED = None
    OLLAMA_MAX_TOKENS = 512
    OLLAMA_STOP = ["</s>"]
    ollama_stop = ["</s>"]


@pytest.mark.asyncio
class TestLLMIntegration:
    """Integration tests for LLM services."""
    
    def test_persona_config_loading(self):
        """Test loading persona configuration from YAML."""
        persona_cfg = load_persona_config('config/persona.yaml')
        
        assert persona_cfg['name'] == "Calm Support Agent"
        assert 'style' in persona_cfg
        assert 'constraints' in persona_cfg
        assert 'guardrails' in persona_cfg
        assert 'llm' in persona_cfg
        
    def test_message_assembly(self):
        """Test message assembly functionality."""
        persona_cfg = load_persona_config('config/persona.yaml')
        guardrails = Guardrails(persona_cfg)
        
        system_prompt = guardrails.build_system_prompt()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        user_text = "How are you?"
        
        messages = assemble_messages(system_prompt, history, user_text)
        
        assert len(messages) == 4  # system + 2 history + 1 new user
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == system_prompt
        assert messages[-1]['role'] == 'user'
        assert messages[-1]['content'] == user_text
        
    def test_guardrails_integration(self):
        """Test guardrails with persona configuration."""
        persona_cfg = load_persona_config('config/persona.yaml')
        guardrails = Guardrails(persona_cfg)
        
        # Test input filtering
        filtered = guardrails.apply_input_filters("My SSN is 123-45-6789 and card is 1234 5678 9012 3456")
        assert "[SSN]" in filtered
        assert "[CARD]" in filtered
        assert "123-45-6789" not in filtered
        assert "1234 5678 9012 3456" not in filtered
        
        # Test output validation with configured blocklist
        allowed, reason = guardrails.is_output_allowed("Please provide your credit card details")
        assert not allowed
        assert "credit card" in reason.lower()
        
        # Test system prompt generation
        prompt = guardrails.build_system_prompt()
        assert "Calm Support Agent" in prompt
        assert "concise, friendly, and helpful" in prompt
        assert "Never give legal or medical advice" in prompt
        
    async def test_end_to_end_safe_content(self):
        """Test end-to-end functionality with safe content."""
        persona_cfg = load_persona_config('config/persona.yaml')
        guardrails = Guardrails(persona_cfg)
        settings = MockSettings()
        
        # Mock LLM response
        mock_llm = MagicMock()
        
        async def mock_generate(*args, **kwargs):
            for chunk in ["Hello! ", "I can ", "help you."]:
                yield chunk
                
        mock_llm.generate.return_value = mock_generate()
        
        # Test the integration function
        history = []
        user_text = "Can you help me?"
        
        chunks = []
        async for chunk in llm_respond_text(history, user_text, settings, guardrails, mock_llm):
            chunks.append(chunk)
            
        assert chunks == ["Hello! ", "I can ", "help you."]
        
    async def test_end_to_end_blocked_content(self):
        """Test end-to-end functionality with blocked content."""
        persona_cfg = load_persona_config('config/persona.yaml')
        guardrails = Guardrails(persona_cfg)
        settings = MockSettings()
        
        # Mock LLM response with blocked content
        mock_llm = MagicMock()
        
        async def mock_generate(*args, **kwargs):
            for chunk in ["Please provide your ", "credit card ", "number"]:
                yield chunk
                
        mock_llm.generate.return_value = mock_generate()
        
        # Test the integration function
        history = []
        user_text = "I need help with payment"
        
        chunks = []
        async for chunk in llm_respond_text(history, user_text, settings, guardrails, mock_llm):
            chunks.append(chunk)
            
        # Should get first chunk, then blocked content marker
        assert chunks == ["Please provide your ", "[blocked]"]