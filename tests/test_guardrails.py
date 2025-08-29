"""Tests for Guardrails system."""


import pytest

from src.phone_agent.services.guardrails import Guardrails


class TestGuardrails:
    """Test Guardrails input filtering and output validation."""

    @pytest.fixture
    def basic_persona_config(self) -> dict:
        """Basic persona configuration for testing."""
        return {
            "name": "Test Agent",
            "style": "You are helpful and concise.",
            "constraints": ["Never give legal advice.", "Never collect sensitive information."],
            "guardrails": {
                "blocklist": ["credit card", "social security number", "password"],
                "allow_tools": ["transfer_call", "create_ticket"],
            },
        }

    @pytest.fixture
    def guardrails(self, basic_persona_config) -> Guardrails:
        """Create Guardrails instance for testing."""
        return Guardrails(basic_persona_config)

    def test_init(self, basic_persona_config):
        """Test Guardrails initialization."""
        guardrails = Guardrails(basic_persona_config)

        assert guardrails.persona_cfg == basic_persona_config
        assert guardrails.blocklist == ["credit card", "social security number", "password"]
        assert guardrails.allowed_tools == ["transfer_call", "create_ticket"]

    def test_init_empty_config(self):
        """Test initialization with empty configuration."""
        guardrails = Guardrails({})

        assert guardrails.blocklist == []
        assert guardrails.allowed_tools == []

    def test_apply_input_filters_basic(self, guardrails):
        """Test basic input filtering."""
        text = "  Hello world  "
        filtered = guardrails.apply_input_filters(text)

        assert filtered == "Hello world"

    def test_apply_input_filters_whitespace(self, guardrails):
        """Test whitespace normalization."""
        text = "Hello    world\n\nwith    spaces"
        filtered = guardrails.apply_input_filters(text)

        assert filtered == "Hello world with spaces"

    def test_apply_input_filters_blocked_content(self, guardrails):
        """Test filtering of blocked content."""
        text = "Please help with my credit card issue"
        filtered = guardrails.apply_input_filters(text)

        assert "[FILTERED]" in filtered
        assert "credit card" not in filtered.lower()

    def test_apply_input_filters_case_insensitive(self, guardrails):
        """Test that filtering is case insensitive."""
        text = "My CREDIT CARD was stolen"
        filtered = guardrails.apply_input_filters(text)

        assert "[FILTERED]" in filtered

    def test_apply_input_filters_empty_text(self, guardrails):
        """Test filtering empty text."""
        assert guardrails.apply_input_filters("") == ""
        assert guardrails.apply_input_filters(None) is None

    def test_is_output_allowed_basic(self, guardrails):
        """Test basic output validation."""
        text = "Hello, how can I help you today?"
        allowed, reason = guardrails.is_output_allowed(text)

        assert allowed is True
        assert "passes all guardrails" in reason

    def test_is_output_allowed_empty(self, guardrails):
        """Test output validation with empty text."""
        allowed, reason = guardrails.is_output_allowed("")

        assert allowed is True
        assert "Empty text is allowed" in reason

    def test_is_output_allowed_blocked_content(self, guardrails):
        """Test output validation blocks forbidden content."""
        text = "Your credit card number is..."
        allowed, reason = guardrails.is_output_allowed(text)

        assert allowed is False
        assert "blocked content" in reason

    def test_is_output_allowed_credit_card_pattern(self, guardrails):
        """Test credit card number detection."""
        text = "The number is 1234 5678 9012 3456"
        allowed, reason = guardrails.is_output_allowed(text)

        assert allowed is False
        assert "credit card" in reason

    def test_is_output_allowed_ssn_pattern(self, guardrails):
        """Test SSN detection."""
        text = "Your SSN is 123-45-6789"
        allowed, reason = guardrails.is_output_allowed(text)

        assert allowed is False
        assert "SSN" in reason

    def test_is_output_allowed_authorized_tool(self, guardrails):
        """Test authorized tool usage is allowed."""
        text = "I will <tool>transfer_call</tool> now."
        allowed, reason = guardrails.is_output_allowed(text)

        assert allowed is True

    def test_is_output_allowed_unauthorized_tool(self, guardrails):
        """Test unauthorized tool usage is blocked."""
        text = "Let me <tool>delete_database</tool> for you."
        allowed, reason = guardrails.is_output_allowed(text)

        assert allowed is False
        assert "unauthorized tool" in reason

    def test_build_system_prompt_basic(self, guardrails):
        """Test basic system prompt building."""
        prompt = guardrails.build_system_prompt()

        assert "Test Agent" in prompt
        assert "You are helpful and concise" in prompt
        assert "Never give legal advice" in prompt
        assert "transfer_call" in prompt
        assert "Never share sensitive information" in prompt

    def test_build_system_prompt_minimal_config(self):
        """Test system prompt with minimal configuration."""
        minimal_config = {"name": "Simple Bot"}
        guardrails = Guardrails(minimal_config)

        prompt = guardrails.build_system_prompt()

        assert "Simple Bot" in prompt
        assert "Never share sensitive information" in prompt

    def test_build_system_prompt_no_name(self):
        """Test system prompt with no name specified."""
        config = {"style": "Be helpful"}
        guardrails = Guardrails(config)

        prompt = guardrails.build_system_prompt()

        assert "AI Assistant" in prompt
        assert "Be helpful" in prompt
