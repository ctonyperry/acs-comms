"""Tests for guardrails input filtering and output validation."""

import pytest

from src.phone_agent.services.guardrails import Guardrails


class TestGuardrails:
    """Test cases for Guardrails functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.persona_cfg = {
            "name": "Test Agent",
            "style": "Helpful and professional",
            "constraints": [
                "Never give legal advice",
                "Never collect sensitive data"
            ],
            "guardrails": {
                "blocklist": [
                    "credit card",
                    "social security number",
                    "password"
                ]
            }
        }
        self.guardrails = Guardrails(self.persona_cfg)
    
    def test_apply_input_filters_ssn_redaction(self):
        """Test SSN pattern redaction in input."""
        # Test various SSN formats
        test_cases = [
            ("My SSN is 123-45-6789", "My SSN is [SSN]"),
            ("SSN: 987-65-4321 please", "SSN: [SSN] please"),
            ("Contact 123-45-6789 and 987-65-4321", "Contact [SSN] and [SSN]"),
            ("No SSN here", "No SSN here"),  # No change
        ]
        
        for input_text, expected in test_cases:
            result = self.guardrails.apply_input_filters(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_apply_input_filters_card_redaction(self):
        """Test credit card pattern redaction in input."""
        test_cases = [
            ("My card is 1234 5678 9012 3456", "My card is [CARD]"),
            ("Card: 4567 8901 2345 6789 expires soon", "Card: [CARD] expires soon"),
            ("Two cards: 1111 2222 3333 4444 and 5555 6666 7777 8888", 
             "Two cards: [CARD] and [CARD]"),
            ("Not a card: 123 456", "Not a card: 123 456"),  # No change
        ]
        
        for input_text, expected in test_cases:
            result = self.guardrails.apply_input_filters(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_apply_input_filters_combined(self):
        """Test combined PII redaction."""
        input_text = "SSN 123-45-6789 and card 1234 5678 9012 3456"
        expected = "SSN [SSN] and card [CARD]"
        
        result = self.guardrails.apply_input_filters(input_text)
        assert result == expected
    
    def test_is_output_allowed_blocklist_terms(self):
        """Test output blocking based on configured blocklist."""
        test_cases = [
            ("Please provide your credit card number", False, "credit card"),
            ("Your social security number is needed", False, "social security number"),
            ("Enter your password here", False, "password"),
            ("This is safe content", True, ""),
            ("Credit union information", True, ""),  # Partial match shouldn't block
        ]
        
        for text, should_allow, blocked_term in test_cases:
            allowed, reason = self.guardrails.is_output_allowed(text)
            assert allowed == should_allow, f"Failed for text: {text}"
            if not should_allow:
                assert blocked_term.lower() in reason.lower()
    
    def test_is_output_allowed_credit_card_patterns(self):
        """Test blocking of credit card number patterns in output."""
        test_cases = [
            ("Card number: 1234 5678 9012 3456", False),
            ("Card: 1234-5678-9012-3456", False),
            ("Card: 1234567890123456", False),
            ("Reference number: 12345", True),  # Too short
            ("Safe text here", True),
        ]
        
        for text, should_allow in test_cases:
            allowed, reason = self.guardrails.is_output_allowed(text)
            assert allowed == should_allow, f"Failed for text: {text}"
            if not should_allow:
                assert "credit card number pattern" in reason
    
    def test_is_output_allowed_ssn_patterns(self):
        """Test blocking of SSN patterns in output."""
        test_cases = [
            ("SSN: 123-45-6789", False),
            ("Social security: 987-65-4321", False),
            ("Reference: 12-34-56", True),  # Wrong format
            ("Phone: 123-456-7890", True),  # Wrong format (too long)
            ("Normal text", True),
        ]
        
        for text, should_allow in test_cases:
            allowed, reason = self.guardrails.is_output_allowed(text)
            assert allowed == should_allow, f"Failed for text: {text}"
            if not should_allow:
                assert "SSN pattern" in reason
    
    def test_is_output_allowed_medical_advice(self):
        """Test blocking of potential medical advice."""
        test_cases = [
            ("I diagnose you with a cold", False, "diagnose"),
            ("Take this medicine twice daily", False, "take this medicine"),
            ("Your prescription needs refilling", False, "prescription"),
            ("Increase dosage to 200mg", False, "increase dosage"),
            ("I can help you find a doctor", True, ""),  # Safe
            ("General health information", True, ""),
        ]
        
        for text, should_allow, keyword in test_cases:
            allowed, reason = self.guardrails.is_output_allowed(text)
            assert allowed == should_allow, f"Failed for text: {text}"
            if not should_allow:
                assert "medical advice" in reason
                if keyword:
                    assert keyword in reason
    
    def test_is_output_allowed_legal_advice(self):
        """Test blocking of potential legal advice."""
        test_cases = [
            ("You should file a lawsuit", False, "file a lawsuit"),
            ("This violates the law", False, "violates the law"),
            ("I can provide legal advice", False, "legal advice"),
            ("Press charges against them", False, "press charges"),
            ("Contact a lawyer for help", True, ""),  # Safe referral
            ("Legal documents are available", True, ""),
        ]
        
        for text, should_allow, keyword in test_cases:
            allowed, reason = self.guardrails.is_output_allowed(text)
            assert allowed == should_allow, f"Failed for text: {text}"
            if not should_allow:
                assert "legal advice" in reason
                if keyword:
                    assert keyword in reason
    
    def test_build_system_prompt_structure(self):
        """Test system prompt building with correct structure."""
        prompt = self.guardrails.build_system_prompt()
        
        # Check required components
        assert "Persona: Test Agent" in prompt
        assert "Style:" in prompt
        assert "Helpful and professional" in prompt
        assert "Constraints:" in prompt
        assert "- Never give legal advice" in prompt
        assert "- Never collect sensitive data" in prompt
        assert "Policy:" in prompt
        assert "brief clarifying question" in prompt
        assert "De-escalate when the user is upset" in prompt
        assert "Guardrails are enforced out-of-band" in prompt
    
    def test_build_system_prompt_minimal_config(self):
        """Test system prompt with minimal persona config."""
        minimal_cfg = {
            "name": "Simple Bot"
        }
        guardrails = Guardrails(minimal_cfg)
        prompt = guardrails.build_system_prompt()
        
        # Should still have basic structure
        assert "Persona: Simple Bot" in prompt
        assert "Policy:" in prompt
        assert "brief clarifying question" in prompt
    
    def test_case_insensitive_blocking(self):
        """Test that blocking is case insensitive."""
        test_cases = [
            "Please provide your CREDIT CARD information",
            "Your Social Security Number is required", 
            "Enter your PASSWORD here",
        ]
        
        for text in test_cases:
            allowed, reason = self.guardrails.is_output_allowed(text)
            assert not allowed, f"Should block case insensitive text: {text}"
    
    def test_empty_blocklist_config(self):
        """Test guardrails with empty blocklist configuration."""
        empty_cfg = {
            "name": "Test Agent",
            "guardrails": {}
        }
        guardrails = Guardrails(empty_cfg)
        
        # Should still block built-in patterns
        allowed, reason = guardrails.is_output_allowed("Card: 1234 5678 9012 3456")
        assert not allowed
        assert "credit card number pattern" in reason
    
    def test_no_guardrails_config(self):
        """Test guardrails without guardrails section in config."""
        no_guardrails_cfg = {
            "name": "Test Agent"
        }
        guardrails = Guardrails(no_guardrails_cfg)
        
        # Should handle missing guardrails section gracefully
        allowed, reason = guardrails.is_output_allowed("Safe content")
        assert allowed