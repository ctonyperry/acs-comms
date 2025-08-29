"""Tests for text normalization functionality."""

import pytest

from src.acs_bridge.audio.textnorm import normalize, preprocess_for_tts


class TestTextNormalization:
    """Test cases for text normalization functions."""

    def test_normalize_simple_sentences(self):
        """Test basic sentence normalization."""
        text = "Hello world. How are you? I'm fine!"
        result = normalize(text)
        expected = ["Hello world.", "How are you?", "I'm fine!"]
        assert result == expected

    def test_normalize_empty_text(self):
        """Test normalization with empty text."""
        assert normalize("") == []
        assert normalize("   ") == []
        assert normalize(None) == []

    def test_normalize_no_punctuation(self):
        """Test normalization when text has no sentence punctuation."""
        text = "Hello world how are you"
        result = normalize(text)
        expected = ["Hello world how are you."]
        assert result == expected

    def test_normalize_abbreviations(self):
        """Test that abbreviations don't break sentences inappropriately."""
        # This case should NOT split because U.S.A. is an abbreviation
        text = "Dr. Smith went to the U.S.A. He loves it there."
        result = normalize(text)
        # The whole thing should be one sentence because "U.S.A. He" is protected
        expected = ["Dr. Smith went to the U.S.A. He loves it there."]
        assert result == expected

        # But this should split because "America." is not an abbreviation
        text = "Dr. Smith went to America. He loves it there."
        result = normalize(text)
        expected = ["Dr. Smith went to America.", "He loves it there."]
        assert result == expected

    def test_normalize_multiple_spaces(self):
        """Test normalization cleans up multiple spaces."""
        text = "Hello    world.  How   are you?"
        result = normalize(text)
        expected = ["Hello world.", "How are you?"]
        assert result == expected

    def test_preprocess_contractions(self):
        """Test contraction expansion."""
        text = "I can't believe it. You're amazing!"
        result = preprocess_for_tts(text)
        assert "cannot" in result
        assert "You are" in result
        assert "can't" not in result
        assert "you're" not in result

    def test_preprocess_symbols(self):
        """Test symbol replacement."""
        text = "Use @ symbol & $10 for 50% off"
        result = preprocess_for_tts(text)
        assert " at " in result
        assert " and " in result
        assert " dollars " in result
        assert " percent" in result
        assert "@" not in result
        assert "&" not in result
        assert "$" not in result
        assert "%" not in result
