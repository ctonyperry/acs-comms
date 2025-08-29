"""Text normalization utilities for TTS synthesis.

This module provides text normalization functions to prepare text for
high-quality TTS synthesis, including sentence segmentation for improved
prosody and natural speech patterns.
"""

import re


def normalize(text: str) -> list[str]:
    """Normalize text and split into sentence chunks for TTS synthesis.

    This function prepares text for TTS by:
    1. Cleaning and normalizing the input text
    2. Splitting into logical sentence chunks
    3. Handling abbreviations and special cases

    Args:
        text: Input text to normalize and segment

    Returns:
        List of sentence chunks ready for TTS synthesis

    Examples:
        >>> normalize("Hello world. How are you? I'm fine!")
        ["Hello world.", "How are you?", "I'm fine!"]

        >>> normalize("Dr. Smith went to the U.S.A. He loves it there.")
        ["Dr. Smith went to the U.S.A.", "He loves it there."]
    """
    if not text or not text.strip():
        return []

    # Clean up the text
    text = text.strip()

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # Handle common abbreviations that shouldn't trigger sentence breaks
    abbreviations = [
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "Prof.",
        "Sr.",
        "Jr.",
        "U.S.A.",
        "U.K.",
        "U.S.",
        "vs.",
        "etc.",
        "i.e.",
        "e.g.",
        "Inc.",
        "Corp.",
        "Ltd.",
        "Co.",
        "St.",
        "Ave.",
        "Rd.",
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Oct.",
        "Nov.",
        "Dec.",
    ]

    # Temporarily replace abbreviations to protect them from sentence splitting
    abbrev_placeholders = {}
    for i, abbrev in enumerate(abbreviations):
        placeholder = f"__ABBREV_{i}__"
        if abbrev in text:
            abbrev_placeholders[placeholder] = abbrev
            text = text.replace(abbrev, placeholder)

    # Split on sentence-ending punctuation followed by whitespace and capital letter
    # This regex handles: . ! ? followed by space and capital letter
    sentence_pattern = r"([.!?])\s+(?=[A-Z])"
    parts = re.split(sentence_pattern, text)

    # Reconstruct sentences (re.split separates the punctuation)
    result = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ".!?":
            # Current part + punctuation
            sentence = parts[i] + parts[i + 1]
            i += 2
        else:
            # Last sentence or sentence without captured punctuation
            sentence = parts[i]
            i += 1

        sentence = sentence.strip()
        if sentence:
            # Restore abbreviations
            for placeholder, abbrev in abbrev_placeholders.items():
                sentence = sentence.replace(placeholder, abbrev)
            result.append(sentence)

    # Handle edge case where text doesn't end with sentence punctuation
    if result and not result[-1].endswith((".", "!", "?")):
        result[-1] += "."

    # If no sentences were found (e.g., no sentence-ending punctuation),
    # treat the whole text as one sentence
    if not result and text:
        # Restore abbreviations in original text
        for placeholder, abbrev in abbrev_placeholders.items():
            text = text.replace(placeholder, abbrev)

        if not text.endswith((".", "!", "?")):
            text += "."
        result = [text]

    return result


def preprocess_for_tts(text: str) -> str:
    """Additional preprocessing for TTS-specific improvements.

    Args:
        text: Input text to preprocess

    Returns:
        Preprocessed text optimized for TTS
    """
    if not text:
        return text

    # Expand common contractions for clearer pronunciation
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }

    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Handle numbers and symbols that might be mispronounced
    # Convert common symbols to words
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("%", " percent")
    text = text.replace("$", " dollars ")

    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
