"""Guardrails system for input filtering and output validation."""

import logging
import re

logger = logging.getLogger(__name__)


class Guardrails:
    """Guardrails for input filtering and output validation."""

    def __init__(self, persona_cfg: dict) -> None:
        """Initialize guardrails with persona configuration.

        Args:
            persona_cfg: Persona configuration dictionary
        """
        self.persona_cfg = persona_cfg
        self.guardrails_cfg = persona_cfg.get("guardrails", {})
        self.blocklist = self.guardrails_cfg.get("blocklist", [])
        self.allowed_tools = self.guardrails_cfg.get("allow_tools", [])

        # Compile blocklist patterns for efficiency
        self._blocklist_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.blocklist
        ]

        logger.info(f"Initialized guardrails with {len(self.blocklist)} blocklist patterns")

    def apply_input_filters(self, text: str) -> str:
        """Apply input filters to sanitize user text.

        Args:
            text: Raw user input text

        Returns:
            Filtered text safe for processing
        """
        if not text:
            return text

        # Basic sanitization
        filtered = text.strip()

        # Remove excessive whitespace
        filtered = re.sub(r"\s+", " ", filtered)

        # Check for blocked content
        for pattern in self._blocklist_patterns:
            if pattern.search(filtered):
                logger.warning(f"Input blocked by pattern: {pattern.pattern}")
                # Replace blocked content with placeholder
                filtered = pattern.sub("[FILTERED]", filtered)

        # Log if any filtering occurred
        if filtered != text.strip():
            logger.info("Input filtering applied")

        return filtered

    def is_output_allowed(self, text: str) -> tuple[bool, str]:
        """Check if output text is allowed by guardrails.

        Args:
            text: Generated output text to validate

        Returns:
            Tuple of (is_allowed: bool, reason: str)
        """
        if not text:
            return True, "Empty text is allowed"

        # Check against blocklist patterns
        for pattern in self._blocklist_patterns:
            if pattern.search(text):
                reason = f"Output contains blocked content: {pattern.pattern}"
                logger.warning(reason)
                return False, reason

        # Check for sensitive information patterns
        # Credit card pattern (basic)
        if re.search(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", text):
            reason = "Output contains potential credit card number"
            logger.warning(reason)
            return False, reason

        # SSN pattern (basic)
        if re.search(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", text):
            reason = "Output contains potential SSN"
            logger.warning(reason)
            return False, reason

        # Check for tool usage restrictions
        tool_mentions = re.findall(r"<tool[^>]*>([^<]+)</tool>", text, re.IGNORECASE)
        for tool in tool_mentions:
            if tool not in self.allowed_tools:
                reason = f"Output uses unauthorized tool: {tool}"
                logger.warning(reason)
                return False, reason

        return True, "Output passes all guardrails"

    def build_system_prompt(self) -> str:
        """Build system prompt with persona and constraints.

        Returns:
            Complete system prompt string
        """
        persona_name = self.persona_cfg.get("name", "AI Assistant")
        persona_style = self.persona_cfg.get("style", "").strip()
        constraints = self.persona_cfg.get("constraints", [])

        # Start with persona identity and style
        prompt_parts = [f"You are {persona_name}.", persona_style]

        # Add constraints
        if constraints:
            prompt_parts.append("Important constraints:")
            for constraint in constraints:
                prompt_parts.append(f"- {constraint}")

        # Add tool information if available
        if self.allowed_tools:
            prompt_parts.append("Available tools:")
            for tool in self.allowed_tools:
                prompt_parts.append(f"- {tool}")

        # Add general guardrails reminder
        prompt_parts.extend(
            [
                "",
                "Always follow these guidelines:",
                "- Never share sensitive information like credit card numbers or SSNs",
                "- Stay professional and helpful",
                "- If unsure about something, ask for clarification",
            ]
        )

        system_prompt = "\n".join(filter(None, prompt_parts))

        logger.debug(f"Built system prompt: {len(system_prompt)} characters")
        return system_prompt
