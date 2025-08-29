"""Guardrails service for input filtering and output validation."""

import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GuardrailsConfig(BaseModel):
    """Configuration for guardrails behavior."""
    
    # Input filtering
    max_input_length: int = 10000
    blocked_patterns: list[str] = []
    allowed_roles: list[str] = ["system", "user", "assistant"]
    
    # Output validation
    max_output_length: int = 5000
    blocked_output_patterns: list[str] = []
    
    # Safety settings
    strict_mode: bool = True


class GuardrailsViolation(Exception):
    """Raised when content violates guardrails."""
    
    def __init__(self, message: str, violation_type: str, content: str = ""):
        super().__init__(message)
        self.violation_type = violation_type
        self.content = content


class GuardrailsService:
    """Service for input filtering and output validation."""
    
    # Default patterns for potentially unsafe content
    DEFAULT_BLOCKED_PATTERNS = [
        # Prevent prompt injection attempts
        r"ignore\s+(all\s+)?(previous|all)\s+(instructions|prompts|rules)",
        r"new\s+(instruction|prompt|task|rule):",
        r"system\s*(message|prompt)?\s*:\s*",
        r"<\s*system\s*>",
        
        # Prevent sensitive information requests
        r"(show|tell|give)\s+me\s+(your|the)\s+(password|key|token|secret)",
        r"(api|access)\s+(key|token|secret|credential)",
        
        # Prevent attempts to break character
        r"(forget|ignore)\s+(your|the)\s+(persona|character|role)",
        r"act\s+as\s+(if\s+you\s+are\s+)?(not|different)",
        
        # Block attempts to access system information
        r"(show|list|display)\s+(files|directories|system|processes)",
        r"execute\s+(command|code|script)",
    ]
    
    DEFAULT_OUTPUT_BLOCKED_PATTERNS = [
        # Block if AI tries to reveal system prompts
        r"my\s+(system\s+)?(prompt|instruction|rule)",
        r"i\s+was\s+(told|instructed|programmed)",
        
        # Block potential harmful outputs
        r"(hack|exploit|attack|virus|malware)",
        r"(illegal|criminal|harmful)\s+(activity|action|behavior)",
        
        # Block personal information patterns (basic)
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
        r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card-like patterns
    ]
    
    def __init__(self, config: GuardrailsConfig | None = None):
        """Initialize guardrails service.
        
        Args:
            config: Guardrails configuration
        """
        self.config = config or GuardrailsConfig()
        
        # Compile regex patterns for efficiency
        self._input_patterns = self._compile_patterns(
            self.config.blocked_patterns + self.DEFAULT_BLOCKED_PATTERNS
        )
        self._output_patterns = self._compile_patterns(
            self.config.blocked_output_patterns + self.DEFAULT_OUTPUT_BLOCKED_PATTERNS
        )
        
    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern]:
        """Compile regex patterns for efficient matching."""
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        return compiled
    
    def validate_input(self, content: str, role: str = "user") -> None:
        """Validate input content against guardrails.
        
        Args:
            content: Content to validate
            role: Message role (system, user, assistant)
            
        Raises:
            GuardrailsViolation: If content violates guardrails
        """
        # Check role validity
        if role not in self.config.allowed_roles:
            raise GuardrailsViolation(
                f"Invalid role '{role}'. Allowed: {', '.join(self.config.allowed_roles)}",
                "invalid_role",
                content
            )
        
        # Check length limits
        if len(content) > self.config.max_input_length:
            raise GuardrailsViolation(
                f"Input too long: {len(content)} > {self.config.max_input_length}",
                "input_too_long",
                content
            )
        
        # Check blocked patterns
        for pattern in self._input_patterns:
            if pattern.search(content):
                logger.warning(f"Blocked input matching pattern: {pattern.pattern}")
                raise GuardrailsViolation(
                    f"Input contains blocked content",
                    "blocked_pattern",
                    content
                )
    
    def validate_output(self, content: str) -> str:
        """Validate and potentially filter output content.
        
        Args:
            content: Content to validate
            
        Returns:
            Filtered content (may be modified or blocked)
            
        Raises:
            GuardrailsViolation: If content violates guardrails and strict_mode is True
        """
        # Check length limits
        if len(content) > self.config.max_output_length:
            if self.config.strict_mode:
                raise GuardrailsViolation(
                    f"Output too long: {len(content)} > {self.config.max_output_length}",
                    "output_too_long",
                    content
                )
            else:
                # Truncate in non-strict mode
                logger.warning(f"Truncating output from {len(content)} to {self.config.max_output_length}")
                content = content[:self.config.max_output_length] + "..."
        
        # Check blocked patterns
        for pattern in self._output_patterns:
            if pattern.search(content):
                logger.warning(f"Blocked output matching pattern: {pattern.pattern}")
                if self.config.strict_mode:
                    raise GuardrailsViolation(
                        "Output contains blocked content",
                        "blocked_output_pattern",
                        content
                    )
                else:
                    # Replace with safe message in non-strict mode
                    return "I can't provide that information."
        
        return content
    
    def is_safe_input(self, content: str, role: str = "user") -> bool:
        """Check if input is safe without raising exceptions.
        
        Args:
            content: Content to check
            role: Message role
            
        Returns:
            True if content is safe, False otherwise
        """
        try:
            self.validate_input(content, role)
            return True
        except GuardrailsViolation:
            return False
    
    def is_safe_output(self, content: str) -> bool:
        """Check if output is safe without raising exceptions.
        
        Args:
            content: Content to check
            
        Returns:
            True if content is safe, False otherwise
        """
        try:
            self.validate_output(content)
            return True
        except GuardrailsViolation:
            return False