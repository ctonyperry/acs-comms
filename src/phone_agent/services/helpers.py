"""Helper functions for LLM message assembly and integration."""

import yaml
from typing import List, AsyncIterator

from .llm_base import Message
from .llm_ollama import OllamaLLM, LLMUnavailable
from .guardrails import Guardrails


def assemble_messages(
    persona_system: str,
    history: List[Message],
    user_text: str,
) -> List[Message]:
    """Assemble complete message list for LLM generation.
    
    Args:
        persona_system: System prompt from persona/guardrails
        history: Previous conversation messages
        user_text: Current user input
        
    Returns:
        Complete message list with system prompt, history, and user input
    """
    messages = [{"role": "system", "content": persona_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


async def llm_respond_text(
    history: List[Message], 
    user_text: str,
    settings,
    guardrails: Guardrails,
    llm: OllamaLLM
) -> AsyncIterator[str]:
    """Generate LLM response with guardrails and persona integration.
    
    Args:
        history: Previous conversation messages
        user_text: Current user input
        settings: Application settings
        guardrails: Guardrails instance
        llm: LLM service instance
        
    Yields:
        Text chunks from LLM response, filtered by guardrails
        
    Raises:
        LLMUnavailable: If LLM service is unavailable
    """
    # Apply input filtering
    filtered_input = guardrails.apply_input_filters(user_text)
    
    # Build system prompt
    system_prompt = guardrails.build_system_prompt()
    
    # Assemble messages
    messages = assemble_messages(system_prompt, history, filtered_input)
    
    # Generate response with streaming
    try:
        async for chunk in llm.generate(messages, stream=True, stop=settings.ollama_stop):
            # Apply output validation
            allowed, reason = guardrails.is_output_allowed(chunk)
            if allowed:
                yield chunk
            else:
                # Replace blocked content with safe response
                yield "[blocked]"
                # Log the blocking event
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Content blocked: {reason}")
                break
                
    except LLMUnavailable:
        # Re-raise LLM unavailable errors for upstream handling
        raise


def load_persona_config(persona_path: str) -> dict:
    """Load persona configuration from YAML file.
    
    Args:
        persona_path: Path to persona YAML file
        
    Returns:
        Persona configuration dictionary
        
    Raises:
        FileNotFoundError: If persona file doesn't exist
        yaml.YAMLError: If persona file is invalid YAML
    """
    with open(persona_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)