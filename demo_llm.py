#!/usr/bin/env python3
"""
Example demonstrating the Phone Agent LLM integration with Ollama.

This example shows how to:
1. Load persona configuration
2. Initialize guardrails
3. Setup Ollama LLM service
4. Generate responses with input/output filtering
5. Handle streaming and non-streaming responses

Note: Requires Ollama to be running locally with a model installed.
"""

import asyncio
import logging
import yaml
from pathlib import Path

from src.phone_agent.services.llm_base import assemble_messages, LLMUnavailable
from src.phone_agent.services.llm_ollama import OllamaLLMService
from src.phone_agent.services.guardrails import Guardrails

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_persona_config(config_path: str = "./config/persona.yaml") -> dict:
    """Load persona configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded persona config: {config['name']}")
        return config
    except FileNotFoundError:
        logger.error(f"Persona config not found: {config_path}")
        # Return minimal config
        return {
            "name": "Demo Agent",
            "style": "You are helpful and concise.",
            "constraints": ["Never give harmful advice."],
            "guardrails": {"blocklist": [], "allow_tools": []},
            "llm": {"temperature": 0.7, "max_tokens": 256},
        }


async def demo_basic_interaction():
    """Demonstrate basic LLM interaction with guardrails."""
    print("=== Phone Agent LLM Demo ===\n")

    # Load persona configuration
    persona_config = await load_persona_config()

    # Initialize guardrails
    guardrails = Guardrails(persona_config)
    system_prompt = guardrails.build_system_prompt()
    print(f"System prompt:\n{system_prompt}\n")

    # Initialize LLM service (requires Ollama running)
    llm_config = persona_config.get("llm", {})
    llm_service = OllamaLLMService(
        model="llama3.2:1b",  # Small model for demo
        temperature=llm_config.get("temperature", 0.7),
        top_p=llm_config.get("top_p", 0.9),
        max_tokens=llm_config.get("max_tokens", 256),
    )

    if not llm_service.is_available:
        print("❌ Ollama service not available!")
        print("Please ensure Ollama is installed and running with a model.")
        print("Example: ollama pull llama3.2:1b")
        return

    print("✅ Ollama service available\n")

    # Demo conversation
    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ["quit", "exit", "bye"]:
                break

            # Apply input filters
            filtered_input = guardrails.apply_input_filters(user_input)
            if filtered_input != user_input:
                print(f"🛡️  Input filtered: '{user_input}' -> '{filtered_input}'")

            # Assemble messages
            messages = assemble_messages(system_prompt, conversation_history, filtered_input)

            # Generate response (streaming)
            print("Assistant: ", end="", flush=True)
            response_parts = []

            async for chunk in await llm_service.generate(messages, stream=True):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)

            response = "".join(response_parts)
            print()  # New line after streaming

            # Check output guardrails
            allowed, reason = guardrails.is_output_allowed(response)
            if not allowed:
                print(f"🛡️  Response blocked: {reason}")
                continue

            # Add to conversation history
            conversation_history.extend(
                [
                    {"role": "user", "content": filtered_input},
                    {"role": "assistant", "content": response},
                ]
            )

            # Limit conversation history to last 10 messages
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except LLMUnavailable as e:
            print(f"❌ LLM unavailable: {e}")
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            break

    print("\n👋 Goodbye!")
    await llm_service.close()


async def demo_summarization():
    """Demonstrate text summarization capability."""
    print("\n=== Summarization Demo ===\n")

    # Initialize LLM service
    llm_service = OllamaLLMService(model="llama3.2:1b", max_tokens=128)

    if not llm_service.is_available:
        print("❌ Ollama service not available for summarization demo")
        return

    # Sample text to summarize
    sample_text = """
    The rapid advancement of artificial intelligence has transformed numerous industries 
    over the past decade. From healthcare and finance to transportation and entertainment, 
    AI technologies have enabled organizations to automate complex processes, gain deeper 
    insights from data, and create entirely new products and services. Machine learning 
    algorithms now power everything from recommendation systems and fraud detection to 
    autonomous vehicles and medical diagnosis. As AI continues to evolve, we can expect 
    even more innovative applications that will reshape how we work, communicate, and 
    interact with technology in our daily lives.
    """

    print(f"Original text ({len(sample_text)} chars):")
    print(sample_text.strip())
    print()

    try:
        summary = await llm_service.summarize(sample_text, max_tokens=50)
        print(f"Summary ({len(summary)} chars):")
        print(summary)
    except LLMUnavailable as e:
        print(f"❌ Summarization failed: {e}")

    await llm_service.close()


async def demo_guardrails():
    """Demonstrate guardrails filtering."""
    print("\n=== Guardrails Demo ===\n")

    # Create test guardrails
    test_config = {
        "name": "Test Agent",
        "guardrails": {
            "blocklist": ["password", "credit card", "social security"],
            "allow_tools": ["transfer_call"],
        },
    }

    guardrails = Guardrails(test_config)

    # Test input filtering
    print("Input filtering tests:")
    test_inputs = [
        "Hello, how are you?",
        "I forgot my password",
        "My credit card was stolen",
        "Can you help with    multiple   spaces?",
    ]

    for test_input in test_inputs:
        filtered = guardrails.apply_input_filters(test_input)
        status = "✅ OK" if filtered == test_input else "🛡️  FILTERED"
        print(f"  {status}: '{test_input}' -> '{filtered}'")

    print("\nOutput validation tests:")
    test_outputs = [
        "I can help you with that.",
        "Your password is secret123",
        "The credit card number is 1234-5678-9012-3456",
        "I will <tool>transfer_call</tool> now.",
        "Let me <tool>delete_database</tool> for you.",
    ]

    for test_output in test_outputs:
        allowed, reason = guardrails.is_output_allowed(test_output)
        status = "✅ ALLOWED" if allowed else "🛡️  BLOCKED"
        print(f"  {status}: '{test_output}'")
        if not allowed:
            print(f"    Reason: {reason}")


async def main():
    """Run all demos."""
    print("🤖 Phone Agent LLM Integration Demo\n")

    # Run demos
    await demo_guardrails()
    await demo_summarization()
    await demo_basic_interaction()


if __name__ == "__main__":
    asyncio.run(main())
