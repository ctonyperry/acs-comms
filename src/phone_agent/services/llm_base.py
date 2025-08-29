"""Base LLM interface definition."""

from typing import Protocol, AsyncIterator, List, Dict, Optional

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    """Protocol for Large Language Model services."""
    
    async def generate(
        self,
        messages: List[Message],
        stream: bool = True,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str] | str:
        """Generate response from messages.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            stream: If True, yield text chunks; else return full string
            stop: Optional list of stop sequences to end generation
            
        Returns:
            AsyncIterator[str] if stream=True, yielding text chunks
            str if stream=False, returning complete response
        """
        ...

    async def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens in summary
            
        Returns:
            Short summary suitable for call wrap-up notes
        """
        ...