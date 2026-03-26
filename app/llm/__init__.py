"""LLM abstraction layer — thin wrapper over LiteLLM.

Usage:
    from app.llm import complete, embed, Message

    result = await complete(
        [Message("system", "You are..."), Message("user", "Analyze this diff")],
        model="gpt-4o",
        temperature=0,
    )

    vectors = await embed(["some text"], model="text-embedding-3-large", dimensions=3072)
"""

from app.llm.base import CompletionResult, LLMProviderError, Message
from app.llm.provider import complete, embed

__all__ = [
    "CompletionResult",
    "LLMProviderError",
    "Message",
    "complete",
    "embed",
]