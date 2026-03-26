"""Thin LiteLLM wrapper for arch-conscience.

LiteLLM handles provider routing, Anthropic system-message splitting,
response normalization, and retries. We add our dataclass contract
and unified error handling on top.

Model string determines the provider:
    "gpt-4o"                          → OpenAI
    "gpt-4o-mini"                     → OpenAI
    "anthropic/claude-sonnet-4-20250514"  → Anthropic
    "text-embedding-3-large"          → OpenAI embeddings

API keys are read from environment variables automatically by LiteLLM:
    OPENAI_API_KEY, ANTHROPIC_API_KEY
"""

import logging

import litellm

from app.llm.base import CompletionResult, LLMProviderError, Message

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose default logging
litellm.suppress_debug_info = True

_NUM_RETRIES = 3


def _infer_provider(model: str) -> str:
    """Extract provider name from a LiteLLM model string for error messages."""
    return model.split("/")[0] if "/" in model else "openai"


async def complete(
    messages: list[Message],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: dict | None = None,
) -> CompletionResult:
    """Run a chat completion via LiteLLM.

    Args:
        messages: Conversation history as Message objects.
        model: LiteLLM model string (e.g. "gpt-4o", "anthropic/claude-sonnet-4-20250514").
        temperature: Sampling temperature. Stage 2 detection MUST use 0.0.
        max_tokens: Maximum tokens in the response.
        response_format: Optional response format (e.g. {"type": "json_object"}).

    Returns:
        CompletionResult with the model's response and model identifier.

    Raises:
        LLMProviderError: On API failure after retries.
    """
    provider = _infer_provider(model)

    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_retries": _NUM_RETRIES,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await litellm.acompletion(**kwargs)
        return CompletionResult(
            content=response.choices[0].message.content or "",
            model=response.model or model,
        )
    except Exception as exc:
        logger.error("LLM completion failed [%s]: %s", model, exc)
        raise LLMProviderError(provider, str(exc)) from exc


async def embed(
    texts: list[str],
    *,
    model: str,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Generate embeddings via LiteLLM.

    Args:
        texts: Strings to embed.
        model: Embedding model string (e.g. "text-embedding-3-large").
        dimensions: Optional output dimensionality (passed through to providers
                    that support it, like OpenAI).

    Returns:
        List of embedding vectors, one per input text, in order.

    Raises:
        LLMProviderError: On API failure after retries.
    """
    provider = _infer_provider(model)

    try:
        kwargs: dict = {"model": model, "input": texts, "num_retries": _NUM_RETRIES}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = await litellm.aembedding(**kwargs)
        sorted_data = sorted(response.data, key=lambda d: d["index"] if isinstance(d, dict) else d.index)
        return [d["embedding"] if isinstance(d, dict) else d.embedding for d in sorted_data]
    except Exception as exc:
        logger.error("Embedding failed [%s]: %s", model, exc)
        raise LLMProviderError(provider, str(exc)) from exc