"""Data types for the LLM abstraction layer."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Message:
    """A single message in a completion request."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """The result of a completion call."""

    content: str
    model: str


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")