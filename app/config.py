import json
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


def _needs_anthropic(model: str) -> bool:
    """True if the model string routes to Anthropic via LiteLLM."""
    return model.startswith("anthropic/")


def _needs_openai(model: str) -> bool:
    """True if the model string routes to OpenAI (the default in LiteLLM)."""
    return not model.startswith("anthropic/")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── API keys ─────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")

    # ── Models ───────────────────────────────────────────────────────
    # LiteLLM routes by model string — no separate provider config needed.
    # OpenAI models:    "gpt-4o", "gpt-4o-mini", "text-embedding-3-large"
    # Anthropic models: "anthropic/claude-sonnet-4-20250514"
    STAGE1_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model for Stage 1 relevance filter (cheap)",
    )
    STAGE2_MODEL: str = Field(
        default="gpt-4o",
        description="Model for Stage 2 gap detection (capable)",
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large",
        description="Embedding model for corpus indexing",
    )
    EMBEDDING_DIM: int = Field(
        default=3072,
        description="Embedding dimension — must match the collection",
    )

    # ── Confluence (optional) ────────────────────────────────────────
    CONFLUENCE_BASE_URL: str = Field(
        default="", description="Confluence instance URL (e.g. https://yourorg.atlassian.net)"
    )
    CONFLUENCE_TOKEN: str = Field(default="", description="Atlassian API token")
    CONFLUENCE_SPACE_KEY: str = Field(
        default="", description="Confluence space key to ingest from"
    )

    # ── Jira (optional) ──────────────────────────────────────────────
    JIRA_BASE_URL: str = Field(
        default="", description="Jira instance URL (e.g. https://yourorg.atlassian.net)"
    )
    JIRA_TOKEN: str = Field(default="", description="Atlassian API token for Jira")

    # ── GitHub ───────────────────────────────────────────────────────
    GITHUB_TOKEN: str = Field(default="", description="GitHub personal access token")
    GITHUB_WEBHOOK_SECRET: str = Field(
        default="", description="GitHub webhook HMAC secret"
    )

    # ── Qdrant ───────────────────────────────────────────────────────
    QDRANT_URL: str = Field(
        default="http://localhost:6333", description="Qdrant instance URL"
    )
    QDRANT_API_KEY: str = Field(default="", description="Qdrant API key (cloud only)")
    QDRANT_COLLECTION: str = Field(
        default="arch_decisions", description="Qdrant collection name"
    )

    # ── Telegram ─────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram bot token")
    TELEGRAM_CHAT_ID: str = Field(default="", description="Telegram chat ID")

    # ── Slack ─────────────────────────────────────────────────────────
    SLACK_WEBHOOK_URL: str = Field(
        default="",
        description="Default Slack incoming webhook URL (fallback when service not in SLACK_CHANNEL_MAP)",
    )
    SLACK_CHANNEL_MAP_RAW: str = Field(
        default="{}",
        alias="SLACK_CHANNEL_MAP",
        description=(
            "JSON map of service names to Slack incoming webhook URLs. "
            'e.g. {"auth-service": "https://hooks.slack.com/...", '
            '"payments-service": "https://hooks.slack.com/..."}'
        ),
    )

    # ── Detection tuning ─────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.7, description="Min confidence to fire a gap alert"
    )
    STAGE1_THRESHOLD: float = Field(
        default=0.5, description="Min relevance score to pass Stage 1"
    )
    ALERT_CHANNEL: str = Field(
        default="telegram", description="Delivery channel: telegram | slack"
    )

    # ── Service mapping ──────────────────────────────────────────────
    SERVICE_MAP_RAW: str = Field(
        default="{}",
        alias="SERVICE_MAP",
        description='JSON map of path prefixes to service names e.g. {"services/auth":"auth-service"}',
    )

    # ── Server ───────────────────────────────────────────────────────
    WEBHOOK_PORT: int = Field(default=3456, description="Webhook server port")

    @property
    def service_map(self) -> dict[str, str]:
        try:
            return json.loads(self.SERVICE_MAP_RAW)
        except json.JSONDecodeError:
            return {}

    @property
    def slack_channel_map(self) -> dict[str, str]:
        try:
            return json.loads(self.SLACK_CHANNEL_MAP_RAW)
        except json.JSONDecodeError:
            return {}

    def validate_required(self) -> None:
        """
        Called at startup. Raises ValueError listing every missing
        required var so the operator sees all problems at once.

        API key requirements are inferred from model strings:
          "gpt-4o"                     → needs OPENAI_API_KEY
          "anthropic/claude-sonnet-4-20250514" → needs ANTHROPIC_API_KEY
        """
        errors: list[str] = []

        if not self.GITHUB_TOKEN:
            errors.append("GITHUB_TOKEN")
        if not self.GITHUB_WEBHOOK_SECRET:
            errors.append("GITHUB_WEBHOOK_SECRET")
        if not self.QDRANT_URL:
            errors.append("QDRANT_URL")

        # Collect which providers the chosen models require
        all_models = [self.STAGE1_MODEL, self.STAGE2_MODEL, self.EMBEDDING_MODEL]

        if any(_needs_openai(m) for m in all_models) and not self.OPENAI_API_KEY:
            using = [m for m in all_models if _needs_openai(m)]
            errors.append(f"OPENAI_API_KEY (required by {', '.join(using)})")

        if any(_needs_anthropic(m) for m in all_models) and not self.ANTHROPIC_API_KEY:
            using = [m for m in all_models if _needs_anthropic(m)]
            errors.append(f"ANTHROPIC_API_KEY (required by {', '.join(using)})")

        if errors:
            raise ValueError(
                f"Missing required environment variables: {', '.join(errors)}"
            )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this everywhere instead of instantiating Settings() directly.
    """
    return Settings()