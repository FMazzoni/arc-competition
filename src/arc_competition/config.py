from typing import Any, Dict

import anthropic
import instructor
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for the LLM model."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model name/identifier
    name: str = Field(
        default="claude-3-7-sonnet-20250219",
        description="The AI model to use",
    )

    # Model API parameters
    max_tokens: int = Field(
        default=2000, description="Maximum number of tokens to generate"
    )

    temperature: float = Field(
        default=0.3, description="Temperature for response generation (0.0 to 1.0)"
    )

    top_p: float = Field(default=0.9, description="Top-p sampling parameter")

    def get_api_parameters(self) -> Dict[str, Any]:
        """Get model API parameters as a dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


class APIConfig(BaseSettings):
    """Configuration for external API keys."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API key
    api_key: str = Field(
        default="",
        description="API key for the LLM model",
        validation_alias=AliasChoices(
            "anthropic_api_key", "openai_api_key", "gemini_api_key", "gpt_api_key"
        ),
    )


class Config:
    """Centralized configuration container."""

    def __init__(self):
        self.model = LLMConfig()
        self.api = APIConfig()
        self.client = instructor.from_anthropic(
            anthropic.Anthropic(api_key=self.api.api_key)
        )


# Global configuration instance
config = Config()
