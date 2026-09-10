import hashlib
import os
import re

from model_library.base import LLM, LLMConfig
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model
from pydantic import SecretStr


DEFAULT_PROXY_URL_ENV_VARS = ("AGENT_BASE_URL", "AGENT_URL")
DEFAULT_PROXY_KEY_ENV_VARS = ("AGENT_API_KEY", "AGENT_KEY")
QWEN_PROXY_URL_ENV_VARS = ("QWEN3_API_URL", "QWEN3_BASE_URL")
QWEN_PROXY_KEY_ENV_VARS = ("QWEN3_API_KEY",)


def _get_first_env(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def _normalize_base_url(base_url: str) -> str:
    """Accept either an API base URL or a full chat-completions URL."""
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized + "/"


def _model_env_prefix(model_name: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", model_name.upper()).strip("_")
    return f"MODEL_{normalized}"


def _proxy_env_vars_for_model(
    model_name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return proxy environment variables in descending precedence."""
    prefix = _model_env_prefix(model_name)
    url_vars = (f"{prefix}_BASE_URL", f"{prefix}_API_URL")
    key_vars = (f"{prefix}_API_KEY",)

    if model_name.lower().startswith("qwen"):
        url_vars += QWEN_PROXY_URL_ENV_VARS
        key_vars += QWEN_PROXY_KEY_ENV_VARS

    return (
        url_vars + DEFAULT_PROXY_URL_ENV_VARS,
        key_vars + DEFAULT_PROXY_KEY_ENV_VARS,
    )


def _api_model_name_for_model(model_name: str) -> tuple[str, str | None]:
    """Return the server-advertised model ID, falling back to the CLI name."""
    env_var = f"{_model_env_prefix(model_name)}_MODEL_ID"
    value = os.getenv(env_var)
    if value and value.strip():
        return value.strip(), env_var
    return model_name, None


def get_model(model_name: str, config: LLMConfig) -> LLM:
    """Build a registry model or an OpenAI-compatible proxy model.

    Provider-qualified names (for example ``openai/gpt-5``) use the
    model-library registry. Bare names (for example ``glm-5.2``) use the
    OpenAI-compatible proxy configured through model-specific, Qwen-family, or
    general ``AGENT_*`` environment variables.
    """
    if "/" in model_name:
        return get_registry_model(model_name, config)

    url_env_vars, key_env_vars = _proxy_env_vars_for_model(model_name)
    base_url, base_url_env = _get_first_env(url_env_vars)
    api_key, api_key_env = _get_first_env(key_env_vars)
    api_model_name, api_model_name_env = _api_model_name_for_model(model_name)

    missing = []
    if not base_url:
        missing.append(f"a base URL ({', '.join(url_env_vars)})")
    if not api_key:
        missing.append(f"an API key ({', '.join(key_env_vars)})")
    if missing:
        raise ValueError(
            f"Bare model name '{model_name}' uses the custom proxy, but "
            f"{', '.join(missing)} is not set."
        )
    assert base_url is not None
    assert api_key is not None

    normalized_base_url = _normalize_base_url(base_url)
    proxy_config = config.model_copy(
        update={
            "custom_endpoint": normalized_base_url,
            "custom_api_key": SecretStr(api_key),
            "supports_tools": True,
            "supports_output_schema": False,
        }
    )

    # model-library 0.1.29 accepts custom endpoint configuration directly on
    # LLMConfig. Chat Completions is required because local OpenAI-compatible
    # servers generally do not implement the Responses API.
    endpoint_id = hashlib.sha256(normalized_base_url.encode()).hexdigest()[:12]
    model = OpenAIModel(
        model_name=api_model_name,
        # Keep clients for different proxy endpoints isolated even when they
        # happen to use the same API key.
        provider=f"agent_proxy_{endpoint_id}",
        config=proxy_config,
        use_completions=True,
    )
    model.instance_logger.info(
        "Using custom OpenAI-compatible proxy configured by %s and %s; API model from %s",
        base_url_env,
        api_key_env,
        api_model_name_env or "--model",
    )
    return model
