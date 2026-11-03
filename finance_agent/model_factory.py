import hashlib
import json
import os
import re
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

from model_library.base import InputItem, LLM, LLMConfig, ToolDefinition
from model_library.providers.openai import OpenAIModel
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel, SecretStr


DEFAULT_PROXY_URL_ENV_VARS = ("AGENT_BASE_URL", "AGENT_URL")
DEFAULT_PROXY_KEY_ENV_VARS = ("AGENT_API_KEY", "AGENT_KEY")
QWEN_PROXY_URL_ENV_VARS = ("QWEN3_API_URL", "QWEN3_BASE_URL")
QWEN_PROXY_KEY_ENV_VARS = ("QWEN3_API_KEY",)


def _has_non_string_keys(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            not isinstance(key, str) or _has_non_string_keys(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_non_string_keys(item) for item in value)
    return False


def validate_extra_body(
    extra_body: object,
    *,
    model_name: str | None = None,
    reasoning_effort: str | bool | None = None,
) -> dict[str, Any]:
    """Validate proxy request fields and return an independent copy."""
    if extra_body is None:
        return {}
    if not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a JSON object")
    try:
        json.dumps(extra_body, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("extra_body must contain JSON-compatible values") from exc
    if _has_non_string_keys(extra_body):
        raise ValueError("extra_body JSON object keys must be strings")
    if extra_body and model_name is not None and "/" in model_name:
        raise ValueError(
            "extra_body is supported only for bare model names using the custom "
            "OpenAI-compatible proxy, not provider-qualified registry models"
        )
    if reasoning_effort is not None and "reasoning_effort" in extra_body:
        raise ValueError(
            "reasoning_effort must be set either in LLMConfig or extra_body, not both"
        )
    return deepcopy(cast(dict[str, Any], extra_body))


def _merge_extra_body(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _merge_extra_body(previous, value)
        else:
            merged[key] = deepcopy(value)
    return merged


class _ProxyOpenAIModel(OpenAIModel):
    """Keep backend-specific request defaults local to this proxy model."""

    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: LLMConfig,
        extra_body: dict[str, Any],
    ):
        self._extra_body_defaults = deepcopy(extra_body)
        super().__init__(
            model_name=model_name,
            provider=provider,
            config=config,
            use_completions=True,
        )

    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        per_call = validate_extra_body(kwargs.pop("extra_body", None))
        # Explicit query arguments take priority over stored request defaults.
        # Per-query extra_body is merged last, matching the OpenAI SDK.
        defaults = {
            key: deepcopy(value)
            for key, value in self._extra_body_defaults.items()
            if key not in kwargs
        }
        body = await super().build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )
        sdk_fields = validate_extra_body(body.get("extra_body"))
        configured = _merge_extra_body(sdk_fields, defaults)
        body["extra_body"] = _merge_extra_body(configured, per_call)
        return body


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


def get_model(
    model_name: str,
    config: LLMConfig,
    *,
    extra_body: dict[str, Any] | None = None,
) -> LLM:
    """Build a registry model or an OpenAI-compatible proxy model.

    Provider-qualified names (for example ``openai/gpt-5``) use the
    model-library registry. Bare names (for example ``glm-5.2``) use the
    OpenAI-compatible proxy configured through model-specific, Qwen-family, or
    general ``AGENT_*`` environment variables.
    """
    extra_body = validate_extra_body(
        extra_body, model_name=model_name, reasoning_effort=config.reasoning_effort
    )
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
    # Keep clients for different proxy endpoints isolated even when they
    # happen to use the same API key.
    provider = f"agent_proxy_{endpoint_id}"
    if extra_body:
        model = _ProxyOpenAIModel(
            model_name=api_model_name,
            provider=provider,
            config=proxy_config,
            extra_body=extra_body,
        )
    else:
        model = OpenAIModel(
            model_name=api_model_name,
            provider=provider,
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
