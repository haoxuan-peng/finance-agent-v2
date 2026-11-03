import contextlib
import io
import json
import logging
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
from model_library.base import LLM, LLMConfig, TextInput
from model_library.providers.openai import OpenAIModel
from openai import AsyncOpenAI

from finance_agent.get_agent import Parameters, get_agent
from finance_agent.model_factory import get_model
from finance_agent.run_agent import main
from finance_agent.tools import RetrieveInformation


def registry_model(model_name: str, config: LLMConfig) -> LLM:
    llm = MagicMock(spec=LLM)
    llm.model_name = model_name
    llm.prompt_caching = config.prompt_caching
    return llm


class GetAgentTest(unittest.TestCase):
    def test_retrieve_information_llm_skips_prompt_caching(self) -> None:
        parameters = Parameters(
            model_name="anthropic/model",
            tools=["retrieve_information"],
            llm_config=LLMConfig(),
        )
        with patch("finance_agent.get_agent.get_model", registry_model):
            agent = get_agent(parameters)

        tool = agent._tools["retrieve_information"]
        assert isinstance(tool, RetrieveInformation)
        self.assertEqual(agent._llm.prompt_caching, "5m")
        self.assertEqual(tool._llm.prompt_caching, "off")
        self.assertEqual(parameters.llm_config.prompt_caching, "5m")

    def test_proxy_retrieval_model_preserves_routing_and_primary_config(self) -> None:
        general_env = {
            "AGENT_BASE_URL": "https://general.invalid/v1/chat/completions",
            "AGENT_API_KEY": "general-test-key",
        }
        routes = [
            (
                "glm-5.2",
                general_env,
                "https://general.invalid/v1/",
                "general-test-key",
                "glm-5.2",
            ),
            (
                "qwen3.5-9b",
                general_env
                | {
                    "QWEN3_API_URL": "https://qwen.invalid/v1",
                    "QWEN3_API_KEY": "qwen-test-key",
                },
                "https://qwen.invalid/v1/",
                "qwen-test-key",
                "qwen3.5-9b",
            ),
            (
                "qwen3.5-9b",
                general_env
                | {
                    "QWEN3_API_URL": "https://qwen.invalid/v1",
                    "QWEN3_API_KEY": "qwen-test-key",
                    "MODEL_QWEN3_5_9B_BASE_URL": "https://specific.invalid/v1",
                    "MODEL_QWEN3_5_9B_API_KEY": "specific-test-key",
                    "MODEL_QWEN3_5_9B_MODEL_ID": "served-qwen",
                },
                "https://specific.invalid/v1/",
                "specific-test-key",
                "served-qwen",
            ),
        ]
        for model_name, env, endpoint, api_key, served_name in routes:
            with self.subTest(model_name=model_name, endpoint=endpoint):
                config = LLMConfig(max_tokens=32000, temperature=0.0)
                parameters = Parameters(
                    model_name=model_name,
                    tools=["retrieve_information"],
                    llm_config=config,
                )
                original_config = config.model_dump()
                models = []

                def proxy_model(**kwargs):
                    llm = registry_model(kwargs["model_name"], kwargs["config"])
                    llm.instance_logger = MagicMock()
                    models.append(llm)
                    return llm

                with (
                    patch.dict(os.environ, env, clear=True),
                    patch(
                        "finance_agent.model_factory.OpenAIModel",
                        side_effect=proxy_model,
                    ) as constructor,
                    patch("finance_agent.model_factory.get_registry_model") as registry,
                ):
                    agent = get_agent(parameters)

                registry.assert_not_called()
                self.assertEqual(constructor.call_count, 2)
                self.assertIs(agent._llm, models[0])
                tool = agent._tools["retrieve_information"]
                assert isinstance(tool, RetrieveInformation)
                self.assertIs(tool._llm, models[1])
                self.assertEqual(
                    [
                        call.kwargs["config"].prompt_caching
                        for call in constructor.call_args_list
                    ],
                    ["5m", "off"],
                )
                for call in constructor.call_args_list:
                    kwargs = call.kwargs
                    routed_config = kwargs["config"]
                    self.assertEqual(kwargs["model_name"], served_name)
                    self.assertTrue(kwargs["use_completions"])
                    self.assertEqual(routed_config.custom_endpoint, endpoint)
                    self.assertEqual(
                        routed_config.custom_api_key.get_secret_value(), api_key
                    )
                    self.assertEqual(routed_config.max_tokens, 32000)
                    self.assertEqual(routed_config.temperature, 0.0)
                self.assertEqual(config.model_dump(), original_config)
                self.assertEqual(parameters.max_turns, 50)
                self.assertIsNone(parameters.max_time)

    def test_injected_primary_model_is_preserved(self) -> None:
        config = LLMConfig(max_tokens=32000, temperature=0.0)
        parameters = Parameters(
            model_name="anthropic/model",
            tools=["retrieve_information"],
            llm_config=config,
        )
        primary = registry_model("injected-model", config)
        helper = registry_model(
            parameters.model_name, config.model_copy(update={"prompt_caching": "off"})
        )
        with patch("finance_agent.get_agent.get_model", return_value=helper) as factory:
            agent = get_agent(parameters, llm=primary)

        self.assertIs(agent._llm, primary)
        tool = agent._tools["retrieve_information"]
        assert isinstance(tool, RetrieveInformation)
        self.assertIs(tool._llm, helper)
        factory.assert_called_once()
        model_name, helper_config = factory.call_args.args
        self.assertEqual(model_name, parameters.model_name)
        self.assertEqual(helper_config.prompt_caching, "off")
        self.assertEqual(primary.prompt_caching, "5m")
        self.assertEqual(config.prompt_caching, "5m")


class ReasoningRequestTest(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, config: LLMConfig, extra_body: dict | None = None):
        parameters = Parameters(
            model_name="custom-model",
            tools=["retrieve_information"],
            llm_config=config,
            extra_body=extra_body or {},
        )
        with patch.dict(
            os.environ,
            {
                "AGENT_BASE_URL": "https://model.invalid/v1",
                "AGENT_API_KEY": "offline-test-key",
            },
            clear=True,
        ):
            agent = get_agent(parameters)
        tool = agent._tools["retrieve_information"]
        assert isinstance(tool, RetrieveInformation)
        return parameters, agent._llm, tool._llm

    async def test_unset_options_keep_backend_defaults(self) -> None:
        parameters, primary, helper = self.make_agent(
            LLMConfig(max_tokens=32000, temperature=0)
        )
        for model in (primary, helper):
            body = await model.build_body([TextInput(text="Question")], tools=[])
            self.assertNotIn("reasoning_effort", body)
            self.assertNotIn("extra_body", body)
            self.assertEqual(body["max_tokens"], 32000)
            self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("reasoning_effort", parameters.llm_config.model_fields_set)
        self.assertEqual(primary.prompt_caching, "5m")
        self.assertEqual(helper.prompt_caching, "off")

    async def test_reasoning_and_backend_options_reach_both_http_requests(self) -> None:
        parameters, primary, helper = self.make_agent(
            LLMConfig(max_tokens=32000, temperature=0, reasoning_effort="high"),
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "thinking_token_budget": 2048,
            },
        )
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "offline-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "custom-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "42"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            )

        async with AsyncOpenAI(
            api_key="offline-test-key",
            base_url="https://model.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            for model in (primary, helper):
                assert isinstance(model, OpenAIModel)
                model.stream_completions = False
                with patch.object(model, "get_client", return_value=client):
                    result = await model._query_impl(
                        [TextInput(text="Question")],
                        tools=[],
                        query_logger=logging.getLogger(__name__),
                    )
                self.assertEqual(result.output_text_str, "42")

        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request["reasoning_effort"], "high")
            self.assertEqual(
                request["chat_template_kwargs"], {"enable_thinking": False}
            )
            self.assertEqual(request["thinking_token_budget"], 2048)
            self.assertEqual(request["max_tokens"], 32000)
            self.assertNotIn("max_completion_tokens", request)
            self.assertNotIn("extra_body", request)
        self.assertFalse(parameters.llm_config.reasoning)
        self.assertEqual(primary.prompt_caching, "5m")
        self.assertEqual(helper.prompt_caching, "off")

    async def test_request_body_mutation_does_not_leak_to_later_queries(self) -> None:
        options = {"chat_template_kwargs": {"enable_thinking": False}}
        _, primary, helper = self.make_agent(LLMConfig(max_tokens=32000), options)
        options["chat_template_kwargs"]["enable_thinking"] = True
        first = await primary.build_body([TextInput(text="Question")], tools=[])
        first["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
        for model in (primary, helper):
            body = await model.build_body([TextInput(text="Question")], tools=[])
            self.assertEqual(
                body["extra_body"]["chat_template_kwargs"], {"enable_thinking": False}
            )

    async def test_request_overrides_merge_with_model_and_sdk_defaults(self) -> None:
        _, primary, _ = self.make_agent(
            LLMConfig(top_k=20),
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                }
            },
        )
        body = await primary.build_body(
            [TextInput(text="Question")],
            tools=[],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(
            body["extra_body"],
            {
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": True,
                },
            },
        )

        _, primary, _ = self.make_agent(LLMConfig(), {"reasoning_effort": "high"})
        body = await primary.build_body(
            [TextInput(text="Question")], tools=[], reasoning_effort="low"
        )
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", body["extra_body"])
        restored = await primary.build_body([TextInput(text="Question")], tools=[])
        self.assertEqual(restored["extra_body"]["reasoning_effort"], "high")
        final_override = await primary.build_body(
            [TextInput(text="Question")],
            tools=[],
            reasoning_effort="low",
            extra_body={"reasoning_effort": "medium"},
        )
        self.assertEqual(final_override["reasoning_effort"], "low")
        self.assertEqual(final_override["extra_body"]["reasoning_effort"], "medium")

        _, primary, _ = self.make_agent(
            LLMConfig(reasoning_effort="high"),
            {"chat_template_kwargs": {"enable_thinking": True}},
        )
        body = await primary.build_body(
            [TextInput(text="Question")],
            tools=[],
            extra_body={"reasoning_effort": "low"},
        )
        self.assertEqual(body["extra_body"]["reasoning_effort"], "low")
        self.assertEqual(primary.reasoning_effort, "high")

    def test_parameter_defaults_are_independent(self) -> None:
        first = Parameters(model_name="custom-model", llm_config=LLMConfig())
        second = Parameters(model_name="custom-model", llm_config=LLMConfig())
        first.extra_body["enable_thinking"] = False
        self.assertEqual(second.extra_body, {})

    def test_factory_rejects_unsupported_route_and_conflicting_effort(self) -> None:
        with patch("finance_agent.model_factory.get_registry_model") as registry:
            with self.assertRaises(ValueError):
                get_model(
                    "anthropic/model", LLMConfig(), extra_body={"enable_thinking": True}
                )
            registry.assert_not_called()
        with self.assertRaises(ValueError):
            get_model(
                "custom-model",
                LLMConfig(reasoning_effort="high"),
                extra_body={"reasoning_effort": "low"},
            )

    def test_injected_primary_rejects_unapplied_extra_body(self) -> None:
        parameters = Parameters(
            model_name="custom-model",
            tools=[],
            llm_config=LLMConfig(),
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        with self.assertRaises(ValueError):
            get_agent(parameters, llm=registry_model("injected-model", LLMConfig()))


class ReasoningCLITest(unittest.IsolatedAsyncioTestCase):
    async def run_cli(self, *arguments: str) -> Parameters:
        runner = AsyncMock(return_value=[])
        with (
            patch.object(
                sys,
                "argv",
                [
                    "finance-agent",
                    "--model",
                    "custom-model",
                    "--questions",
                    "Question",
                    *arguments,
                ],
            ),
            patch("finance_agent.run_agent.load_dotenv"),
            patch("finance_agent.run_agent.run_tests_parallel", runner),
        ):
            await main()
        runner.assert_awaited_once()
        assert runner.await_args is not None
        return runner.await_args.kwargs["parameters"]

    async def test_default_does_not_override_registry_reasoning_profile(self) -> None:
        parameters = await self.run_cli()
        self.assertNotIn("reasoning_effort", parameters.llm_config.model_fields_set)
        self.assertEqual(parameters.extra_body, {})

    async def test_explicit_effort_and_extra_body_are_preserved(self) -> None:
        for effort in ("high", "none", "backend-specific"):
            with self.subTest(effort=effort):
                parameters = await self.run_cli(
                    "--reasoning-effort",
                    effort,
                    "--extra-body",
                    '{"chat_template_kwargs":{"enable_thinking":false},"thinking_token_budget":2048}',
                )
                self.assertEqual(parameters.llm_config.reasoning_effort, effort)
                self.assertEqual(
                    parameters.extra_body,
                    {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "thinking_token_budget": 2048,
                    },
                )

    async def test_invalid_options_fail_before_starting_agent(self) -> None:
        cases = [
            ["--extra-body", "{"],
            ["--extra-body", "[]"],
            ["--extra-body", "null"],
            ["--extra-body", '{"thinking_token_budget":NaN}'],
            ["--reasoning-effort", ""],
            ["--reasoning-effort", "  "],
            [
                "--reasoning-effort",
                "high",
                "--extra-body",
                '{"reasoning_effort":"low"}',
            ],
            ["--model", "anthropic/model", "--extra-body", '{"enable_thinking":true}'],
        ]
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                runner = AsyncMock()
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "finance-agent",
                            "--model",
                            "custom-model",
                            "--questions",
                            "Question",
                            *arguments,
                        ],
                    ),
                    patch("finance_agent.run_agent.load_dotenv"),
                    patch("finance_agent.run_agent.run_tests_parallel", runner),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        await main()
                self.assertEqual(raised.exception.code, 2)
                runner.assert_not_awaited()
