import os
import unittest
from unittest.mock import MagicMock, patch

from model_library.base import LLM, LLMConfig

from finance_agent.get_agent import Parameters, get_agent
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
