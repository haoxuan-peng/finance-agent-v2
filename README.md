# Running Finance Agent Benchmark

Our Finance Agent benchmark evaluates LLMs on their ability to use tools to research and answer complex financial questions about companies, financial statements, and SEC filings.

The agent has access to the following tools:

- `web_search`: Search the web for information (via Tavily)
- `edgar_search`: Search the SEC's EDGAR database for filings
- `parse_html_page`: Parse and extract content from web pages
- `retrieve_information`: Access stored information from previous steps
- `price_history`: Fetch historical daily OHLCV price data for supported equities/ETFs, crypto, and FX using a single tool with `asset_class` routing

For more details on the benchmark, please refer to our [public website](https://www.vals.ai/benchmarks/fabv2).

## Set up

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Then run:

```
make install
source .venv/bin/activate
```

### Platform

Access to the Vals platform is gated and requires approval. Please reach out to us at [vals.ai](https://www.vals.ai/) to request access.

Once approved, make an account on [platform.vals.ai](https://www.platform.vals.ai/auth) with your company email address. Go to the admin page and create a new API key for yourself.

### Environment Variables

Create a `.env` file in the root of the project and add the following:

```
# LLM API Keys (only set the ones you plan on using)
OPENAI_API_KEY=<openai_api_key>
ANTHROPIC_API_KEY=<anthropic_api_key>
GOOGLE_API_KEY=<google_api_key>
ETC_API_KEY=<etc_api_key>

# OpenAI-compatible model proxy (used when --model is a bare model name)
AGENT_BASE_URL=https://your-proxy.example.com/v1
AGENT_API_KEY=<proxy_api_key>

# Optional Qwen-family endpoint. Qwen model names prefer these over AGENT_*.
QWEN3_API_URL=https://your-qwen-server.example.com/v1
QWEN3_API_KEY=<qwen_api_key>

# Tool API Keys
TAVILY_API_KEY=<tavily_api_key>  # supports semicolon-separated failover keys, e.g. key1;key2;key3
SEC_EDGAR_API_KEY=<sec_api_key>  # supports semicolon-separated keys for round-robin rotation, e.g. key1;key2;key3
PRICING_DATA_API_KEY=<pricing_data_api_key> # Tiingo API key
```

You can create a Tavily API key [here](https://tavily.com/), and an SEC API key [here](https://sec-api.io/).

The `.env` takes precedence over set environment variables.

When `--model` is a bare model name such as `glm-5.2`, the agent sends model
requests to `AGENT_BASE_URL` using `AGENT_API_KEY` and the OpenAI-compatible
Chat Completions protocol. `AGENT_URL` and `AGENT_KEY` are accepted as aliases.
The base URL may be either an API base such as `https://host/v1` or a full
`https://host/v1/chat/completions` URL. Provider-qualified names continue to use
the model-library registry.

Bare Qwen model names first use `QWEN3_API_URL` and `QWEN3_API_KEY`, falling back
to the general `AGENT_*` configuration. Model-specific variables take priority;
for example, `MODEL_QWEN3_5_9B_BASE_URL`, `MODEL_QWEN3_5_9B_API_KEY`, and
`MODEL_QWEN3_5_9B_MODEL_ID` configure `qwen3.5-9b` independently.

Finally, you should add the "Test Suite IDs" to suites.json. These should have generally been provided to you via email, but you can also find them in the platform, by navigating to the "Test Suites" page, clicking the relevant test suite, and looking on the right sidebar under "Test Suite ID".

## Running the benchmark

For a list of command line options, run `finance-agent --help`

To run, for example, a single question on openai/gpt-5.2-2025-12-11:

```
finance-agent --questions "What was Apple's revenue in 2023?" --model openai/gpt-5.2-2025-12-11
```

You can specify multiple questions at once:

```
finance-agent --questions "What was Apple's revenue in 2023?" "What was NFLX's revenue in 2024?"
```

You can also specify a list of questions in a text file, one question per line:

```
finance-agent --question-file data/public.txt
```

The default configuration is the one we used to run the benchmark.

### List of Models

A list of available models can be found at our [model library](https://github.com/vals-ai/model-library/blob/main/model_library/config/all_models.json), and also by running `make browse-models` in the model library repository.

To run your own harness or model, just modify the `get_custom_model` function as needed. To see the full documentation on how the SDK works, visit [our docs](https://docs.vals.ai/sdk/running_suites).

## Logs

The agent writes detailed logs to the `logs/` directory. Each run creates a timestamped directory with per-question log files containing tool usage, token counts, and error tracking.
