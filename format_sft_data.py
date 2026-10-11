#!/usr/bin/env python3
"""Convert finance-agent trace folders into SFT JSONL.

Input can be either:
  - one question directory, e.g. .../q054
  - a run directory containing qxxx subdirectories

Default output is the native ms-swift Agent format.  In this format Qwen's
chat/agent template inserts <|im_start|>, <tools>, <tool_call> and
<tool_response> at tokenization time.  Only <think> is kept directly in the
assistant content because it is supervised model output.
Keep normalize_sft_loss.py beside this script when copying it elsewhere.
By default, failed tool-call groups have loss=false. Source success/error
flags take priority over error-return heuristics. ms-swift merges parallel
calls under one loss flag, so any failed call masks its entire group while
keeping the observations and assistant reasoning. Disable this policy with
--no-mask-failed-tool-calls.

Modes:
  trajectory: preserve the agent conversation, including reasoning, tool calls
              and tool results in OpenAI-compatible message fields.
  final:      keep only the initial task prompt and the final answer.

The converter supports current `trajectory_atif.json`/`history.json` traces and
legacy `history.bin` traces.  Run this with the finance-agent virtualenv when
converting legacy pickles or serialized responses:
  cd /Users/chongyanghe/Desktop/金融_Agent/finance-agent-main
  .venv/bin/python /path/to/format_sft_data.py --input logs/.../q054 --output trace_sft.jsonl
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import pickle
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from normalize_sft_loss import mask_failed_tool_calls, normalize_sample_loss, tool_result_failed


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\x00", "").strip()


def dump_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_json_value(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def turn_number(path: Path) -> int:
    match = re.search(r"turn_(\d+)$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def find_question_dirs(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if (input_path / "result.json").exists() and (
        (input_path / "turns").is_dir()
        or (input_path / "trajectory_atif.json").exists()
    ):
        return [input_path]
    # A model log root normally contains timestamped run directories and then
    # qXXX directories, so discovery must be recursive. Restricting the glob to
    # q* avoids walking every turn file as a candidate trajectory directory.
    result = [
        path
        for path in input_path.rglob("q*")
        if path.is_dir()
        and re.fullmatch(r"q\d+", path.name)
        and (path / "result.json").exists()
        and (
            (path / "turns").is_dir()
            or (path / "trajectory_atif.json").exists()
        )
    ]
    return sorted(result, key=lambda path: (str(path.parent), int(path.name[1:])))


def record_id(question_dir: Path, input_path: Path) -> str:
    """Create stable unique IDs when converting multiple timestamped runs."""
    if question_dir == input_path:
        return question_dir.name
    relative = question_dir.relative_to(input_path)
    if len(relative.parts) == 1:
        return question_dir.name
    return "__".join(relative.parts)


def latest_history_path(question_dir: Path) -> Path:
    turns_path = question_dir / "turns"
    if not turns_path.is_dir():
        raise FileNotFoundError(f"No turns directory found under {question_dir}")

    turn_dirs = sorted(
        [
            p
            for p in turns_path.iterdir()
            if p.is_dir() and p.name.startswith("turn_")
        ],
        key=turn_number,
    )
    for turn_dir in reversed(turn_dirs):
        for filename in ("history.bin", "history.json"):
            candidate = turn_dir / filename
            if candidate.exists():
                return candidate
    init_dir = turns_path / "init"
    for filename in ("history.bin", "history.json"):
        candidate = init_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No history.bin or history.json found under {question_dir}"
    )


def load_history(question_dir: Path) -> list[Any]:
    path = latest_history_path(question_dir)
    if path.suffix == ".json":
        history = load_json_value(path)
        if not isinstance(history, list):
            raise ValueError(f"Expected list in {path}, got {type(history)}")
        return history

    try:
        with path.open("rb") as f:
            history = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cannot unpickle history.bin because model_library is not importable. "
            "Run this script with the finance-agent virtualenv, for example: "
            ".venv/bin/python agent_trace_to_sft.py ..."
        ) from exc
    if not isinstance(history, list):
        raise ValueError(f"Expected list in {path}, got {type(history)}")
    return history


def load_trace(question_dir: Path) -> tuple[str, list[Any] | dict[str, Any]]:
    """Load the richest available trace representation.

    ATIF is preferred because it stores reasoning content separately from the
    provider response.  This matters for current logs, whose history.json
    response field contains a serialized ChatCompletionMessage without the
    separate reasoning field.
    """
    atif_path = question_dir / "trajectory_atif.json"
    if atif_path.exists():
        trajectory = load_json(atif_path)
        if not isinstance(trajectory.get("steps"), list):
            raise ValueError(f"Expected steps list in {atif_path}")
        return "atif", trajectory
    return "history", load_history(question_dir)


def load_tools(
    question_dir: Path,
    trajectory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert finance-agent config tools to OpenAI/ms-swift function schemas."""
    config_path = question_dir / "turns" / "init" / "config.json"
    if not config_path.exists():
        if trajectory is None:
            return []
        agent = trajectory.get("agent") or {}
        fallback_tools = agent.get("tool_definitions") or []
        return fallback_tools if isinstance(fallback_tools, list) else []
    config = load_json(config_path)
    tools: list[dict[str, Any]] = []
    for item in config.get("tools", []) or []:
        if not isinstance(item, dict):
            continue
        body = item.get("body") or {}
        name = clean_text(body.get("name") or item.get("name"))
        if not name:
            continue
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": body.get("properties") or {},
        }
        required = body.get("required") or []
        if required:
            parameters["required"] = required
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": clean_text(body.get("description")),
                    "parameters": parameters,
                },
            }
        )
    return tools


def get_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def extract_text_input(item: Any) -> str | None:
    text = get_field(item, "text")
    text = clean_text(text)
    return text or None


def get_tool_call_name(tool_call: Any) -> str:
    function = get_field(tool_call, "function")
    return clean_text(
        get_field(function, "name") or get_field(tool_call, "name")
    )


def get_tool_call_arguments(tool_call: Any) -> str:
    function = get_field(tool_call, "function")
    value = get_field(function, "arguments")
    if value is None:
        value = get_field(tool_call, "args")
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, ensure_ascii=False)


def convert_tool_call(tool_call: Any) -> dict[str, Any]:
    return {
        "id": clean_text(
            get_field(tool_call, "id") or get_field(tool_call, "call_id")
        ),
        "type": "function",
        "function": {
            "name": get_tool_call_name(tool_call),
            "arguments": get_tool_call_arguments(tool_call),
        },
    }


def convert_tool_call_ms_swift(tool_call: Any) -> dict[str, Any]:
    """Return the canonical payload used by an ms-swift tool_call message."""
    arguments = get_tool_call_arguments(tool_call)
    try:
        arguments_obj = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        arguments_obj = {"value": arguments}
    if not isinstance(arguments_obj, dict):
        arguments_obj = {"value": arguments_obj}
    return {
        "name": get_tool_call_name(tool_call),
        "arguments": arguments_obj,
    }


def assistant_message_from_raw_response(item: Any, *, include_reasoning: bool) -> dict[str, Any] | None:
    response = get_field(item, "response")
    if response is None:
        return None

    content = clean_text(get_field(response, "content"))
    reasoning = clean_text(get_field(response, "reasoning_content"))
    if include_reasoning and reasoning:
        if content:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        else:
            content = f"<think>\n{reasoning}\n</think>"

    message: dict[str, Any] = {"role": "assistant", "content": content}

    raw_tool_calls = get_field(response, "tool_calls") or []
    tool_calls = [convert_tool_call(tool_call) for tool_call in raw_tool_calls]
    if tool_calls:
        message["tool_calls"] = tool_calls

    if not message["content"] and not tool_calls:
        return None
    return message


def ms_swift_messages_from_raw_response(
    item: Any, *, include_reasoning: bool,
    tool_statuses: dict[str, bool] | None = None,
    tool_outcomes: list[tuple[str, bool]] | None = None,
) -> list[dict[str, Any]]:
    """Split one model response into assistant text and tool_call messages.

    ms-swift's agent_template later concatenates these pieces and renders the
    model-specific <tool_call> markers.  Keeping the call structured also gives
    ms-swift the correct loss mask: tool_call is trained, tool_response is not.
    """
    response = get_field(item, "response")
    if response is None:
        return []

    content = clean_text(get_field(response, "content"))
    reasoning = clean_text(get_field(response, "reasoning_content"))
    if include_reasoning and reasoning:
        reasoning_text = f"<think>\n{reasoning}\n</think>"
        content = f"{reasoning_text}\n\n{content}" if content else reasoning_text

    messages: list[dict[str, Any]] = []
    if content:
        messages.append({"role": "assistant", "content": content})

    for position, tool_call in enumerate(get_field(response, "tool_calls") or []):
        call_id = clean_text(get_field(tool_call, "id") or get_field(tool_call, "call_id"))
        failed = (tool_statuses or {}).get(call_id, tool_result_failed(tool_call))
        if call_id not in (tool_statuses or {}) and tool_outcomes and position < len(tool_outcomes):
            result_id, result_failed = tool_outcomes[position]
            if not call_id or not result_id:
                failed = result_failed
        messages.append(
            {
                "role": "tool_call",
                "content": dump_json(convert_tool_call_ms_swift(tool_call)),
                "loss": not failed,
            }
        )
    return messages


def tool_message_from_result(item: Any, *, as_user: bool) -> dict[str, Any] | None:
    tool_call = get_field(item, "tool_call")
    result = clean_text(get_field(item, "result"))
    if tool_call is None:
        return None

    name = clean_text(get_field(tool_call, "name"))
    tool_call_id = clean_text(
        get_field(tool_call, "id") or get_field(tool_call, "call_id")
    )
    if as_user:
        return {
            "role": "user",
            "content": f"Tool result from {name} ({tool_call_id}):\n{result}",
        }
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": result,
    }


def ms_swift_tool_response_from_result(item: Any) -> dict[str, Any] | None:
    """Create a model-visible tool response without runtime-only metadata."""
    tool_call = get_field(item, "tool_call")
    if tool_call is None:
        return None
    return {
        "role": "tool_response",
        "content": clean_text(get_field(item, "result")),
    }


def decode_history_response(item: Mapping[str, Any]) -> Any:
    """Decode a current history.json raw_response payload.

    Current model-library logs store OpenAI response objects as base64-encoded
    pickles.  A dict or plain JSON response is accepted as well for forward
    compatibility.
    """
    response = item.get("response")
    if not isinstance(response, str):
        return response

    compact = "".join(response.split())
    try:
        return pickle.loads(base64.b64decode(compact, validate=True))
    except (binascii.Error, EOFError, ValueError, pickle.UnpicklingError):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"content": response}


def atif_tool_call_id(tool_call: Mapping[str, Any]) -> str:
    return clean_text(
        tool_call.get("tool_call_id")
        or tool_call.get("id")
        or tool_call.get("call_id")
    )


def atif_tool_call_name(tool_call: Mapping[str, Any]) -> str:
    return clean_text(tool_call.get("function_name") or tool_call.get("name"))


def atif_tool_call_arguments(tool_call: Mapping[str, Any]) -> Any:
    arguments = tool_call.get("arguments")
    return arguments if isinstance(arguments, dict) else {"value": arguments}


def convert_atif_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": atif_tool_call_id(tool_call),
        "type": "function",
        "function": {
            "name": atif_tool_call_name(tool_call),
            "arguments": json.dumps(
                atif_tool_call_arguments(tool_call), ensure_ascii=False
            ),
        },
    }


def convert_atif_tool_call_ms_swift(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": atif_tool_call_name(tool_call),
        "arguments": atif_tool_call_arguments(tool_call),
    }


def trajectory_to_messages(
    trajectory: dict[str, Any],
    *,
    include_reasoning: bool,
    tool_role: str,
    output_schema: str,
    tool_statuses: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Convert an ATIF trajectory into the same message layout as history."""
    messages: list[dict[str, Any]] = []
    for raw_step in trajectory.get("steps", []):
        if not isinstance(raw_step, dict):
            continue
        source = raw_step.get("source")
        if source in {"system", "user"}:
            content = clean_text(raw_step.get("message"))
            if content:
                messages.append({"role": source, "content": content})
            continue
        if source != "agent":
            continue

        content = clean_text(raw_step.get("message"))
        reasoning = clean_text(raw_step.get("reasoning_content"))
        if include_reasoning and reasoning:
            reasoning_text = f"<think>\n{reasoning}\n</think>"
            content = f"{reasoning_text}\n\n{content}" if content else reasoning_text

        raw_tool_calls = raw_step.get("tool_calls") or []
        tool_calls = [
            call for call in raw_tool_calls if isinstance(call, dict)
        ]
        observation = raw_step.get("observation") or {}
        observations = observation.get("results") or []
        observations = [item for item in observations if isinstance(item, dict)]

        if output_schema == "ms-swift":
            statuses = dict(tool_statuses or {})
            for item in observations:
                call_id = clean_text(item.get("source_call_id"))
                if call_id and call_id not in statuses:
                    statuses[call_id] = tool_result_failed(item, content=clean_text(item.get("content")))
            if content:
                messages.append({"role": "assistant", "content": content})
            for position, tool_call in enumerate(tool_calls):
                call_id = atif_tool_call_id(tool_call)
                failed = statuses.get(call_id, tool_result_failed(tool_call))
                if call_id not in statuses and position < len(observations):
                    item = observations[position]
                    if not call_id or not item.get("source_call_id"):
                        failed = tool_result_failed(item, content=clean_text(item.get("content")))
                messages.append(
                    {
                        "role": "tool_call",
                        "content": dump_json(
                            convert_atif_tool_call_ms_swift(tool_call)
                        ),
                        "loss": not failed,
                    }
                )
            for item in observations:
                messages.append(
                    {
                        "role": "tool_response",
                        "content": clean_text(item.get("content")),
                    }
                )
            continue

        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        converted_calls = [convert_atif_tool_call(call) for call in tool_calls]
        if converted_calls:
            assistant["tool_calls"] = converted_calls
        if assistant["content"] or converted_calls:
            messages.append(assistant)

        names_by_id = {
            atif_tool_call_id(call): atif_tool_call_name(call)
            for call in tool_calls
        }
        for item in observations:
            call_id = clean_text(item.get("source_call_id"))
            messages.append(
                {
                    "role": "user" if tool_role == "user" else "tool",
                    **(
                        {
                            "content": (
                                f"Tool result from {names_by_id.get(call_id, '')} "
                                f"({call_id}):\n{clean_text(item.get('content'))}"
                            )
                        }
                        if tool_role == "user"
                        else {
                            "tool_call_id": call_id,
                            "name": names_by_id.get(call_id, ""),
                            "content": clean_text(item.get("content")),
                        }
                    ),
                }
            )
    return messages


def history_to_messages(
    history: list[Any],
    *,
    include_reasoning: bool,
    tool_role: str,
    output_schema: str,
    tool_statuses: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    statuses = dict(tool_statuses or {})
    outcomes: dict[int, list[tuple[str, bool]]] = {}
    if output_schema == "ms-swift":
        response_index = None
        for index, item in enumerate(history):
            if get_field(item, "kind") == "raw_response" or item.__class__.__name__ == "RawResponse":
                response_index = index
            if get_field(item, "kind") != "tool_result" and item.__class__.__name__ != "ToolResult":
                continue
            call = get_field(item, "tool_call")
            call_id = clean_text(get_field(call, "id") or get_field(call, "call_id"))
            if call_id and call_id not in statuses:
                statuses[call_id] = tool_result_failed(item, content=clean_text(get_field(item, "result")))
            if response_index is not None:
                outcomes.setdefault(response_index, []).append(
                    (call_id, statuses.get(call_id, tool_result_failed(item, content=clean_text(get_field(item, "result")))))
                )
    for index, item in enumerate(history):
        if isinstance(item, Mapping):
            kind = item.get("kind")
            if kind == "system":
                text = clean_text(item.get("text"))
                if text:
                    messages.append({"role": "system", "content": text})
            elif kind == "text":
                text = clean_text(item.get("text"))
                if text:
                    messages.append({"role": "user", "content": text})
            elif kind == "raw_response":
                decoded = decode_history_response(item)
                response_item = {"response": decoded}
                if output_schema == "ms-swift":
                    messages.extend(
                        ms_swift_messages_from_raw_response(
                            response_item, include_reasoning=include_reasoning,
                            tool_statuses=statuses,
                            tool_outcomes=outcomes.get(index),
                        )
                    )
                else:
                    message = assistant_message_from_raw_response(
                        response_item, include_reasoning=include_reasoning
                    )
                    if message:
                        messages.append(message)
            elif kind == "tool_result":
                if output_schema == "ms-swift":
                    message = ms_swift_tool_response_from_result(item)
                else:
                    message = tool_message_from_result(
                        item, as_user=(tool_role == "user")
                    )
                if message:
                    messages.append(message)
            continue

        class_name = item.__class__.__name__
        if class_name == "TextInput":
            text = extract_text_input(item)
            if text:
                messages.append({"role": "user", "content": text})
        elif class_name == "RawResponse":
            if output_schema == "ms-swift":
                messages.extend(
                    ms_swift_messages_from_raw_response(
                        item, include_reasoning=include_reasoning,
                        tool_statuses=statuses,
                        tool_outcomes=outcomes.get(index),
                    )
                )
            else:
                message = assistant_message_from_raw_response(
                    item, include_reasoning=include_reasoning
                )
                if message:
                    messages.append(message)
        elif class_name == "ToolResult":
            if output_schema == "ms-swift":
                message = ms_swift_tool_response_from_result(item)
            else:
                message = tool_message_from_result(item, as_user=(tool_role == "user"))
            if message:
                messages.append(message)
    return messages


def extract_initial_prompt(trace: list[Any] | dict[str, Any]) -> str:
    if isinstance(trace, dict):
        for step in trace.get("steps", []):
            if not isinstance(step, dict) or step.get("source") != "user":
                continue
            text = clean_text(step.get("message"))
            if text:
                return text
        raise ValueError("No initial user step found in trajectory_atif.json")

    history = trace
    for item in history:
        if isinstance(item, Mapping) and item.get("kind") == "text":
            text = clean_text(item.get("text"))
            if text:
                return text
            continue
        if item.__class__.__name__ == "TextInput":
            text = extract_text_input(item)
            if text:
                return text
    raise ValueError("No initial TextInput found in history")


def build_final_record(
    question_dir: Path,
    trace: list[Any] | dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    final_answer = clean_text(result.get("final_answer"))
    if not final_answer:
        raise ValueError(f"Missing final_answer in {question_dir / 'result.json'}")
    return {
        "id": question_dir.name,
        "messages": [
            {"role": "user", "content": extract_initial_prompt(trace)},
            {"role": "assistant", "content": final_answer},
        ],
        "source": "finance_agent_final",
        "metadata": metadata_from_result(question_dir, result),
    }


def build_trajectory_record(
    question_dir: Path,
    trace: list[Any] | dict[str, Any],
    result: dict[str, Any],
    *,
    include_reasoning: bool,
    tool_role: str,
    output_schema: str,
    include_tool_metadata: bool,
    mask_failed_calls: bool = True,
) -> dict[str, Any]:
    if isinstance(trace, dict):
        expected_ids = {
            atif_tool_call_id(call)
            for step in trace.get("steps", []) if isinstance(step, dict)
            for call in step.get("tool_calls", []) or [] if isinstance(call, dict)
        }
    else:
        # Decode once so call IDs remain available even if a legacy tool result
        # has lost its ID. history_to_messages can reuse the decoded response.
        trace = [
            {**item, "response": decode_history_response(item)}
            if isinstance(item, Mapping) and item.get("kind") == "raw_response"
            else item
            for item in trace
        ]
        expected_ids = {
            clean_text(get_field(get_field(item, "tool_call"), "id") or get_field(get_field(item, "tool_call"), "call_id"))
            for item in trace if get_field(item, "tool_call") is not None
        }
        expected_ids.update(
            clean_text(get_field(call, "id") or get_field(call, "call_id"))
            for item in trace
            for call in get_field(get_field(item, "response"), "tool_calls", []) or []
        )
    expected_ids.discard("")
    statuses = collect_tool_statuses(question_dir, result, expected_call_ids=expected_ids) if output_schema == "ms-swift" and mask_failed_calls else {}
    if isinstance(trace, dict):
        messages = trajectory_to_messages(
            trace,
            include_reasoning=include_reasoning,
            tool_role=tool_role,
            output_schema=output_schema,
            tool_statuses=statuses,
        )
    else:
        messages = history_to_messages(
            trace,
            include_reasoning=include_reasoning,
            tool_role=tool_role,
            output_schema=output_schema,
            tool_statuses=statuses,
        )
    if not messages:
        raise ValueError(f"No messages extracted from {question_dir}")
    if output_schema == "ms-swift" and not mask_failed_calls:
        for message in messages:
            if message["role"] == "tool_call":
                message["loss"] = True
    record = {
        "id": question_dir.name,
        "messages": messages,
        "source": "finance_agent_trace",
        "metadata": metadata_from_result(
            question_dir,
            result,
            include_tool_metadata=include_tool_metadata,
        ),
    }
    if output_schema == "ms-swift":
        # ms-swift documents this field as a JSON string, rather than a list,
        # which also avoids PyArrow type-inference inconsistencies.
        record["tools"] = dump_json(
            load_tools(question_dir, trace if isinstance(trace, dict) else None)
        )
        if mask_failed_calls:
            # Failure status was matched to call IDs above. Do not override a
            # known success with a text heuristic here; just mask whole groups.
            record, _ = mask_failed_tool_calls(record, inspect_responses=False)
    return record


def collect_tool_statuses(
    question_dir: Path, result: dict[str, Any], *, expected_call_ids: set[str] | None = None,
) -> dict[str, bool]:
    """Read structured call failures by ID, independent of tool-result order."""
    statuses: dict[str, bool] = {}
    for turn in result.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        for call in turn.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            call_id = clean_text(call.get("tool_call_id"))
            if call_id and (isinstance(call.get("success"), bool) or call.get("error")):
                statuses[call_id] = tool_result_failed(call)
    if expected_call_ids is None or not expected_call_ids.issubset(statuses):
        # Older run summaries lack per-call state; detailed turn records still
        # provide success/error alongside the same call IDs.
        for record in collect_tool_runtime_metadata(question_dir):
            call_id = clean_text(record.get("tool_call_id"))
            if call_id and (isinstance(record.get("success"), bool) or record.get("error")):
                statuses.setdefault(call_id, tool_result_failed(record))
    return statuses


def collect_tool_runtime_metadata(question_dir: Path) -> list[dict[str, Any]]:
    """Collect observability metadata without duplicating full tool outputs."""
    records: list[dict[str, Any]] = []
    turn_dirs = sorted(
        [p for p in (question_dir / "turns").glob("turn_*") if p.is_dir()],
        key=turn_number,
    )
    for turn_dir in turn_dirs:
        result_path = turn_dir / "result.json"
        if not result_path.exists():
            continue
        turn_result = load_json(result_path)
        for item in turn_result.get("tool_call_records", []) or []:
            tool_call = item.get("tool_call") or {}
            tool_output = item.get("tool_output") or {}
            output = tool_output.get("output")
            records.append(
                {
                    "turn": turn_number(turn_dir),
                    "tool_name": tool_call.get("name"),
                    "tool_call_id": tool_call.get("id") or tool_call.get("call_id"),
                    "success": tool_output.get("success", item.get("success")),
                    "error": tool_output.get("error", item.get("error")),
                    "duration_seconds": item.get("duration_seconds"),
                    "done": tool_output.get("done"),
                    "metadata": tool_output.get("metadata"),
                    "output_chars": len(output) if isinstance(output, str) else None,
                }
            )
    return records


def metadata_from_result(
    question_dir: Path,
    result: dict[str, Any],
    *,
    include_tool_metadata: bool = False,
) -> dict[str, Any]:
    turns = result.get("turns")
    metadata = {
        "trace_dir": str(question_dir),
        "success": result.get("final_error") is None and bool(clean_text(result.get("final_answer"))),
        "turn_count": len(turns) if isinstance(turns, list) else None,
        "final_error": result.get("final_error"),
    }
    if include_tool_metadata:
        metadata["tool_call_records"] = collect_tool_runtime_metadata(question_dir)
    return metadata


def convert_question_dir(question_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    result = load_json(question_dir / "result.json")
    if args.success_only and result.get("final_error") is not None:
        raise ValueError("skip unsuccessful trace")

    _, trace = load_trace(question_dir)
    if args.mode == "final":
        record = build_final_record(question_dir, trace, result)
    else:
        record = build_trajectory_record(
            question_dir,
            trace,
            result,
            include_reasoning=args.include_reasoning,
            tool_role=args.tool_role,
            output_schema=args.output_schema,
            include_tool_metadata=args.include_tool_metadata,
            mask_failed_calls=getattr(args, "mask_failed_tool_calls", True),
        )
    if args.output_schema == "ms-swift":
        return normalize_sample_loss(record)
    return record


def iter_records(input_path: Path, args: argparse.Namespace) -> Iterable[tuple[Path, dict[str, Any] | None, str | None]]:
    question_dirs = find_question_dirs(input_path)
    if not question_dirs:
        raise FileNotFoundError(f"No qxxx trace directories found under {input_path}")
    for question_dir in question_dirs:
        try:
            record = convert_question_dir(question_dir, args)
            record["id"] = record_id(question_dir, input_path)
            yield question_dir, record, None
        except Exception as exc:
            if args.fail_fast:
                raise
            yield question_dir, None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="qxxx trace dir or run dir containing qxxx dirs")
    parser.add_argument("--output", required=True, help="Output SFT JSONL path")
    parser.add_argument("--mode", choices=("trajectory", "final"), default="trajectory")
    parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tool-role", choices=("tool", "user"), default="tool")
    parser.add_argument(
        "--output-schema",
        choices=("ms-swift", "openai"),
        default="ms-swift",
        help=(
            "ms-swift emits tool_call/tool_response roles plus a JSON-string tools field; "
            "openai preserves assistant.tool_calls and role=tool"
        ),
    )
    parser.add_argument(
        "--include-tool-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store runtime tool metadata under sample metadata (never in model-visible messages)",
    )
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-failed-tool-calls", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "ms-swift trajectory only: mask a contiguous tool-call group if any call failed; "
            "prefer source success/error flags, falling back to recognizable error returns"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_path = output_path.with_suffix(output_path.suffix + ".rejected.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    rejected = 0
    with output_path.open("w", encoding="utf-8") as out, rejected_path.open("w", encoding="utf-8") as rej:
        for question_dir, record, error in iter_records(input_path, args):
            if record is None:
                rejected += 1
                rej.write(dump_json({"trace_dir": str(question_dir), "error": error}) + "\n")
                continue
            out.write(dump_json(record) + "\n")
            written += 1

    print(f"wrote={written} output={output_path}")
    print(f"rejected={rejected} rejected_file={rejected_path}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
