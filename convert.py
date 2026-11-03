#!/usr/bin/env python3
"""Convert successful finance-agent trajectories to ms-swift Agent SFT JSONL.

The converter reads the cumulative ``history.json`` from each successful question
directory. It masks non-assistant context, failed tool calls, and the assistant
message immediately preceding a ``Continue`` user message. Only use this script
on trusted local harness logs: response fields can contain Python pickles.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TURN_RE = re.compile(r"turn_(\d+)$")
QUESTION_RE = re.compile(r"q\d+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, nargs="+", type=Path,
        help="One or more finance-agent model trajectory roots",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N successful records")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def turn_number(path: Path) -> int:
    match = TURN_RE.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def successful_result(result: dict[str, Any]) -> bool:
    return (
        result.get("success") is True
        and result.get("stop_reason") == "done_tool"
        and bool(str(result.get("final_answer") or "").strip())
    )


def question_dirs(root: Path) -> Iterable[Path]:
    return sorted(
        (p for p in root.glob("*/q*") if p.is_dir() and QUESTION_RE.fullmatch(p.name)),
        key=lambda p: str(p),
    )


def final_history_path(question_dir: Path) -> Path:
    turn_dirs = sorted(
        (p for p in (question_dir / "turns").glob("turn_*") if p.is_dir()),
        key=turn_number,
    )
    for turn_dir in reversed(turn_dirs):
        history_path = turn_dir / "history.json"
        if history_path.is_file():
            return history_path
    raise FileNotFoundError(f"No history.json under {question_dir / 'turns'}")


def failed_tool_call_ids(question_dir: Path) -> tuple[set[str], int]:
    """Return failed call IDs and number of tool-call records inspected."""
    failed: set[str] = set()
    total = 0
    turn_dirs = sorted(
        (p for p in (question_dir / "turns").glob("turn_*") if p.is_dir()),
        key=turn_number,
    )
    for turn_dir in turn_dirs:
        path = turn_dir / "result.json"
        if not path.is_file():
            continue
        data = load_json(path)
        for record in data.get("tool_call_records") or []:
            if not isinstance(record, dict):
                continue
            call = record.get("tool_call") or {}
            call_id = str(call.get("id") or "")
            if not call_id:
                continue
            total += 1
            output = record.get("tool_output") or {}
            is_failure = (
                record.get("success") is False
                or output.get("success") is False
                or record.get("error") not in (None, "")
                or output.get("error") not in (None, "")
            )
            if is_failure:
                failed.add(call_id)
    return failed, total


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return str(value)


def is_continue(text: str) -> bool:
    return text.strip().rstrip(".").strip().casefold() == "continue"


def parse_arguments(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Preserve malformed arguments losslessly; the associated call can
            # still be masked according to its execution record.
            return {"_raw": raw}
    return {"value": raw}


def get_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def decode_response(response: Any) -> Any:
    if not isinstance(response, str):
        return response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    try:
        return pickle.loads(base64.b64decode("".join(response.split()), validate=True))
    except (binascii.Error, EOFError, ValueError, pickle.UnpicklingError):
        return {"content": response}


def assistant_text(response: Any) -> str:
    reasoning = text_content(get_field(response, "reasoning_content")).strip()
    visible = text_content(get_field(response, "content")).strip()
    parts: list[str] = []
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if visible:
        parts.append(visible)
    return "\n\n".join(parts)


def tool_call_fields(tool_call: Any) -> tuple[str, str, Any]:
    call_id = str(get_field(tool_call, "id", "") or "")
    function = get_field(tool_call, "function")
    if function is not None:
        return (
            call_id,
            str(get_field(function, "name", "") or ""),
            parse_arguments(get_field(function, "arguments")),
        )
    return (
        call_id,
        str(get_field(tool_call, "name", "") or ""),
        parse_arguments(get_field(tool_call, "args")),
    )


def convert_history(history: list[Any], failed_ids: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    messages: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for item in history:
        kind = item.get("kind") if isinstance(item, dict) else type(item).__name__

        if kind in {"text", "system", "TextInput", "SystemInput"}:
            content = text_content(get_field(item, "text"))
            role = "system" if kind in {"system", "SystemInput"} else "user"
            if is_continue(content):
                stats["continue_messages"] += 1
                for previous in reversed(messages):
                    if previous["role"] == "assistant":
                        if previous["loss"]:
                            previous["loss"] = False
                            stats["continue_preceding_assistant_masked"] += 1
                        break
                else:
                    stats["continue_without_preceding_assistant"] += 1
            messages.append({"role": role, "content": content, "loss": False})
            continue

        if kind in {"raw_response", "RawResponse"}:
            response = decode_response(get_field(item, "response"))
            if response is None:
                stats["empty_raw_response"] += 1
                continue

            content = assistant_text(response)
            if content:
                messages.append({"role": "assistant", "content": content, "loss": True})
            else:
                stats["raw_response_without_text"] += 1

            for tool_call in get_field(response, "tool_calls") or []:
                call_id, name, arguments = tool_call_fields(tool_call)
                loss = call_id not in failed_ids
                payload = {"name": name, "arguments": arguments}
                messages.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        "loss": loss,
                    }
                )
                stats["tool_calls"] += 1
                if not loss:
                    stats["failed_tool_calls_masked"] += 1
            continue

        if kind in {"tool_result", "ToolResult"}:
            result = get_field(item, "result")
            messages.append({"role": "tool_response", "content": text_content(result), "loss": False})
            stats["tool_responses"] += 1
            continue

        stats[f"unknown_history_type:{kind}"] += 1

    return messages, stats


def convert_tools(config_path: Path) -> str:
    config = load_json(config_path)
    converted: list[dict[str, Any]] = []
    for entry in config.get("tools") or []:
        if not isinstance(entry, dict):
            continue
        body = entry.get("body") if isinstance(entry.get("body"), dict) else entry
        name = str(body.get("name") or entry.get("name") or "")
        if not name:
            continue
        parameters = {
            "type": "object",
            "properties": body.get("properties") or {},
            "required": body.get("required") or [],
        }
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(body.get("description") or ""),
                    "parameters": parameters,
                },
            }
        )
    if not converted:
        raise ValueError(f"No tool definitions found in {config_path}")
    return json.dumps(converted, ensure_ascii=False, separators=(",", ":"))


def validate_record(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("Empty message list")
    if messages[0]["role"] not in {"user", "system"}:
        raise ValueError("Initial instruction/question is missing")
    valid_roles = {"system", "user", "assistant", "tool_call", "tool_response"}
    for index, message in enumerate(messages):
        if message.get("role") not in valid_roles:
            raise ValueError(f"Invalid role at message {index}: {message.get('role')!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"Non-string content at message {index}")
        if not isinstance(message.get("loss"), bool):
            raise ValueError(f"Non-boolean loss at message {index}")
        if message["role"] in {"system", "user", "tool_response"} and message["loss"]:
            raise ValueError(f"Non-assistant context is supervised at message {index}")
        if message["role"] == "user" and is_continue(message["content"]):
            previous_assistant = next(
                (m for m in reversed(messages[:index]) if m["role"] == "assistant"),
                None,
            )
            if previous_assistant is None or previous_assistant["loss"]:
                raise ValueError(f"Continue at message {index} has no masked preceding assistant")
    if not any(
        m["role"] == "tool_call"
        and m["loss"]
        and json.loads(m["content"]).get("name") == "submit_final_result"
        for m in messages
    ):
        raise ValueError("Successful submit_final_result call is missing from history")


def main() -> None:
    args = parse_args()
    roots = [path.expanduser().resolve() for path in args.input]
    output = args.output.expanduser().resolve()
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"Input directory does not exist: {root}")
    if len(set(roots)) != len(roots):
        raise SystemExit("Duplicate input directories")

    total_stats: Counter = Counter()
    errors: list[dict[str, str]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            for root in roots:
                for question_dir in question_dirs(root):
                    total_stats[f"{root.name}:question_dirs"] += 1
                    result_path = question_dir / "result.json"
                    if not result_path.is_file():
                        total_stats[f"{root.name}:missing_result_skipped"] += 1
                        continue
                    try:
                        result = load_json(result_path)
                        if not successful_result(result):
                            total_stats[f"{root.name}:unsuccessful_skipped"] += 1
                            continue

                        failed_ids, recorded_calls = failed_tool_call_ids(question_dir)
                        with final_history_path(question_dir).open("r", encoding="utf-8") as history_file:
                            history = json.load(history_file)
                        if not isinstance(history, list):
                            raise ValueError("history.json does not contain a list")

                        messages, record_stats = convert_history(history, failed_ids)
                        validate_record(messages)
                        config_path = question_dir / "turns" / "init" / "config.json"
                        if not config_path.is_file():
                            config_path = question_dir / "init" / "config.json"
                        tools = convert_tools(config_path)

                        represented_calls = sum(m["role"] == "tool_call" for m in messages)
                        if represented_calls != recorded_calls:
                            raise ValueError(
                                f"Tool call count mismatch: history={represented_calls}, "
                                f"execution records={recorded_calls}"
                            )

                        record = {"messages": messages, "tools": tools}
                        output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        total_stats.update(record_stats)
                        total_stats[f"{root.name}:written"] += 1
                        total_stats["written"] += 1
                        total_stats["messages"] += len(messages)
                        for message in messages:
                            total_stats[f"role:{message['role']}"] += 1
                        if args.limit is not None and total_stats["written"] >= args.limit:
                            break
                    except Exception as exc:  # Report every rejected trajectory.
                        errors.append({"question_dir": str(question_dir), "error": f"{type(exc).__name__}: {exc}"})
                        total_stats["conversion_errors"] += 1
                if args.limit is not None and total_stats["written"] >= args.limit:
                    break
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)

    report_path = output.with_suffix(output.suffix + ".report.json")
    report = {
        "inputs": [str(root) for root in roots],
        "output": str(output),
        "stats": dict(sorted(total_stats.items())),
        "errors": errors,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
