#!/usr/bin/env python3
"""Make loss fields consistent before ms-swift loads a merged JSONL dataset.

With --loss_scale default, assistant/tool_call messages are supervised and
context messages are not. Existing boolean masks are preserved. Missing/null
masks are made explicit so every JSONL chunk infers the same message schema.
Failed tool-call groups are masked by default. A uniform boolean bookkeeping
column prevents later cleaning from reclassifying calls checked against source
status; this column is not part of the model-visible conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SUPERVISED_ROLES = {"assistant", "tool_call"}
CONTEXT_ROLES = {"system", "user", "tool", "tool_response"}

# Match runtime error formats, not the word "error" inside retrieved documents.
TOOL_ERROR_PREFIX = re.compile(
    r"^\s*(?:ERROR\s*:|Error code\s*:|Timeout error\b|"
    r"Missing required parameters for\b|Unparseable tool call args\s*:|"
    r"Cannot connect to host\b|Cannot write to closing transport\b|"
    r"['\"]utf-8['\"] codec can't decode byte\b)",
    re.IGNORECASE,
)


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def tool_result_failed(record: Any, *, content: str | None = None) -> bool:
    """Prefer explicit runtime status; use narrow text checks if it is absent."""
    candidates = [record, _field(record, "tool_output"), _field(record, "extra")]
    successes = [_field(candidate, "success") for candidate in candidates]
    # Runtime success flags are authoritative, even if a retrieved page starts
    # with an error-like string. Any explicit failure wins conflicting records.
    if any(value is False for value in successes):
        return True
    if any(value is True for value in successes):
        return False
    if any(_field(candidate, "error") for candidate in candidates):
        return True
    if content is None:
        return False
    if TOOL_ERROR_PREFIX.match(content):
        return True
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if isinstance(payload, dict):
        return tool_result_failed(payload)
    return False


def normalize_sample_loss(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with explicit boolean loss flags; preserve existing masks."""
    if not isinstance(sample, dict):
        raise ValueError("sample must be a JSON object")
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be a JSON object")
        role = message.get("role")
        if role not in SUPERVISED_ROLES | CONTEXT_ROLES:
            raise ValueError(f"message {index} has unsupported role: {role!r}")
        loss = message.get("loss")
        if loss is not None and not isinstance(loss, bool):
            raise ValueError(f"message {index} loss must be boolean or null")
        normalized.append(
            {**message, "loss": role in SUPERVISED_ROLES if loss is None else loss}
        )
    applied = sample.get("tool_loss_masking_applied", False)
    if not isinstance(applied, bool):
        raise ValueError("tool_loss_masking_applied must be boolean")
    return {**sample, "messages": normalized, "tool_loss_masking_applied": applied}


def mask_failed_tool_calls(
    sample: dict[str, Any], *, inspect_responses: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Mask failed native tool-call groups without dropping context.

    SWIFT 4.5.x merges contiguous tool_call messages and keeps the first loss
    flag, so a group containing any failed/already-masked call must be masked
    together. For exports without IDs, calls and responses are paired in their
    original order. Source converters can supply masks by ID and disable the
    text fallback with inspect_responses=False.
    """
    result = normalize_sample_loss(sample)
    inspect_responses = inspect_responses and not result["tool_loss_masking_applied"]
    messages = result["messages"]
    stats = {"failed_tool_calls": 0, "masked_tool_groups": 0, "newly_masked_tool_calls": 0}
    index = 0
    while index < len(messages):
        if messages[index]["role"] != "tool_call":
            index += 1
            continue
        start = index
        while index < len(messages) and messages[index]["role"] == "tool_call":
            index += 1
        calls = messages[start:index]
        response_end = index
        while response_end < len(messages) and messages[response_end]["role"] in {"tool", "tool_response"}:
            response_end += 1
        responses = messages[index:response_end]
        failed = False
        for offset, call in enumerate(calls):
            error = False
            if inspect_responses and offset < len(responses):
                response = responses[offset]
                error = tool_result_failed(response, content=response.get("content"))
            stats["failed_tool_calls"] += int(error)
            failed |= error or call["loss"] is False
        if failed:
            stats["masked_tool_groups"] += 1
            for call in calls:
                stats["newly_masked_tool_calls"] += int(call["loss"] is True)
                call["loss"] = False
        index = response_end
    result["tool_loss_masking_applied"] = True
    return result, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Merged SFT JSONL")
    parser.add_argument("--output", required=True, type=Path, help="New normalized JSONL")
    parser.add_argument(
        "--mask-failed-tool-calls", action=argparse.BooleanOptionalAction, default=True,
        help="Mask failed tool-call groups using explicit status or recognizable error returns",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.output.exists():
        parser.error("output already exists; choose a new output path")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    filled = Counter()
    preserved_false = Counter()
    mask_stats = Counter()
    temporary_path = None
    try:
        with args.input.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent,
            prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    normalized = normalize_sample_loss(sample)
                    if args.mask_failed_tool_calls:
                        normalized, stats = mask_failed_tool_calls(normalized)
                        mask_stats.update(stats)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
                for message in sample["messages"]:
                    if message.get("loss") is None:
                        filled[message["role"]] += 1
                    elif message["loss"] is False:
                        preserved_false[message["role"]] += 1
                output.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows += 1
        # Hard-link publication refuses to replace an existing file, including
        # one created while this process was reading the source.
        os.link(temporary_path, args.output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"rows={rows} output={args.output}")
    print(f"filled_loss_by_role={dict(filled)}")
    print(f"preserved_false_by_role={dict(preserved_false)}")
    print(f"tool_failure_masks={dict(mask_stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
