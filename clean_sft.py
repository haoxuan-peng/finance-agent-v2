import argparse
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

from swift import get_processor, get_template

from normalize_sft_loss import mask_failed_tool_calls as apply_failed_tool_masks, normalize_sample_loss


CONTINUE_PATTERN = re.compile(
    r"^\s*(continue\.?|please continue\.?|go on\.?)\s*$",
    re.IGNORECASE,
)


def dumps(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_tools(tools):
    """
    SWIFT 推荐 tools 保存成 JSON string。
    返回:
      normalized_tools: str
      tool_names: set[str]
    """
    if tools is None:
        return None, set()

    if isinstance(tools, str):
        parsed = json.loads(tools)
    elif isinstance(tools, list):
        parsed = tools
    else:
        raise ValueError(f"Invalid tools type: {type(tools)}")

    if not isinstance(parsed, list):
        raise ValueError("tools must decode to a list")

    names = set()

    for tool in parsed:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be a dict")

        fn = tool.get("function", {})
        name = fn.get("name")

        if name:
            names.add(name)

    return dumps(parsed), names


def normalize_content(content):
    """
    保证 message.content 尽量统一为 string。
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (dict, list)):
        return dumps(content)

    return str(content)


def clean_messages(messages):
    """
    删除 Continue 类人工 continuation。

    如果结构是:
        assistant A
        user Continue.
        assistant B

    转成:
        assistant A + B
    """
    cleaned = []

    removed_continue = 0
    merged_assistant = 0

    # 表示刚删除了一个 Continue user message
    continuation_removed = False

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("message must be a dict")

        role = msg.get("role")
        content = normalize_content(msg.get("content"))

        if not role:
            raise ValueError("message missing role")

        # 删除 Continue.
        if role == "user" and CONTINUE_PATTERN.match(content):
            removed_continue += 1
            continuation_removed = True
            continue

        new_msg = dict(msg)
        new_msg["content"] = content

        # assistant -> Continue -> assistant
        # 合并为同一条 reasoning
        if (
            continuation_removed
            and role == "assistant"
            and cleaned
            and cleaned[-1].get("role") == "assistant"
            and cleaned[-1].get("loss") == new_msg.get("loss")
            and cleaned[-1].get("loss_scale") == new_msg.get("loss_scale")
        ):
            prev = cleaned[-1]["content"].rstrip()
            curr = content.lstrip()

            cleaned[-1]["content"] = prev + "\n" + curr

            merged_assistant += 1
            continuation_removed = False
            continue

        continuation_removed = False
        cleaned.append(new_msg)

    return cleaned, removed_continue, merged_assistant


def validate_tool_calls(messages, tool_names):
    """
    检查 tool_call:
      {
        "role": "tool_call",
        "content": "{\"name\":\"xxx\",\"arguments\":{...}}"
      }
    """
    tool_call_count = 0
    tool_response_count = 0
    unknown_tools = []
    normalized_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # 兼容有些数据写 role=tool
        if role == "tool":
            role = "tool_response"

        new_msg = dict(msg)
        new_msg["role"] = role

        if role == "tool_call":
            tool_call_count += 1

            try:
                if isinstance(content, str):
                    call = json.loads(content)
                else:
                    call = content
            except Exception as e:
                raise ValueError(
                    f"Invalid tool_call JSON: {content[:200]}"
                ) from e

            if not isinstance(call, dict):
                raise ValueError(
                    "tool_call content must decode to dict"
                )

            name = call.get("name")
            arguments = call.get("arguments")

            if not name:
                raise ValueError(
                    f"tool_call missing name: {call}"
                )

            if arguments is None:
                call["arguments"] = {}

            if tool_names and name not in tool_names:
                unknown_tools.append(name)

            # 统一重新 dump 成 JSON string
            new_msg["content"] = dumps(call)

        elif role == "tool_response":
            tool_response_count += 1
            new_msg["content"] = normalize_content(content)

        else:
            new_msg["content"] = normalize_content(content)

        normalized_messages.append(new_msg)

    if unknown_tools:
        raise ValueError(
            f"Unknown tools called: {sorted(set(unknown_tools))}"
        )

    return (
        normalized_messages,
        tool_call_count,
        tool_response_count,
    )


def get_assistant_stats(messages):
    lengths = []

    for msg in messages:
        if msg["role"] == "assistant":
            lengths.append(len(msg.get("content", "")))

    return {
        "assistant_messages": len(lengths),
        "max_assistant_chars": max(lengths, default=0),
        "assistant_chars": sum(lengths),
    }


def clean_sample(sample, *, mask_failed_tool_calls=True):
    """
    对一条 trajectory 做 deterministic cleaning。
    """
    if not isinstance(sample, dict):
        raise ValueError("sample must be dict")

    if "messages" not in sample:
        raise ValueError("missing messages")

    if not isinstance(sample["messages"], list):
        raise ValueError("messages must be list")

    result = dict(sample)

    # -------- tools --------
    normalized_tools, tool_names = parse_tools(
        result.get("tools")
    )

    if normalized_tools is not None:
        result["tools"] = normalized_tools

    # -------- continuation --------
    messages, removed_continue, merged_assistant = (
        clean_messages(result["messages"])
    )

    # -------- tool calls --------
    (
        messages,
        num_tool_calls,
        num_tool_responses,
    ) = validate_tool_calls(
        messages,
        tool_names,
    )

    result["messages"] = messages
    # A merged JSONL must have the same message fields in every Arrow chunk.
    # Preserve explicit masks from other sources while filling missing ones.
    result = normalize_sample_loss(result)
    mask_stats = {"failed_tool_calls": 0, "masked_tool_groups": 0, "newly_masked_tool_calls": 0}
    if mask_failed_tool_calls:
        result, mask_stats = apply_failed_tool_masks(result)

    assistant_stats = get_assistant_stats(messages)

    stats = {
        "removed_continue": removed_continue,
        "merged_assistant": merged_assistant,
        "tool_calls": num_tool_calls,
        "tool_responses": num_tool_responses,
        **assistant_stats,
        **mask_stats,
    }

    return result, stats


def percentile(values, p):
    if not values:
        return 0
    return int(np.percentile(values, p))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Cleaned JSONL",
    )

    parser.add_argument(
        "--rejected",
        default="rejected.jsonl",
        help="Rejected samples",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Local Qwen3.5 model path",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help=(
            "0 = only analyze, don't reject by length; "
            "e.g. 16384 / 32768 / 65536"
        ),
    )

    parser.add_argument(
        "--max-assistant-chars",
        type=int,
        default=0,
        help=(
            "Optional: reject if any single assistant "
            "message exceeds this many chars. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--mask-failed-tool-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask failed tool-call groups using explicit status or recognizable error returns",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_path = Path(args.rejected)

    print("Loading SWIFT processor/template...")
    print(f"Model: {args.model}")

    processor = get_processor(args.model)

    template = get_template(
        processor,
        agent_template="qwen3_5",
    )

    template.set_mode("train")

    print(
        f"Agent template: "
        f"{getattr(template, '_agent_template', 'unknown')}"
    )

    total = 0
    kept = 0
    rejected = 0

    token_lengths = []
    kept_lengths = []

    reason_counter = Counter()

    total_continue_removed = 0
    total_assistant_merged = 0
    total_tool_calls = 0
    total_tool_responses = 0
    total_failed_tool_calls = 0
    total_newly_masked_tool_calls = 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rejected_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as fin,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as fout,
        rejected_path.open(
            "w",
            encoding="utf-8",
        ) as freject,
    ):
        for line_no, line in enumerate(
            tqdm(fin),
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            total += 1

            sample_id = None

            try:
                sample = json.loads(line)
                sample_id = sample.get(
                    "id",
                    f"line_{line_no}",
                )

                cleaned, stats = clean_sample(
                    sample, mask_failed_tool_calls=args.mask_failed_tool_calls,
                )

                # 用 SWIFT 真正的 training template encode
                encoded = template.encode(cleaned)

                input_ids = encoded["input_ids"]

                num_tokens = len(input_ids)
                token_lengths.append(num_tokens)

                # ---------- optional filters ----------

                if (
                    args.max_assistant_chars > 0
                    and stats["max_assistant_chars"]
                    > args.max_assistant_chars
                ):
                    raise RuntimeError(
                        "FILTER:max_assistant_chars:"
                        f"{stats['max_assistant_chars']}"
                    )

                if (
                    args.max_tokens > 0
                    and num_tokens > args.max_tokens
                ):
                    raise RuntimeError(
                        f"FILTER:max_tokens:{num_tokens}"
                    )

                # ---------- save ----------
                fout.write(
                    json.dumps(
                        cleaned,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                kept += 1
                kept_lengths.append(num_tokens)

                total_continue_removed += stats[
                    "removed_continue"
                ]

                total_assistant_merged += stats[
                    "merged_assistant"
                ]

                total_tool_calls += stats[
                    "tool_calls"
                ]

                total_tool_responses += stats[
                    "tool_responses"
                ]
                total_failed_tool_calls += stats["failed_tool_calls"]
                total_newly_masked_tool_calls += stats["newly_masked_tool_calls"]

            except Exception as e:
                rejected += 1

                error = str(e)

                if error.startswith("FILTER:max_tokens"):
                    reason = "too_long"
                elif error.startswith(
                    "FILTER:max_assistant_chars"
                ):
                    reason = "assistant_too_long"
                elif "Unknown tools" in error:
                    reason = "unknown_tool"
                elif "tool_call" in error:
                    reason = "invalid_tool_call"
                else:
                    reason = "invalid_sample"

                reason_counter[reason] += 1

                record = {
                    "line": line_no,
                    "id": sample_id,
                    "reason": reason,
                    "error": error,
                }

                freject.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print("\n" + "=" * 60)
    print("DATASET REPORT")
    print("=" * 60)

    print(f"Total samples:       {total}")
    print(f"Kept samples:        {kept}")
    print(f"Rejected samples:    {rejected}")

    if total:
        print(
            f"Keep rate:           "
            f"{kept / total * 100:.2f}%"
        )

    print()
    print(f"Continue removed:    {total_continue_removed}")
    print(f"Assistant merged:    {total_assistant_merged}")
    print(f"Tool calls:          {total_tool_calls}")
    print(f"Tool responses:      {total_tool_responses}")
    print(f"Failed tool calls:   {total_failed_tool_calls}")
    print(f"Newly masked calls:  {total_newly_masked_tool_calls}")

    if token_lengths:
        print("\nTOKEN LENGTH — ALL VALID ENCODED")
        print("-" * 40)
        print(f"p50:  {percentile(token_lengths, 50):,}")
        print(f"p90:  {percentile(token_lengths, 90):,}")
        print(f"p95:  {percentile(token_lengths, 95):,}")
        print(f"p99:  {percentile(token_lengths, 99):,}")
        print(f"max:  {max(token_lengths):,}")
        print(f"mean: {int(np.mean(token_lengths)):,}")

        for threshold in (
            8192,
            16384,
            32768,
            65536,
            131072,
        ):
            n = sum(
                x > threshold
                for x in token_lengths
            )

            print(
                f">{threshold:6,d}: "
                f"{n:6,d} "
                f"({n / len(token_lengths) * 100:.2f}%)"
            )

    if kept_lengths:
        print("\nTOKEN LENGTH — KEPT")
        print("-" * 40)
        print(f"p50: {percentile(kept_lengths, 50):,}")
        print(f"p90: {percentile(kept_lengths, 90):,}")
        print(f"p95: {percentile(kept_lengths, 95):,}")
        print(f"p99: {percentile(kept_lengths, 99):,}")
        print(f"max: {max(kept_lengths):,}")

    if reason_counter:
        print("\nREJECTION REASONS")
        print("-" * 40)

        for reason, count in reason_counter.most_common():
            print(f"{reason:25s}: {count}")

    print("\nOutput:")
    print(f"  Cleaned:  {output_path}")
    print(f"  Rejected: {rejected_path}")


if __name__ == "__main__":
    main()
