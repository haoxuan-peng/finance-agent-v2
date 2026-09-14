"""Prepare an ID-preserving question file for retrying unfinished rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .question_files import QUESTION_ID_PATTERN, load_question_file


def prepare_remaining(
    question_file: Path,
    logs_path: list[Path],
    output_file: Path,
) -> dict[str, Any]:
    questions, question_ids = load_question_file(question_file)
    if question_file.resolve() == output_file.resolve():
        raise ValueError("Output file must not overwrite the original question file")
    known_ids = set(question_ids)
    completed = set()
    unreadable_results = 0
    unknown_ids = set()
    for root in logs_path:
        if not root.is_dir():
            raise ValueError(f"Log directory does not exist: {root}")
        for result_file in root.rglob("result.json"):
            qid = result_file.parent.name
            if not QUESTION_ID_PATTERN.fullmatch(qid) or "turns" in result_file.parts:
                continue
            if qid not in known_ids:
                unknown_ids.add(qid)
                continue
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                unreadable_results += 1
                continue
            if not isinstance(result, dict):
                unreadable_results += 1
                continue
            answer = result.get("final_answer")
            if (
                result.get("final_error") is None
                and isinstance(answer, str)
                and answer.strip()
            ):
                completed.add(qid)

    remaining = [
        (qid, question)
        for qid, question in zip(question_ids, questions)
        if qid not in completed
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{qid}\t{question}\n" for qid, question in remaining), encoding="utf-8"
    )
    temporary.replace(output_file)
    return {
        "total": len(questions),
        "completed": len(completed),
        "remaining": len(remaining),
        "remaining_ids": [qid for qid, _ in remaining],
        "unreadable_results": unreadable_results,
        "unknown_question_ids": sorted(unknown_ids),
        "output_file": str(output_file),
    }


def main_sync() -> None:
    parser = argparse.ArgumentParser(
        description="Find unfinished rollouts and preserve original IDs in a retry TXT"
    )
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument(
        "--logs-path",
        type=Path,
        nargs="+",
        required=True,
        help="Only runs using this same dataset and original question IDs",
    )
    parser.add_argument("--output-file", type=Path, default=Path("data/remaining.txt"))
    args = parser.parse_args()
    try:
        summary = prepare_remaining(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Could not prepare remaining questions: {error}\n")
    print(f"Total questions: {summary['total']}")
    print(f"Completed: {summary['completed']}")
    print(f"Remaining: {summary['remaining']}")
    print(f"Remaining question IDs: {summary['remaining_ids']}")
    print(f"Output: {summary['output_file']}")
    if summary["unreadable_results"]:
        print(
            f"Warning: {summary['unreadable_results']} unreadable results were ignored."
        )
    if summary["unknown_question_ids"]:
        print(
            f"Warning: unknown question IDs were ignored: {summary['unknown_question_ids']}"
        )
    print(
        "Retry using this TXT; evaluate the new trajectories against the original full dataset."
    )


if __name__ == "__main__":
    main_sync()
