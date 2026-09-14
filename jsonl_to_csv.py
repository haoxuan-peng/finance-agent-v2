#!/usr/bin/env python3
"""Convert FF-style JSONL to the four-column public.csv schema."""

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = ["Question", "Answer", "Question Type", "Rubric"]


def convert_record(record: dict, question_type: str | None = None) -> dict:
    question = record.get("query")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("query 必须是非空字符串")

    answer = record.get("answer") or ""
    source_type = record.get("question_type") or ""
    if not isinstance(answer, str) or not isinstance(source_type, str):
        raise ValueError("answer 和 question_type 必须是字符串或 null")

    source_rubrics = record.get("rubrics", [])
    if not isinstance(source_rubrics, list):
        raise ValueError("rubrics 必须是数组")

    rubrics = []
    for index, rubric in enumerate(source_rubrics, start=1):
        if not isinstance(rubric, dict):
            raise ValueError(f"第 {index} 条 rubric 必须是对象")
        criteria = rubric.get("rubric_text")
        if not isinstance(criteria, str) or not criteria.strip():
            raise ValueError(f"第 {index} 条 rubric 缺少非空 rubric_text")
        # No reference answer exists in FF_test, so do not invent contradiction rules.
        rubrics.append({"operator": "correctness", "criteria": criteria})

    return {
        "Question": question,
        "Answer": answer,
        "Question Type": source_type if question_type is None else question_type,
        "Rubric": json.dumps(rubrics, ensure_ascii=False),
    }


def convert(input_path: Path, output_path: Path, question_type: str | None = None,
            overwrite: bool = False) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出不能是同一个文件")

    # Validate all records before creating the output file.
    rows = []
    with input_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("记录必须是 JSON 对象")
                rows.append(convert_record(record, question_type))
            except ValueError as error:
                raise ValueError(f"第 {line_number} 行：{error}") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM matches the sample CSV and helps Excel display Chinese correctly.
    with output_path.open("w" if overwrite else "x", encoding="utf-8-sig",
                          newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 FF 格式 JSONL 转换为 Question/Answer/Question Type/Rubric 四列 CSV。"
    )
    parser.add_argument("input", type=Path, help="输入 JSONL 文件")
    parser.add_argument("output", nargs="?", type=Path,
                        help="输出 CSV 文件（默认同目录、同名 .csv）")
    parser.add_argument("--question-type", help="为全部题目指定固定类型；默认保留源类型，没有则留空")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出文件")
    args = parser.parse_args()
    if args.input.suffix.lower() != ".jsonl":
        parser.error("输入文件必须是 .jsonl")
    output_path = args.output or args.input.with_suffix(".csv")
    try:
        count = convert(args.input, output_path, args.question_type, args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f"转换失败：{error}\n")
    print(f"已转换 {count} 道题目 -> {output_path}")


if __name__ == "__main__":
    main()
