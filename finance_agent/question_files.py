"""Read plain question files or ID-preserving retry files."""

import re
from pathlib import Path


QUESTION_ID_PATTERN = re.compile(r"q\d{3,}")


def validate_question_ids(question_ids: list[str]) -> None:
    if any(
        not QUESTION_ID_PATTERN.fullmatch(qid) or int(qid[1:]) < 1
        for qid in question_ids
    ):
        raise ValueError("Question IDs must look like q001, q116, or q1000")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate question IDs in the question file")


def load_question_file(path: Path) -> tuple[list[str], list[str]]:
    """Accept either one plain question per line or qNNN<TAB>question.

    Reject mixed formats rather than accidentally assigning conflicting IDs.
    The ID is metadata only and is never included in the model prompt.
    """
    questions = []
    question_ids = []
    modes = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        prefix, separator, question = raw.strip(" \r\n").partition("\t")
        if separator and re.fullmatch(r"q\d+", prefix):
            modes.add("explicit")
            if not question.strip():
                raise ValueError(f"Empty question on line {line_number}")
            question_ids.append(prefix)
            questions.append(question.strip())
        else:
            modes.add("plain")
            question_ids.append(f"q{len(questions) + 1:03d}")
            questions.append(line)
    if len(modes) > 1:
        raise ValueError("Do not mix plain questions with qNNN<TAB>question records")
    if not questions:
        raise ValueError(f"No questions found in {path}")
    validate_question_ids(question_ids)
    return questions, question_ids
