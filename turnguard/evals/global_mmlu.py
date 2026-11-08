"""All English/Hindi Global-MMLU-Lite test items through shared chat generation."""

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from turnguard.evals.common import auditkit_cache
from turnguard.metrics import rate
from turnguard.models import Generator, GenRequest

EXTRACTION_VERSION = "mmlu-answer-v1"
LETTERS = "ABCD"
HINDI_LETTERS = {"ए": "A", "बी": "B", "सी": "C", "डी": "D"}
_LABEL = r"(?:[A-Da-d]|ए|बी|सी|डी)"
_FINAL = re.compile(
    rf"(?:final\s+answer|अंतिम\s+(?:उत्तर|जवाब))\s*(?:is|है)?\s*[:=\-]?\s*"
    rf"[\s(*`\[]*({_LABEL})[\s)*`\].!।]*(?:है[।.]*)?\s*$",
    re.IGNORECASE,
)
_EXPLICIT = re.compile(
    rf"(?:\b(?:answer|option|choice)\b|(?:सही\s+)?(?:उत्तर|जवाब|विकल्प))"
    rf"\s*(?:is|है)?\s*[:=\-]?\s*[\s(*`\[]*({_LABEL})(?![A-Za-z\u0900-\u0963\u0966-\u097f])",
    re.IGNORECASE,
)
_ASCII = re.compile(r"(?<![A-Za-z0-9_])([ABCD])(?![A-Za-z0-9_])")
_HINDI = re.compile(
    r"(?<![\w\u0900-\u0963\u0966-\u097f])(ए|बी|सी|डी)(?![\w\u0900-\u0963\u0966-\u097f])"
)


@dataclass(frozen=True)
class MMLUItem:
    id: str
    source_id: str
    language: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str
    subject: str = ""


def _letter(text: str) -> str:
    return HINDI_LETTERS[text] if text in HINDI_LETTERS else text.upper()


def extract_answer(response: str) -> str | None:
    """Parse bounded Latin/Hindi answer markers, never the first letter of prose.

    An explicit terminal final answer wins over reasoning. Otherwise multiple
    distinct letters are ambiguous, including 'B or C'; no option-text guessing.
    Lowercase letters count only alone or after an answer label (avoid 'a').
    """
    text = unicodedata.normalize("NFKC", response).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if match := _FINAL.search(text):
        return _letter(match.group(1))
    clean = text.strip(" \n\r\t()[]*`.!।")
    if clean in HINDI_LETTERS or clean.upper() in LETTERS and len(clean) == 1:
        return _letter(clean)
    explicit = {_letter(match.group(1)) for match in _EXPLICIT.finditer(text)}
    candidates = (
        explicit
        | set(_ASCII.findall(text))
        | {_letter(token) for token in _HINDI.findall(text)}
    )
    if len(candidates) != 1:
        return None
    letter = candidates.pop()
    if explicit:
        return letter
    # A lone English article in prose is not a selected option. Accept an
    # unlabelled answer only at a terminal position or in familiar answer syntax.
    token = (
        "(?:"
        + re.escape(letter)
        + "|"
        + re.escape(
            next(hindi for hindi, latin in HINDI_LETTERS.items() if latin == letter)
        )
        + ")"
    )
    terminal = re.search(token + r"[\s)\]*`.!।]*$", text)
    leading = re.match(
        r"^[\s(*`\[]*"
        + token
        + r"(?:[)\]*`.:,\-]+|\s+(?:(?:is\s+)?correct\b|सही(?:\s|$)|है[।.!]?$))",
        text,
    )
    return letter if terminal or leading else None


def parse_items(rows: list[dict], language: str) -> list[MMLUItem]:
    if language not in ("en", "hi"):
        raise ValueError("Global-MMLU supports en and hi")
    items = []
    for row in rows:
        source_id = row.get("sample_id")
        question = row.get("question")
        choices = tuple(row.get("option_" + letter.lower()) for letter in LETTERS)
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Global-MMLU row needs a sample_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Global-MMLU row needs a question")
        if not all(isinstance(choice, str) and choice.strip() for choice in choices):
            raise ValueError("Global-MMLU row needs four nonempty options")
        if not isinstance(row.get("answer"), str) or row["answer"] not in tuple(
            LETTERS
        ):
            raise ValueError("Global-MMLU answer must be A, B, C or D")
        items.append(
            MMLUItem(
                id=f"mmlu-{language}-{source_id}",
                source_id=source_id,
                language=language,
                question=question,
                choices=choices,
                answer=row["answer"],
                subject=row.get("subject", ""),
            )
        )
    if len({item.id for item in items}) != len(items):
        raise ValueError("Global-MMLU sample IDs must be unique within a language")
    return items


def load_items(config: dict, *, cache_dir: Path) -> list[MMLUItem]:
    """Load pinned test sets only; fixtures bypass all dataset/network machinery."""
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Global-MMLU revision must be an immutable HF commit hash")
    if config["languages"] != ["en", "hi"] or config["split"] != "test":
        raise ValueError("Global-MMLU requires all en/hi test items")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Global-MMLU requires pip install -e '.[eval]'") from error
    items = []
    for language in config["languages"]:
        rows = load_dataset(
            config["dataset"],
            name=language,
            split=config["split"],
            revision=config["revision"],
            cache_dir=str(cache_dir),
        )
        group = parse_items(list(rows), language)
        if len(group) != config["expected_counts"][language]:
            raise ValueError(
                f"Pinned Global-MMLU {language}: expected "
                f"{config['expected_counts'][language]} items, got {len(group)}"
            )
        items.extend(group)
    return items


def select_items(
    items: list[MMLUItem], limit: int | None, *, seed: int = 1234
) -> list[MMLUItem]:
    """A smoke limit counts total questions, interleaving seeded language strata."""
    if limit is None:
        return list(items)
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    rng = random.Random(seed)
    groups = [
        [item for item in items if item.language == lang] for lang in ("en", "hi")
    ]
    for group in groups:
        rng.shuffle(group)
    chosen = []
    while any(groups) and len(chosen) < limit:
        for group in groups:
            if group and len(chosen) < limit:
                chosen.append(group.pop())
    return chosen


def render_question(item: MMLUItem, instructions: dict[str, str]) -> str:
    options = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(LETTERS, item.choices, strict=True)
    )
    return f"{item.question}\n\n{options}\n\n{instructions[item.language]}"


def evaluate_mmlu(
    items: list[MMLUItem],
    generator: Generator,
    *,
    instructions: dict[str, str],
    system_prompt: str | None = None,
    batch_size: int = 8,
) -> tuple[list[dict], dict]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len({item.id for item in items}) != len(items):
        raise ValueError("Global-MMLU IDs must be unique")
    records = []
    for offset in range(0, len(items), batch_size):
        chunk = items[offset : offset + batch_size]
        requests = [
            GenRequest(
                item.id,
                1,
                (
                    [{"role": "system", "content": system_prompt}]
                    if system_prompt is not None
                    else []
                )
                + [{"role": "user", "content": render_question(item, instructions)}],
            )
            for item in chunk
        ]
        for _ in range(2):
            try:
                outputs = generator.generate(requests)
                if (
                    not isinstance(outputs, list)
                    or len(outputs) != len(chunk)
                    or not all(isinstance(text, str) for text in outputs)
                ):
                    raise ValueError("Invalid generator batch")
                break
            except Exception:
                # Keep provider/model exceptions out of console output.
                pass
        else:
            outputs = [None] * len(chunk)
        for item, request, output in zip(chunk, requests, outputs, strict=True):
            predicted = extract_answer(output) if output is not None else None
            record = {
                "id": item.id,
                "source_id": item.source_id,
                "language": item.language,
                "subject": item.subject,
                "messages": request.messages,
                "response": output,
                "target": item.answer,
                "predicted": predicted,
                "correct": predicted == item.answer if output is not None else None,
                "unparseable": output is not None and predicted is None,
                "model": generator.name,
            }
            if output is None:
                record["generation_error"] = "generation_failed"
            records.append(record)
    metrics = {"total": len(records), "languages": {}}
    for language in ("en", "hi"):
        group = [row for row in records if row["language"] == language]
        graded = [row for row in group if row["correct"] is not None]
        metrics["languages"][language] = {
            "total": len(group),
            "graded": len(graded),
            "generation_errors": len(group) - len(graded),
            "unparseable": sum(row["unparseable"] for row in group),
            "accuracy": rate(sum(row["correct"] for row in graded), len(graded)),
        }
    return records, metrics


def auditkit_scores(records: list[dict], *, cache_dir: Path) -> dict:
    """Score precomputed raw answers through AuditKIT, retaining its run records."""
    import auditkit as ak

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="accuracy")
    def accuracy(sample, output):
        letter = extract_answer(output)
        return ak.Score(
            name="accuracy",
            value=float(letter == sample.target),
            label=letter or "unparseable",
            metadata={"unparseable": letter is None},
        )

    results = {}
    for language in ("en", "hi"):
        samples = [
            ak.Sample(
                id=row["id"],
                input=row["messages"][-1]["content"],
                target=row["target"],
                actual_output=row["response"],
                metadata={"language": language},
            )
            for row in records
            if row["language"] == language and row["correct"] is not None
        ]
        if not samples:
            continue
        with auditkit_cache(cache_dir):
            result = ak.evaluate(
                samples,
                model="precomputed",
                scorers=[accuracy],
                config=ak.RunConfig(
                    seed=1234,
                    track_performance=False,
                    extra={"extraction_version": EXTRACTION_VERSION},
                ),
                verbose=False,
            )
        results[language] = result.to_dict()
    return results
