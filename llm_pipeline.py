from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama

logger = logging.getLogger("ipip_pipeline")


MODELS = ["qwen2.5:3b", "llama3.2:3b", "aisingapore/Llama-SEA-LION-v3.5-8B-R"]
LANGUAGE_FILES = {
    "english": Path("ipip-50_english.json"),
    "burmese": Path("ipip-50_burmese.json"),
}
SESSIONS_PER_COMBO = 20
OUTPUT_CSV = Path("llm_persona_results.csv")
CSV_COLUMNS = [
    "session_id",
    "model_name",
    "language",
    "item_id",
    "trait",
    "keying",
    "raw_output",
    "parsed_score",
]
GENERATE_OPTIONS = {"temperature": 0.7, "num_predict": 8}

BURMESE_DIGITS = {d: i for i, d in enumerate("၀၁၂၃၄၅၆၇၈၉")}
SCORE_RE = re.compile(r"[1-5၁-၅]")

ENGLISH_PROMPT = """\
You are completing a personality questionnaire. Rate how accurately the
statement describes you on this scale:
1 = Very Inaccurate
2 = Moderately Inaccurate
3 = Neither Accurate Nor Inaccurate
4 = Moderately Accurate
5 = Very Accurate

Statement: "{text}"

Respond with EXACTLY ONE integer from 1 to 5 and nothing else. Do not add
punctuation, words, explanations, or introductory phrases.\
"""

BURMESE_PROMPT = """\
ဒါက ကိုယ်ရည်ကိုယ်သွေး စစ်တမ်းမေးခွန်းတစ်ခုဖြစ်ပါတယ်။ အောက်ပါဖော်ပြချက်က
သင့်ကိုယ်သင် မည်မျှကိုက်ညီကြောင်း အောက်ပါအဆင့်ဖြင့် သတ်မှတ်ပါ -
၁ = လုံးဝမကိုက်ညီပါ
၂ = အတန်အသင့်မကိုက်ညီပါ
၃ = ကြားနေ
၄ = အတန်အသင့်ကိုက်ညီပါသည်
၅ = လုံးဝကိုက်ညီပါသည်

ဖော်ပြချက် - "{text}"

၁ မှ ၅ အတွင်း ကိန်းဂဏန်းတစ်လုံးတည်းကိုသာ ဖြေဆိုပါ။ ပုဒ်ဖြတ်ပုဒ်ရပ်၊ စကားလုံး၊
ရှင်းလင်းချက် သို့မဟုတ် နိဒါန်းစကား မထည့်ပါနှင့်။\
"""

PROMPT_TEMPLATES = {"english": ENGLISH_PROMPT, "burmese": BURMESE_PROMPT}


@dataclass(frozen=True)
class Item:
    id: int
    text: str
    trait: str
    keying: str


def load_items(language: str) -> list[Item]:
    """Load and validate the IPIP-50 items for the given language."""
    path = LANGUAGE_FILES[language]
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [Item(**q) for q in data["questions"]]
    if len(items) != 50:
        logger.warning("%s: expected 50 items, found %d", language, len(items))
    return items


def parse_score(raw: str) -> Optional[int]:
    """Return the first western or Burmese Likert digit, if present."""
    match = SCORE_RE.search(raw or "")
    if not match:
        return None

    char = match.group()
    return BURMESE_DIGITS.get(char, int(char) if char.isdigit() else None)


def query_model(model: str, prompt: str) -> str:
    """Run one stateless Ollama completion."""
    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options=GENERATE_OPTIONS,
        keep_alive="5m",
    )
    return (response.get("response") or "").strip()


def evaluate_item(model: str, language: str, item: Item) -> tuple[Optional[str], Optional[int]]:
    """Evaluate one questionnaire item and return raw text plus parsed score."""
    try:
        prompt = PROMPT_TEMPLATES[language].format(text=item.text)
        raw = query_model(model, prompt)
        score = parse_score(raw)
        if score is None:
            logger.warning("parse failed (item %d): %r", item.id, raw)
        return raw, score
    except Exception:  # noqa: BLE001 - resilience is required across 6,000 runs
        logger.exception("inference failed (model=%s lang=%s item=%d)", model, language, item.id)
        return None, None


def prune_incomplete_sessions() -> set[tuple[str, str, int]]:
    """Drop interrupted sessions and return the already-complete session keys."""
    if not (OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0):
        return set()

    with OUTPUT_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    counts: Counter[tuple[str, str, int]] = Counter(
        (r["model_name"], r["language"], int(r["session_id"])) for r in rows
    )

    # Preserve unparseable text, but rerun sessions with empty outputs.
    failed = {
        (r["model_name"], r["language"], int(r["session_id"]))
        for r in rows
        if (r["raw_output"] or "") == ""
    }
    complete = {key for key, n in counts.items() if n == 50 and key not in failed}

    kept = [
        r for r in rows
        if (r["model_name"], r["language"], int(r["session_id"])) in complete
    ]
    if len(kept) != len(rows):
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(kept)
        logger.info(
            "pruned %d rows from incomplete sessions",
            len(rows) - len(kept),
        )
    return complete


def run_pipeline() -> None:
    """Execute the full resumable evaluation grid."""
    datasets = {lang: load_items(lang) for lang in LANGUAGE_FILES}
    expected = len(MODELS) * len(datasets) * SESSIONS_PER_COMBO * 50
    completed = 0

    complete = prune_incomplete_sessions()
    if complete:
        logger.info("resuming: %d session(s) already complete", len(complete))

    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for model in MODELS:
            for language, items in datasets.items():
                for session_id in range(1, SESSIONS_PER_COMBO + 1):
                    if (model, language, session_id) in complete:
                        completed += 50
                        continue

                    logger.info(
                        "model=%s lang=%s session=%d/%d",
                        model,
                        language,
                        session_id,
                        SESSIONS_PER_COMBO,
                    )
                    for item in items:
                        raw, score = evaluate_item(model, language, item)
                        writer.writerow({
                            "session_id": session_id,
                            "model_name": model,
                            "language": language,
                            "item_id": item.id,
                            "trait": item.trait,
                            "keying": item.keying,
                            "raw_output": raw,
                            "parsed_score": score,
                        })
                        fh.flush()
                        completed += 1

    logger.info("Done: %d/%d runs written to %s", completed, expected, OUTPUT_CSV)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_pipeline()
