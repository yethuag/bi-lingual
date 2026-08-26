from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))


RESULTS_CSV = Path("llm_persona_results.csv")
PER_SESSION_CSV = Path("llm_persona_trait_scores.csv")
SUMMARY_CSV = Path("llm_persona_summary.csv")
CHART_PNG = Path("llm_persona_traits.png")

TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
MAX_SCORE = 5

SessionTraitKey: TypeAlias = tuple[str, str, int, str]
SummaryKey: TypeAlias = tuple[str, str, str]
SummaryValue: TypeAlias = tuple[float, float, int]


@dataclass
class Row:
    session_id: int
    model_name: str
    language: str
    trait: str
    keying: str
    score: Optional[int]


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("parsed_score") or "").strip()
            rows.append(Row(
                session_id=int(r["session_id"]),
                model_name=r["model_name"],
                language=r["language"],
                trait=r["trait"],
                keying=r["keying"],
                score=int(raw) if raw else None,
            ))
    return rows


def adjust(score: int, keying: str) -> int:
    """Reverse-key scores so higher always means more of the trait."""
    return (MAX_SCORE + 1) - score if keying == "-" else score


def session_trait_means(rows: list[Row]) -> dict[SessionTraitKey, float]:
    buckets: dict[SessionTraitKey, list[int]] = defaultdict(list)
    for row in rows:
        if row.score is None:
            continue
        key = (row.model_name, row.language, row.session_id, row.trait)
        buckets[key].append(adjust(row.score, row.keying))
    return {key: statistics.mean(vals) for key, vals in buckets.items() if vals}


def parse_failures(rows: list[Row]) -> dict[tuple[str, str], tuple[int, int]]:
    total: dict[tuple[str, str], int] = defaultdict(int)
    fails: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (row.model_name, row.language)
        total[key] += 1
        if row.score is None:
            fails[key] += 1
    return {key: (fails[key], total[key]) for key in total}


def write_per_session(means: dict[SessionTraitKey, float]) -> None:
    with PER_SESSION_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model_name", "language", "session_id", "trait", "trait_mean"])
        for (model, lang, sess, trait), val in sorted(means.items()):
            w.writerow([model, lang, sess, trait, round(val, 4)])


def write_summary(means: dict[SessionTraitKey, float]) -> dict[SummaryKey, SummaryValue]:
    grouped: dict[SummaryKey, list[float]] = defaultdict(list)
    for (model, lang, _sess, trait), val in means.items():
        grouped[(model, lang, trait)].append(val)

    summary: dict[SummaryKey, SummaryValue] = {}
    for key, vals in grouped.items():
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[key] = (mean, sd, len(vals))

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model_name", "language", "trait", "mean", "stdev", "n_sessions"])
        for (model, lang, trait), (mean, sd, n) in sorted(summary.items()):
            w.writerow([model, lang, trait, round(mean, 4), round(sd, 4), n])
    return summary


def print_report(
    summary: dict[SummaryKey, SummaryValue],
    failures: dict[tuple[str, str], tuple[int, int]],
) -> None:
    models = sorted({m for (m, _l, _t) in summary})
    languages = sorted({l for (_m, l, _t) in summary})

    print("\nBig Five trait means (1-5, reverse-keyed; higher = more of the trait)\n")
    for model in models:
        print(f"  {model}")
        header = "    {:<22}".format("trait") + "".join(f"{lang:>14}" for lang in languages)
        print(header)
        for trait in TRAITS:
            cells = ""
            for lang in languages:
                entry = summary.get((model, lang, trait))
                cells += f"{entry[0]:>8.2f}±{entry[1]:>4.2f}" if entry else f"{'-':>14}"
            print(f"    {trait:<22}{cells}")

        if len(languages) == 2:
            gaps = []
            for trait in TRAITS:
                a = summary.get((model, languages[0], trait))
                b = summary.get((model, languages[1], trait))
                if a and b:
                    gaps.append(abs(a[0] - b[0]))
            if gaps:
                print(f"    {'mean |Δ language|':<22}{statistics.mean(gaps):>14.2f}")
        print()

    print("Parse failures (model x language)")
    for (model, lang), (fails, total) in sorted(failures.items()):
        pct = 100 * fails / total if total else 0
        flag = "  <-- check" if pct >= 1 else ""
        print(f"    {model:<42} {lang:<10} {fails:>5}/{total:<6} ({pct:4.1f}%){flag}")


def try_chart(summary: dict[SummaryKey, SummaryValue]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(matplotlib not installed; skipping {CHART_PNG}. `pip install matplotlib` to enable.)")
        return

    models = sorted({m for (m, _l, _t) in summary})
    languages = sorted({l for (_m, l, _t) in summary})
    x = range(len(TRAITS))

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.2 * len(models)), squeeze=False)
    width = 0.8 / max(len(languages), 1)
    for ax, model in zip(axes[:, 0], models):
        for i, lang in enumerate(languages):
            means = [summary.get((model, lang, t), (0, 0, 0))[0] for t in TRAITS]
            errs = [summary.get((model, lang, t), (0, 0, 0))[1] for t in TRAITS]
            offsets = [xi + i * width for xi in x]
            ax.bar(offsets, means, width, yerr=errs, capsize=3, label=lang)
        ax.set_title(model, fontsize=10)
        ax.set_ylim(1, 5)
        ax.set_xticks([xi + width * (len(languages) - 1) / 2 for xi in x])
        ax.set_xticklabels([t[:5] for t in TRAITS])
        ax.legend(fontsize=8)
        ax.set_ylabel("mean score")
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=120)
    print(f"\nChart written to {CHART_PNG}")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found - run llm_pipeline.py first.")

    rows = load_rows(RESULTS_CSV)
    print(f"Loaded {len(rows)} item responses from {RESULTS_CSV}")

    means = session_trait_means(rows)
    failures = parse_failures(rows)

    write_per_session(means)
    summary = write_summary(means)

    print_report(summary, failures)
    try_chart(summary)

    print(f"\nWrote {PER_SESSION_CSV} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
