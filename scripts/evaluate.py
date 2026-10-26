"""Measure the pipeline on a fixed set of test questions.

    python scripts/evaluate.py              # retrieval only (fast, no LLM needed)
    python scripts/evaluate.py --with-llm   # also generate answers to read through

Every run is saved to storage/eval/ with a timestamp, so you can compare
before/after any change (new model, threshold, rules...) on exactly the same
questions. These are the numbers to quote, together with how many questions
they are based on.

Metrics
  in_shortlist  - the right FAQ was among FAISS's top_k candidates
  faiss_top1    - FAISS alone ranked the right FAQ first
  final_top1    - after threshold + reranker, the chosen FAQ is the right one
  rejected      - for out-of-scope questions: correctly answered "not in the FAQs"
  leak          - a forbidden (other role's) FAQ appeared in the candidates
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def parse_ids(value) -> list[int]:
    value = str(value).strip()
    if value in ("", "NONE", "nan"):
        return []
    return [int(x) for x in value.split("|")]


def evaluate(assistant, questions: pd.DataFrame, with_llm: bool = False) -> pd.DataFrame:
    rows = []
    for _, q in questions.iterrows():
        expected = parse_ids(q["expected_ids"])
        forbidden = parse_ids(q.get("forbidden_ids", ""))
        r = assistant.retrieve(q["question"], q["role"])
        cand_ids = [c.faq_id for c in r.candidates]
        row = {
            "test_type": q["test_type"],
            "role": q["role"],
            "question": q["question"],
            "expected": q["expected_ids"],
            "faiss_top": cand_ids[0] if cand_ids else None,
            "faiss_top_sim": r.candidates[0].similarity if r.candidates else None,
            "chosen": r.best.faq_id if r.best else None,
            "chosen_sim": r.best.similarity if r.best else None,
            "chosen_rerank": r.best.rerank_score if r.best else None,
            "shortlist": " ".join(map(str, cand_ids)),
            "leak": bool(set(forbidden) & set(cand_ids)),
        }
        if expected:
            row["in_shortlist"] = bool(set(expected) & set(cand_ids))
            row["faiss_top1"] = bool(cand_ids) and cand_ids[0] in expected
            row["final_top1"] = r.best is not None and r.best.faq_id in expected
            row["rejected"] = None
            row["pass"] = row["final_top1"] and not row["leak"]
        elif forbidden:
            # Access test: passing means the other role's FAQ never showed up.
            row["in_shortlist"] = row["faiss_top1"] = row["final_top1"] = row["rejected"] = None
            row["pass"] = not row["leak"]
        else:
            row["in_shortlist"] = row["faiss_top1"] = row["final_top1"] = None
            row["rejected"] = r.best is None
            row["pass"] = row["rejected"]
        if with_llm:
            row["answer"] = assistant.answer(q["question"], q["role"], mode="generate", log=False).answer
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    def pct(series) -> str:
        s = series.dropna()
        return "–" if s.empty else f"{int(s.sum())}/{len(s)} ({100 * s.mean():.0f}%)"

    out = []
    for name, group in list(results.groupby("test_type", sort=False)) + [("ALL", results)]:
        out.append({
            "test_type": name,
            "questions": len(group),
            "in_shortlist": pct(group["in_shortlist"]),
            "faiss_top1": pct(group["faiss_top1"]),
            "final_top1": pct(group["final_top1"]),
            "rejected": pct(group["rejected"]),
            "leaks": int(group["leak"].sum()),
            "pass": pct(group["pass"]),
        })
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(ROOT / "data" / "eval_questions.csv"))
    parser.add_argument("--with-llm", action="store_true", help="also generate answers (needs Ollama)")
    args = parser.parse_args()

    from faqbot.config import load_settings
    from faqbot.pipeline import Assistant

    settings = load_settings()
    assistant = Assistant(settings)
    questions = pd.read_csv(args.questions, encoding="utf-8-sig", dtype=str, keep_default_na=False)

    print(f"Evaluating {len(questions)} questions | threshold {settings.similarity_threshold} | "
          f"top_k {settings.top_k}\n  embedder: {settings.embedding_model}\n  reranker: {settings.reranker_model}"
          + (f"\n  LLM: {settings.llm.model}" if args.with_llm else ""))
    results = evaluate(assistant, questions, with_llm=args.with_llm)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\nSummary")
    print(summarise(results).to_string(index=False))

    failures = results[~results["pass"].astype(bool)]
    if not failures.empty:
        print("\nFailures (worth showing honestly)")
        print(failures[["test_type", "question", "expected", "faiss_top", "chosen", "chosen_sim",
                        "chosen_rerank"]].to_string(index=False))

    out_dir = settings.storage_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.csv"
    results.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nFull results saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
