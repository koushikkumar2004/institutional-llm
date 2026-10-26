"""Loads and validates a university's FAQ knowledge base (CSV)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["id", "category", "role", "question", "answer"]


class KnowledgeBaseError(ValueError):
    """The knowledge base file has problems that must be fixed before use."""


def load_knowledge_base(path: str | Path, allowed_roles: set[str]) -> pd.DataFrame:
    """Read the CSV, clean it and check it. Raises KnowledgeBaseError listing every problem found."""
    path = Path(path)
    if not path.exists():
        raise KnowledgeBaseError(f"Knowledge base not found: {path}")

    # utf-8-sig quietly removes the hidden byte-order mark Excel adds to CSVs.
    # If someone saved it from Excel as plain "CSV" (Windows encoding), fall back to that.
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KnowledgeBaseError(f"Missing column(s): {', '.join(missing)}. Found: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["role"] = df["role"].str.lower()

    problems: list[str] = []
    ids = pd.to_numeric(df["id"], errors="coerce")
    for i in df.index[ids.isna()]:
        problems.append(f"Row {i + 2}: id '{df.at[i, 'id']}' is not a number")
    for col in ("question", "answer", "category", "role"):
        for i in df.index[df[col] == ""]:
            problems.append(f"Row {i + 2}: empty {col}")
    for i in df.index[~df["role"].isin(allowed_roles) & (df["role"] != "")]:
        problems.append(
            f"Row {i + 2}: role '{df.at[i, 'role']}' is not one of {sorted(allowed_roles)}"
        )
    dupes = ids[ids.duplicated(keep=False) & ids.notna()]
    if not dupes.empty:
        problems.append(f"Duplicate ids: {sorted(set(int(x) for x in dupes))}")

    if problems:
        raise KnowledgeBaseError("Knowledge base has problems:\n  - " + "\n  - ".join(problems))

    df["id"] = ids.astype(int)
    return df.sort_values("id").reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """A short hash of the knowledge base content. Changes whenever any FAQ changes."""
    payload = json.dumps(df[REQUIRED_COLUMNS].to_dict(orient="records"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def index_text(question: str, answer: str) -> str:
    """The text that gets embedded for each FAQ (question + answer, as in the final notebook)."""
    return f"{question} {answer}"
