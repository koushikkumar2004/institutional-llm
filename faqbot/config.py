"""Loads config.yaml into a typed Settings object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class LLMSettings:
    base_url: str
    model: str
    api_key: str = "none"
    temperature: float = 0.0
    max_tokens: int = 200
    timeout_seconds: int = 120


@dataclass
class Settings:
    tenant_id: str
    tenant_name: str
    knowledge_base_path: Path
    roles: dict[str, list[str]]
    embedding_model: str
    reranker_model: str
    top_k: int
    similarity_threshold: float
    rerank_min_score: float | None
    llm: LLMSettings
    answer_mode: str
    storage_dir: Path
    seed_rules: list[dict[str, str]] = field(default_factory=list)

    @property
    def index_dir(self) -> Path:
        return self.storage_dir / "index" / self.tenant_id

    @property
    def database_path(self) -> Path:
        return self.storage_dir / f"{self.tenant_id}.sqlite3"

    @property
    def faq_roles(self) -> set[str]:
        """Every FAQ role label that at least one user role may see."""
        return {r for allowed in self.roles.values() for r in allowed}


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    base = config_path.resolve().parent
    llm_raw = raw.get("llm", {})
    retrieval = raw.get("retrieval", {})

    settings = Settings(
        tenant_id=str(raw["tenant"]["id"]),
        tenant_name=str(raw["tenant"]["name"]),
        knowledge_base_path=_resolve(raw["knowledge_base"]["path"], base),
        roles={str(k): [str(r) for r in v] for k, v in raw["roles"].items()},
        embedding_model=raw["models"]["embedding"],
        reranker_model=raw["models"]["reranker"],
        top_k=int(retrieval.get("top_k", 10)),
        similarity_threshold=float(retrieval.get("similarity_threshold", 0.40)),
        rerank_min_score=(
            None if retrieval.get("rerank_min_score") is None else float(retrieval["rerank_min_score"])
        ),
        llm=LLMSettings(
            base_url=str(llm_raw.get("base_url", "http://localhost:11434/v1")).rstrip("/"),
            model=str(llm_raw.get("model", "qwen2.5:1.5b")),
            api_key=str(llm_raw.get("api_key", "none")),
            temperature=float(llm_raw.get("temperature", 0)),
            max_tokens=int(llm_raw.get("max_tokens", 200)),
            timeout_seconds=int(llm_raw.get("timeout_seconds", 120)),
        ),
        answer_mode=str(raw.get("answer_mode", "generate")),
        storage_dir=_resolve(raw.get("storage", {}).get("dir", "storage"), base),
        seed_rules=list(raw.get("seed_rules") or []),
    )
    if settings.answer_mode not in ("generate", "direct"):
        raise ValueError("answer_mode must be 'generate' or 'direct'")
    return settings


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
