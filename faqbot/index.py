"""Role-aware vector index, built once and saved to disk.

In the notebook, embeddings and a temporary FAISS index were rebuilt on every
question. Here they are built once, saved, and only rebuilt when the knowledge
base or embedding model changes (checked with a fingerprint).

One FAISS IndexFlatIP (exact inner-product search) is kept per user role and
contains only the FAQs that role may see, so restricted FAQs can never enter
the candidate list. If FAISS is not installed, an identical exact search is
done with NumPy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .knowledge_base import fingerprint, index_text
from .models import EmbedderProtocol

try:  # FAISS is the intended backend; NumPy gives identical results for small sets.
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover - depends on the machine
    faiss = None
    HAS_FAISS = False


class StaleIndexError(RuntimeError):
    """Saved index does not match the current knowledge base or model."""


@dataclass
class SearchHit:
    row: int  # position in the knowledge base DataFrame
    similarity: float


class FAQIndex:
    def __init__(self, embeddings: np.ndarray, role_rows: dict[str, list[int]], manifest: dict):
        self.embeddings = embeddings.astype("float32")
        self.role_rows = role_rows
        self.manifest = manifest
        self._faiss: dict[str, object] = {}
        if HAS_FAISS:
            for role, rows in role_rows.items():
                idx = faiss.IndexFlatIP(self.embeddings.shape[1])
                if rows:
                    idx.add(self.embeddings[rows])
                self._faiss[role] = idx

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, kb: pd.DataFrame, embedder: EmbedderProtocol, roles: dict[str, list[str]]) -> "FAQIndex":
        texts = [index_text(q, a) for q, a in zip(kb["question"], kb["answer"])]
        embeddings = embedder.encode(texts)
        role_rows = {
            user_role: [int(i) for i in kb.index[kb["role"].isin(allowed)]]
            for user_role, allowed in roles.items()
        }
        manifest = {
            "kb_fingerprint": fingerprint(kb),
            "embedding_model": embedder.name,
            "dimension": int(embeddings.shape[1]),
            "faq_ids": [int(x) for x in kb["id"]],
            "roles": roles,
            "role_rows": role_rows,
            "backend": "faiss.IndexFlatIP" if HAS_FAISS else "numpy (exact inner product)",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(embeddings, role_rows, manifest)

    # ------------------------------------------------------------ save / load
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        if HAS_FAISS:
            for role, idx in self._faiss.items():
                faiss.write_index(idx, str(directory / f"role_{role}.faiss"))
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, kb: pd.DataFrame, embedding_model: str, roles: dict[str, list[str]]) -> "FAQIndex":
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise StaleIndexError("No saved index yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kb_fingerprint") != fingerprint(kb):
            raise StaleIndexError("Knowledge base changed since the index was built")
        if manifest.get("embedding_model") != embedding_model:
            raise StaleIndexError("Embedding model changed since the index was built")
        if manifest.get("roles") != roles:
            raise StaleIndexError("Role settings changed since the index was built")
        embeddings = np.load(directory / "embeddings.npy")
        role_rows = {k: [int(x) for x in v] for k, v in manifest["role_rows"].items()}
        return cls(embeddings, role_rows, manifest)

    # ----------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, user_role: str, k: int) -> list[SearchHit]:
        rows = self.role_rows.get(user_role)
        if rows is None:
            raise ValueError(f"Unknown role '{user_role}'")
        if not rows:
            return []
        k = min(k, len(rows))
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if HAS_FAISS:
            scores, local = self._faiss[user_role].search(q, k)
            pairs = zip(local[0], scores[0])
        else:
            sims = self.embeddings[rows] @ q[0]
            order = np.argsort(-sims, kind="stable")[:k]
            pairs = ((i, sims[i]) for i in order)
        return [SearchHit(row=rows[int(i)], similarity=float(s)) for i, s in pairs if int(i) >= 0]
