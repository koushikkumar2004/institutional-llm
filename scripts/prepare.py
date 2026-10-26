"""One-time setup and health check. Safe to run again any time.

    python scripts/prepare.py

1. Checks the Python version
2. Loads and validates the knowledge base
3. Downloads the embedding model and reranker (first run only, ~600 MB)
4. Builds and saves the search index
5. Creates the database and loads the starter rules
6. Checks that Ollama is running and the model is pulled
7. Runs one test question end to end
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:  # Windows terminals otherwise crash on Kannada text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def step(title: str) -> None:
    print(f"\n=== {title}")


def main() -> int:
    problems = 0

    step("Python")
    v = sys.version_info
    print(f"{OK} Python {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) >= (3, 14) or (v.major, v.minor) < (3, 10):
        print(f"{WARN} Python 3.11 or 3.12 is recommended; some AI libraries lag behind the newest Python.")

    step("Configuration and knowledge base")
    from faqbot.config import load_settings
    from faqbot.knowledge_base import KnowledgeBaseError, load_knowledge_base

    settings = load_settings()
    print(f"{OK} Tenant: {settings.tenant_name} ({settings.tenant_id})")
    try:
        kb = load_knowledge_base(settings.knowledge_base_path, settings.faq_roles)
    except KnowledgeBaseError as exc:
        print(f"{FAIL} {exc}")
        return 1
    print(f"{OK} {len(kb)} FAQs from {settings.knowledge_base_path.name}")
    print(f"      by role: {kb['role'].value_counts().to_dict()}")
    print(f"      categories: {', '.join(sorted(kb['category'].unique()))}")

    step("Retrieval models (downloads on first run)")
    from faqbot.models import Embedder, Reranker

    try:
        t = time.perf_counter()
        embedder = Embedder(settings.embedding_model)
        print(f"{OK} Embedding model: {settings.embedding_model} ({time.perf_counter() - t:.0f}s)")
        t = time.perf_counter()
        reranker = Reranker(settings.reranker_model)
        print(f"{OK} Reranker: {settings.reranker_model} ({time.perf_counter() - t:.0f}s)")
    except ImportError as exc:
        print(f"{FAIL} {exc}")
        print("      Run: pip install -r requirements.txt  (with the .venv activated)")
        return 1
    except Exception as exc:  # noqa: BLE001 - usually a download problem
        print(f"{FAIL} Could not load a model: {type(exc).__name__}: {exc}")
        print("      Check your internet connection (college Wi-Fi may block huggingface.co; try a hotspot).")
        return 1

    step("Search index, database and rules")
    from faqbot.index import HAS_FAISS
    from faqbot.pipeline import Assistant

    if not HAS_FAISS:
        print(f"{WARN} FAISS not installed; using the identical NumPy exact search instead.")
    assistant = Assistant(settings, embedder=embedder, reranker=reranker)
    assistant.load_knowledge_base(force_rebuild=True)
    m = assistant.index.manifest
    print(f"{OK} Index built with {m['backend']} ({m['dimension']} dims) -> {settings.index_dir}")
    for role, rows in m["role_rows"].items():
        print(f"      role '{role}' can search {len(rows)} FAQs")
    rules = assistant.db.list_rules(("active",))
    print(f"{OK} Database: {settings.database_path} ({len(rules)} active rules)")

    step("Retrieval check")
    r = assistant.retrieve("How much money do I need to pay to apply?", "student")
    if r.best and r.best.faq_id == 3:
        print(f"{OK} Paraphrase found FAQ #3 (similarity {r.best.similarity:.3f}, reranker {r.best.rerank_score:.2f})")
    else:
        print(f"{WARN} Expected FAQ #3, got {r.best.faq_id if r.best else 'no match'}")
    r = assistant.retrieve("What is the intake capacity for programs?", "student")
    leaked = 28 in [c.faq_id for c in r.candidates]
    print(f"{FAIL if leaked else OK} Admin-only FAQ #28 {'LEAKED to' if leaked else 'hidden from'} students")
    problems += leaked

    step("LLM server")
    ready, message = assistant.llm.status()
    if not ready:
        print(f"{WARN} {message}")
        print("      The app still works in 'Stored FAQ answer' mode. To enable the LLM:")
        print("        1. Install Ollama from https://ollama.com/download and open it")
        print(f"        2. Run:  ollama pull {settings.llm.model}")
        print("        3. Run this script again")
    else:
        print(f"{OK} {message} at {settings.llm.base_url}")
        t = time.perf_counter()
        res = assistant.answer("What is the application fee?", "student", mode="generate", log=False)
        print(f"{OK} Test answer in {time.perf_counter() - t:.1f}s: {res.answer}")

    print("\nAll set." if problems == 0 else f"\n{problems} problem(s) above need fixing.")
    print("Start the app with:  streamlit run app.py")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
