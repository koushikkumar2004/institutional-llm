# University FAQ Assistant: project context

Final-year project at Ramaiah University of Applied Sciences (RUAS), Bengaluru, intended to become a product sold to other universities: each university plugs in its own knowledge base and rules; the same engine answers.

## What it is

A role-based RAG (retrieval-augmented generation) FAQ assistant with a human-approved feedback loop. No model is trained or fine-tuned: models are pretrained and frozen. The system improves through (1) the knowledge base and (2) editable rules that a person approves. Facts never go into model weights.

Answer flow (`faqbot/pipeline.py`):
1. Role check: unknown roles refused. Roles and visibility in `config.yaml` (`student` sees student FAQs; `admin` sees student + admin).
2. FAISS search (exact inner product, `IndexFlatIP`) over the role's own saved index; embeddings of question + answer text; top_k 10.
3. Similarity threshold 0.40: nothing above it -> "I couldn't find that information in the available FAQs." and the LLM is never called.
4. CrossEncoder reranker re-scores the remaining candidates; best one wins.
5. Active rules for that FAQ's category (scope `all` or the category) are loaded from the database.
6. LLM writes the answer from that single FAQ + rules (or `direct` mode returns the stored answer). If the LLM is down, falls back to the stored answer.
7. Everything is logged to SQLite.

## Current configuration (on the developer's laptop)

- Windows, HP Pavilion i5-1335U, 16 GB RAM, no NVIDIA GPU (CPU only).
- Python 3.12.10 in `.venv` (installed via the Python install manager; 3.15 also present, not used).
- LLM: **Qwen2.5 3B** via **Ollama** (`llm.model: qwen2.5:3b`, `base_url: http://localhost:11434/v1`). About 10-15 s per answer on CPU. `OLLAMA_KEEP_ALIVE=1h` set so the model stays loaded.
- Embedder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims).
- Reranker: **`BAAI/bge-reranker-v2-m3`** (multilingual; replaced the English-only `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- `HF_HUB_OFFLINE=1` is set in the .bat launchers; models are cached in `C:\Users\donth\.cache\huggingface\hub\`. Unset it (`$env:HF_HUB_OFFLINE=$null`) before downloading a new model.
- `.streamlit/config.toml` has `[server] fileWatcherType = "none"` (stops harmless torchvision traceback spam).

Local edits made after the original zip (keep them):
- `faqbot/prompts.py`: rules are also repeated at the end of the user message ("Follow every one of these rules in your answer:"), because small models ignored rules placed only in the system prompt.
- `app.py`: a no-match answer's caption says "LLM not called" instead of the mode label.
- Starter rule #3 ("Begin the answer with Yes or No only when…") is **inactive**: with the rules repeated next to the question, 3B started every answer with "Yes,". Starter rules #1, #2, #4, #5 are active.

## How to run

- First time: `setup_windows.bat` (creates `.venv`, installs `requirements.txt`, downloads models, builds index, checks Ollama).
- Student portal + API: `run_portal.bat` -> http://localhost:8000 (admin PIN `ruas2026`, in `server.py` as `DEFAULT_ADMIN_PIN` or env `FAQ_ADMIN_PIN`).
- Admin/testing console (Streamlit): `run_app.bat` -> http://localhost:8501.
- Evaluation: `python scripts\evaluate.py` (set `$env:HF_HUB_OFFLINE=1` first). Results saved to `storage\eval\`.
- Tests: `python -m unittest discover -s tests` (34 tests, fake models, no downloads, no Ollama needed).

## Layout and where data lives

- `faqbot/`: the intelligence, no UI code. `config.py`, `knowledge_base.py` (CSV validation), `models.py`, `index.py` (saved per-role FAISS indexes, rebuilt automatically when CSV/model/roles change via a fingerprint), `llm.py` (OpenAI-style client, `chat` and streaming `chat_stream`), `prompts.py`, `database.py`, `pipeline.py` (`Assistant.answer`, `answer_stream`, `draft_rule`).
- `server.py`: standard-library HTTP server. Student portal + JSON API. Sessions in memory; role comes from the session, never the request. Endpoints: `/api/health`, `/api/login`, `/api/me`, `/api/logout`, `/api/ask`, `/api/ask/stream` (NDJSON, word by word), `/api/feedback`, admin-only `/api/admin/summary`, `/api/admin/rules`, `/api/admin/rules/status`, `/api/admin/feedback/draft`, `/api/admin/feedback/dismiss`. Details in `PORTAL.md`.
- `web/index.html`: portal page (sign-in, streaming chat with "Official FAQ #N" source seal, Yes/No feedback; admins get a Dashboard tab with stats, feedback -> rule flow, rule on/off switches).
- `app.py`: Streamlit console (Chat with "Behind the scenes" scores, Feedback review, Rules with history, Knowledge base, Activity + CSV exports).
- `data/ruas_faqs.csv`: knowledge base, 43 FAQs (columns id, category, role, question, answer). `data/original/` = untouched Colab CSV. Fixes in `data/CHANGES.md` (FAQ 43 role admin -> student; FAQ 28 category -> Admission_Process).
- `data/eval_questions.csv`: 38 fixed test questions with expected FAQ ids.
- `storage/ruas.sqlite3`: tables `conversations`, `feedback`, `rules`, `rule_history`; every table has `tenant_id`.
- `storage/index/ruas/`: embeddings + per-role FAISS indexes + manifest. `storage/eval/`: evaluation CSVs.
- `config.yaml`: tenant, KB path, roles, models, retrieval settings, LLM server, answer mode, seed rules (only loaded into an empty DB).

## Evaluation (38 questions, retrieval only)

| | Old reranker (ms-marco, English) | bge-reranker-v2-m3 |
|---|---|---|
| Overall pass | 32/38 (84%) | **34/38 (89%)** |
| Paraphrases | 19/20 | 20/20 |
| Kannada | 1/4 | 2/4 |
| Role leaks | 0 | 0 |

Remaining failures: 2 Kannada questions fall below the 0.40 threshold at the embedding stage (fix: BGE-M3 embedder, then re-tune the threshold); "Does RUAS offer scholarships?" matches the skill-development FAQ with high scores (needs an answerability check); "placement percentage" scrapes just over 0.40.

## Decisions and why

- Colab abandoned: runtimes reset, uploads and installs vanished, run-order bugs. Now a VS Code project.
- The LLM is a separate OpenAI-compatible server so moving to the HPC (vLLM, ~14B model) is a `config.yaml` change (`llm.base_url`, `llm.model`).
- Feedback never changes behaviour on its own: remark -> drafted rule -> human approves -> rule sent with every answer. Wrong facts are fixed in the knowledge base instead.
- Qwen chosen for open weights (Apache 2.0, commercial use OK); exact 14B model to be chosen by comparing candidates on the eval set once HPC GPUs are known.
- Small-model quirks seen: 1.5B ignored rules and invented details ("admissions committee", "COMEDK = Common Admission Test"); 3B follows rules but was primed by rule #3. Rules can conflict: on yes/no questions "begin with As per…" lost to the FAQ's own "No,".
- Demo rule used to show the loop: `Begin every answer with "As per the official RUAS FAQs,".` (asked with "What is the COMEDK code?"; delete or switch off after demos).

## Status and next steps

- Done: pipeline, saved indexes, role filtering, rules + feedback DB, Streamlit console, evaluation, student portal + API with streaming and admin dashboard.
- HOD wants an MVP (formal review on 28 Oct 2026); HPC access depends on that review.
- Next: FAQ editor in the portal dashboard; BGE-M3 embedder + threshold re-tune; answerability check; more FAQs/documents; real login (university SSO) and protect the Streamlit console; second university config to prove multi-tenancy; move to HPC (vLLM, 14B), Kannada answers with the larger model; later, optional LoRA fine-tuning on collected feedback (behaviour only).

## Working with the developer

Explain step by step and wait for confirmation before moving on. Use plain language; the developer is learning as they go. Give exact Windows/PowerShell commands (VS Code terminal needs `.\` before .bat files). Don't overwrite files they've edited (`config.yaml`, `faqbot/prompts.py`, `app.py`, `run_app.bat`) without saying so.
