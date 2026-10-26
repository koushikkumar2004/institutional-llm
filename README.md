# University FAQ Assistant

A role-based FAQ chatbot for **Ramaiah University of Applied Sciences**, built so that the same code can serve other universities by swapping the knowledge base and config.

```
question + role
  -> role check            student / admin; unknown roles refused
  -> FAISS search          multilingual MiniLM embeddings, one saved index per role
  -> similarity threshold  nothing close enough -> "not in the FAQs" (LLM never called)
  -> reranker             bge-reranker-v2-m3 (multilingual) re-scores question + answer, best FAQ wins
  -> active rules          editable rules from the database
  -> LLM answer            Qwen2.5 3B via Ollama, word by word (or the stored answer, word for word)
  -> logged                so testers can rate it and leave remarks
```

Feedback loop: **rate an answer → remark → draft a rule → a person approves it → the rule is sent to the LLM with every future answer.** Nothing changes from a remark alone.

---

## 1. One-time setup (Windows)

You need about 3 GB of free space and internet for the first run.

1. **Python 3.12** – install from https://www.python.org/downloads/ (scroll to 3.12.x).
   Tick **"Add python.exe to PATH"**. Even if you already have a newer Python, install 3.12 as well: PyTorch and FAISS support it reliably.
2. **VS Code** – https://code.visualstudio.com, plus the *Python* extension.
3. **Ollama** (runs the LLM on your laptop) – install from https://ollama.com/download, open it once (it then runs in the background), then in a terminal:
   ```
   ollama pull qwen2.5:3b
   ```
   That downloads about 2 GB. Optional but recommended, so the model stays loaded between questions:
   ```
   setx OLLAMA_KEEP_ALIVE 1h
   ```
   then quit Ollama from the tray icon and open it again.
4. Unzip this project, open the folder in VS Code (*File → Open Folder*).
5. Double-click **`setup_windows.bat`** (or run it in the VS Code terminal). It:
   - creates a private Python environment in `.venv`
   - installs the packages in `requirements.txt`
   - downloads the embedding model and reranker (first time only)
   - builds the search index and the database
   - checks Ollama and answers one test question

   Read the `[ OK ] / [WARN] / [FAIL]` lines at the end.
6. In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the one inside `.venv`.

## 2. Every day

Double-click **`run_app.bat`**, or in the VS Code terminal:

```
.venv\Scripts\activate
streamlit run app.py
```

Your browser opens http://localhost:8501. If the terminal asks for an email the first time, just press Enter.

Other commands (with `.venv` activated):

| Command | What it does |
|---|---|
| `python scripts/prepare.py` | Health check; rebuilds the index |
| `python scripts/evaluate.py` | Scores retrieval on 38 fixed test questions, saves a CSV to `storage/eval/` |
| `python scripts/evaluate.py --with-llm` | Same, plus generated answers to read through |
| `python -m unittest discover -s tests -v` | 34 offline tests of the pipeline, database, rules and portal API (no models needed) |

## 2b. Student portal (the product side)

Double-click **`run_portal.bat`** (or `.\run_portal.bat` in the VS Code terminal). The browser opens http://localhost:8000.

- **Students** sign in with a name (optional), ask questions, see each answer appear word by word with the official FAQ it came from, and mark answers helpful or not with a note.
- **Admins** sign in with the PIN (`ruas2026`, set in `server.py`) and get a **Dashboard**: numbers, student feedback they can turn into a rule or dismiss, and every rule with an on/off switch.

The portal and the console share the same intelligence (`faqbot/`) and the same database, so feedback from the portal shows up in the console and approved rules apply everywhere. API details are in `PORTAL.md`. Run both at once only when needed; each loads the models.

## 3. The app (admin console)

| Page | Use it to |
|---|---|
| 💬 **Chat** | Ask as *student* or *admin*. Open **Behind the scenes** to see the FAISS shortlist, similarity scores, reranker scores, the chosen FAQ and the rules sent to the LLM. Rate the answer and leave a remark. |
| 📝 **Feedback review** | For each remark: **Draft a rule** (the LLM suggests one), edit it, then save as pending or approve. Or mark it as a knowledge-base fix, or dismiss it. |
| 📏 **Rules** | Approve or reject pending rules; edit, switch off or delete active ones; add your own. Each rule shows why it exists and its full change history. Rules can apply to *all* answers or to one FAQ category. |
| 📚 **Knowledge base** | Browse and search the FAQs. After editing the CSV, click **Reload knowledge base and rebuild index**. |
| 📊 **Activity** | Totals, the full question log, and CSV downloads of conversations, feedback and rules. |

The sidebar switches between **Generate with LLM** and **Stored FAQ answer** (the FAQ's own words, no LLM), which makes a good side-by-side comparison.

Five starter rules are loaded the first time. Each one comes from a failure actually seen in Colab: the invented admissions committee, "COMEDK = Common Admission Test", and the "No, …" reply to a "how" question.

## 4. Suggested demo (about 10 minutes)

1. Show the folder in VS Code: no Colab, nothing to re-upload.
2. **Paraphrase:** *"How much money do I need to pay to apply?"* → Behind the scenes → FAQ #3 chosen.
3. **Roles:** *"What is the intake capacity?"* as student, then as admin. Only the admin gets FAQ #28.
4. **Out of scope:** *"Who won the FIFA World Cup?"* → no match, and the LLM is never called.
5. **Hallucination check:** *"Can I change my program after admission?"* Compare with the stored answer. Rate it and write a remark.
6. **Learning loop:** Feedback review → Draft a rule → edit → Approve. Ask again; the new rule now appears under *Rules sent to the LLM*.
7. **Activity** page, then in the terminal run `python scripts/evaluate.py` and talk through the numbers, including the failures (Kannada, scholarships).
8. **What's next:** better multilingual models and a 14B LLM on the HPC are changes in `config.yaml`, not a rewrite.

## 5. What is where

```
university-faq-assistant/
├── app.py                  Admin/testing console (Streamlit)
├── server.py               Student portal + JSON API (standard library only)
├── web/index.html          Student portal page (chat + admin dashboard)
├── run_portal.bat          Start the student portal
├── PORTAL.md               Portal and API reference
├── CLAUDE.md               Project context for Claude Code
├── config.yaml             Everything you might change: models, threshold, LLM server, roles, starter rules
├── requirements.txt
├── setup_windows.bat       One-time setup
├── run_app.bat             Start the UI
├── faqbot/                 The "intelligence" - no UI code in here
│   ├── config.py           Reads config.yaml
│   ├── knowledge_base.py   Loads and validates the FAQ CSV
│   ├── models.py           MiniLM embedder + CrossEncoder reranker
│   ├── index.py            Saved FAISS indexes, one per role
│   ├── llm.py              Talks to Ollama / vLLM / any OpenAI-style server
│   ├── prompts.py          Answer prompt (rules inserted) + rule-drafting prompt
│   ├── database.py         SQLite: conversations, feedback, rules, rule history
│   └── pipeline.py         The full flow: retrieve -> rerank -> rules -> answer -> log
├── scripts/
│   ├── prepare.py          Setup and health check
│   └── evaluate.py         Fixed test set -> scores -> CSV
├── data/
│   ├── ruas_faqs.csv       Knowledge base used by the app
│   ├── eval_questions.csv  38 test questions with expected FAQ ids
│   ├── CHANGES.md          What was fixed in the dataset and why
│   └── original/           The Colab CSV, unchanged
├── tests/                  Offline tests with fake models (pipeline + portal API)
└── storage/                Created automatically: index files, ruas.sqlite3, eval results
```

`faqbot/` has no UI code on purpose. The real product frontend will call the same `Assistant` through an API, so this testing UI can be thrown away later without touching the intelligence.

## 6. Changes from the Colab notebook

| Notebook | Here |
|---|---|
| CSV re-uploaded every session | Lives in `data/`, checked when loaded (bad rows are listed) |
| Embeddings and FAISS index rebuilt on every question | Built once, saved, rebuilt only when the CSV or model changes |
| `rag_answer` defined ~10 times; last one run wins | One pipeline in `faqbot/pipeline.py` |
| Unknown role silently treated as student | Unknown roles refused |
| Qwen loaded with bitsandbytes (needs an NVIDIA GPU) | Qwen served by Ollama on CPU; any OpenAI-style server works |
| Keyword "payment safeguard" | Replaced by editable rules |
| Results only printed | Every question, answer, score and rule logged to SQLite |
| Ad-hoc test cells | `scripts/evaluate.py` on a fixed 38-question set |

Kept the same: the embedding model, the reranker, question+answer embedding text, top-10 shortlist, 0.40 threshold, reranking on question+answer, and generating from the single best FAQ.

## 7. Moving to other models or the HPC

Edit `config.yaml`:

- **Different LLM on the laptop:** `ollama pull <model>`, then set `llm.model` (3B is the current choice; 1.5B is faster but ignores rules and invents details).
- **LLM on the HPC:** run a vLLM server there and set `llm.base_url: http://<hpc-host>:8000/v1` and `llm.model` to the served model.
- **Better multilingual embedder (next step):** set `models.embedding: BAAI/bge-m3`, run `prepare.py`, re-tune `similarity_threshold` (its scores use a different scale), then `evaluate.py` to compare against the saved results.
- **Downloading a new model:** the .bat files set `HF_HUB_OFFLINE=1`; in a plain terminal run `$env:HF_HUB_OFFLINE=$null` before `prepare.py` or `evaluate.py` so the model can download.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `py` or `python` is not recognized | Install Python 3.12 and tick "Add python.exe to PATH", then open a new terminal. |
| `pip install` fails on torch or faiss | You are probably on a too-new Python. Install 3.12, delete the `.venv` folder, run `setup_windows.bat` again. |
| Model download fails or hangs | College Wi-Fi may block Hugging Face; try a phone hotspot. Models are cached after the first download. |
| Sidebar says the LLM isn't ready | Open the Ollama app. In a terminal: `ollama list` should show `qwen2.5:3b`; if not, `ollama pull qwen2.5:3b`. The app still works in *Stored FAQ answer* mode. |
| First answer is slow | Ollama loads the model into RAM on first use; later answers are faster. CPU-only, so expect a few seconds per answer. |
| Warning about "symlinks" from Hugging Face | Harmless on Windows. |
| "Knowledge base has problems" | The message lists the exact rows. Fix the CSV and reload. |
| Start completely fresh | Delete the `storage` folder. **This also deletes all conversations, feedback and rules.** |
