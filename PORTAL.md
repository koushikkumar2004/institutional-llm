# Student portal (MVP layer)

Adds a student-facing chat portal and a JSON API on top of the existing assistant. Nothing in the existing project is changed.

## Files

| File | What it is |
|---|---|
| `server.py` | The API + web server (Python standard library only, nothing to install) |
| `web/index.html` | The student portal page |
| `run_portal.bat` | Starts the portal at http://localhost:8000 |
| `tests/test_server.py` | 12 tests for the API |
| `faqbot/llm.py`, `faqbot/pipeline.py` | Updated to stream answers word by word |

## Run

1. Make sure Ollama is running.
2. Double-click `run_portal.bat` (or `.\run_portal.bat` in VS Code's terminal).
3. The browser opens http://localhost:8000.
   - **Student:** type a name (optional) and click Continue.
   - **Admin:** admin PIN is `ruas2026` (change `DEFAULT_ADMIN_PIN` in `server.py`). Admins also see admin-only FAQs and get a **Dashboard** tab: numbers, student feedback (turn it into a rule or dismiss it) and the rules list (switch rules on/off, approve pending ones, add new ones).

The full Streamlit console is still available for deeper work (scores, rule history, FAQ list). To show it at the same time, run `run_app.bat` in a second terminal (it opens http://localhost:8501). Both use the same database, so feedback sent from the portal appears on the console's Feedback review page.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | university name, FAQ count, LLM status |
| POST | `/api/login` | `{"role":"student","name":"Priya"}` or `{"role":"admin","pin":"..."}` | `token` |
| GET | `/api/me` | | role and name for the token |
| POST | `/api/logout` | | |
| POST | `/api/ask` | `{"question":"..."}` | answer, status, source FAQ, latency, conversation id |
| POST | `/api/ask/stream` | `{"question":"..."}` | the answer word by word, one JSON object per line, ending with a `done` line |
| POST | `/api/feedback` | `{"conversation_id":1,"verdict":"good"\|"bad","remark":"..."}` | feedback id |

Admin only: `GET /api/admin/summary`, `POST /api/admin/rules`, `POST /api/admin/rules/status`, `POST /api/admin/feedback/draft`, `POST /api/admin/feedback/dismiss` (details at the top of `server.py`).

Send the token as `Authorization: Bearer <token>`. The role always comes from the signed-in session, not from the request, so a student can't ask as an admin.

This is a demo login (in-memory sessions, one admin PIN). A real deployment would use the university's own login.
