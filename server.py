"""Student portal + REST API for the University FAQ Assistant.

Run:  python server.py            (then open http://localhost:8000)
      python server.py --port 9000 --no-browser

This is the product-facing layer. It uses the same intelligence as the testing
console (faqbot.Assistant) and exposes it as a small JSON API that any frontend
(this portal, a React app, a mobile app) can call:

    GET  /api/health      tenant name, LLM status
    POST /api/login       {"role": "student", "name": "..."} or {"role": "admin", "pin": "..."}
    GET  /api/me          who is signed in
    POST /api/logout
    POST /api/ask         {"question": "..."}              (needs sign-in)
    POST /api/ask/stream  same, but the answer arrives word by word (newline-delimited JSON)
    POST /api/feedback    {"conversation_id": 1, "verdict": "good"|"bad", "remark": "..."}

  Admin only:
    GET  /api/admin/summary           numbers, latest feedback, all rules
    POST /api/admin/rules             {"text": "...", "scope": "all", "feedback_id": 3}  -> active rule
    POST /api/admin/rules/status      {"id": 4, "status": "active"|"inactive"|"rejected"}
    POST /api/admin/feedback/draft    {"id": 3}   LLM suggests a rule from the remark
    POST /api/admin/feedback/dismiss  {"id": 3}

The user's role comes from their session on the server, never from the request,
so a student cannot ask as an admin by changing a field. Only the Python
standard library is used here: no extra packages to install.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:  # Windows terminals otherwise crash on Kannada text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

WEB_DIR = ROOT / "web"
MAX_BODY = 64 * 1024
MAX_QUESTION = 500
SESSION_HOURS = 12
# Demo admin PIN. Change it here or set the FAQ_ADMIN_PIN environment variable.
DEFAULT_ADMIN_PIN = "ruas2026"


class Portal:
    """Holds the assistant, the signed-in sessions and a lock around the models."""

    def __init__(self, assistant, admin_pin: str):
        self.assistant = assistant
        self.admin_pin = admin_pin
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()  # one question at a time through the models
        self.sessions_lock = threading.Lock()

    def new_session(self, role: str, name: str) -> str:
        token = secrets.token_urlsafe(24)
        with self.sessions_lock:
            self.sessions[token] = {"role": role, "name": name, "expires": time.time() + SESSION_HOURS * 3600}
        return token

    def session(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self.sessions_lock:
            s = self.sessions.get(token)
            if s and s["expires"] < time.time():
                self.sessions.pop(token, None)
                return None
            return s

    def end_session(self, token: str | None) -> None:
        with self.sessions_lock:
            self.sessions.pop(token or "", None)


def _result_json(r) -> dict:
    source = None
    if r.best is not None and r.status == "answered":
        source = {"faq_id": r.best.faq_id, "category": r.best.category, "question": r.best.question}
    return {
        "conversation_id": r.conversation_id,
        "answer": r.answer,
        "status": r.status,
        "answer_mode": r.answer_mode,
        "source": source,
        "latency_ms": r.latency_ms,
    }


def make_handler(portal: Portal):
    settings = portal.assistant.settings

    class Handler(BaseHTTPRequestHandler):
        server_version = "FAQAssistant/0.2"

        # ------------------------------------------------------------ helpers
        def log_message(self, fmt, *args):  # short one-line logs
            sys.stdout.write(f"[portal] {self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}\n")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _body(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > MAX_BODY:
                self._error(413, "Request too large")
                return None
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(400, "Body must be JSON")
                return None
            if not isinstance(data, dict):
                self._error(400, "Body must be a JSON object")
                return None
            return data

        def _token(self) -> str | None:
            auth = self.headers.get("Authorization", "")
            return auth[7:].strip() if auth.lower().startswith("bearer ") else None

        def _require_session(self) -> dict | None:
            s = portal.session(self._token())
            if s is None:
                self._error(401, "Please sign in")
            return s

        # --------------------------------------------------------------- GET
        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                page = WEB_DIR / "index.html"
                if not page.exists():
                    return self._error(500, "web/index.html is missing")
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/health":
                ready, message = portal.assistant.llm.status()
                return self._json(200, {
                    "tenant_id": settings.tenant_id,
                    "tenant_name": settings.tenant_name,
                    "faqs": len(portal.assistant.kb),
                    "llm_ready": ready,
                    "llm_message": message,
                    "llm_model": settings.llm.model,
                    "answer_mode": settings.answer_mode,
                })
            if path == "/api/me":
                s = self._require_session()
                if s:
                    self._json(200, {"role": s["role"], "name": s["name"]})
                return
            if path == "/api/admin/summary":
                if self._require_admin():
                    self._admin_summary()
                return
            self._error(404, "Not found")

        # -------------------------------------------------------------- POST
        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            data = self._body()
            if data is None:
                return
            if path == "/api/login":
                return self._login(data)
            if path == "/api/logout":
                portal.end_session(self._token())
                return self._json(200, {"ok": True})
            if path == "/api/ask":
                return self._ask(data)
            if path == "/api/ask/stream":
                return self._ask_stream(data)
            if path == "/api/feedback":
                return self._feedback(data)
            if path.startswith("/api/admin/"):
                if not self._require_admin():
                    return
                handler = {
                    "/api/admin/rules": self._admin_add_rule,
                    "/api/admin/rules/status": self._admin_rule_status,
                    "/api/admin/feedback/draft": self._admin_draft_rule,
                    "/api/admin/feedback/dismiss": self._admin_dismiss,
                }.get(path)
                if handler:
                    return handler(data)
            self._error(404, "Not found")

        def _require_admin(self) -> dict | None:
            s = self._require_session()
            if s and s["role"] != "admin":
                self._error(403, "Admins only")
                return None
            return s

        def _question(self, data: dict) -> tuple[str, str | None] | None:
            question = " ".join(str(data.get("question", "")).split())
            if not question:
                self._error(400, "Type a question first")
                return None
            if len(question) > MAX_QUESTION:
                self._error(400, f"Please keep questions under {MAX_QUESTION} characters")
                return None
            mode = data.get("mode")
            if mode not in (None, "generate", "direct"):
                self._error(400, "mode must be 'generate' or 'direct'")
                return None
            return question, mode

        def _login(self, data: dict) -> None:
            role = str(data.get("role", "")).strip().lower()
            name = " ".join(str(data.get("name", "")).split())[:60]
            if role not in settings.roles:
                return self._error(400, f"Unknown role. Choose one of: {', '.join(settings.roles)}")
            if role == "admin":
                if not secrets.compare_digest(str(data.get("pin", "")), portal.admin_pin):
                    return self._error(403, "Wrong admin PIN")
                name = name or "Admin"
            name = name or "Student"
            token = portal.new_session(role, name)
            self._json(200, {"token": token, "role": role, "name": name})

        def _ask(self, data: dict) -> None:
            s = self._require_session()
            if not s:
                return
            parsed = self._question(data)
            if not parsed:
                return
            question, mode = parsed
            try:
                with portal.lock:
                    r = portal.assistant.answer(question, s["role"], mode=mode, tester=f"portal:{s['name']}")
            except Exception as exc:  # noqa: BLE001
                sys.stdout.write(f"[portal] error answering: {type(exc).__name__}: {exc}\n")
                return self._error(500, "Something went wrong while answering. Please try again.")
            self._json(200, _result_json(r))

        def _ask_stream(self, data: dict) -> None:
            """Answer as newline-delimited JSON: {"type":"token","text":...} lines, then {"type":"done",...}."""
            s = self._require_session()
            if not s:
                return
            parsed = self._question(data)
            if not parsed:
                return
            question, mode = parsed
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            def emit(obj: dict) -> None:
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                with portal.lock:
                    stream = portal.assistant.answer_stream(question, s["role"], mode=mode,
                                                            tester=f"portal:{s['name']}")
                    for kind, value in stream:
                        if kind == "token":
                            emit({"type": "token", "text": value})
                        elif kind == "done":
                            emit({"type": "done", **_result_json(value)})
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away
            except Exception as exc:  # noqa: BLE001
                sys.stdout.write(f"[portal] error answering: {type(exc).__name__}: {exc}\n")
                try:
                    emit({"type": "error", "message": "Something went wrong while answering. Please try again."})
                except OSError:
                    pass

        # ------------------------------------------------------------- admin
        def _admin_summary(self) -> None:
            a = portal.assistant
            stats = a.db.stats()
            feedback = [{
                "id": f["id"], "created_at": f["created_at"], "who": (f["tester"] or "").replace("portal:", ""),
                "verdict": f["verdict"], "rating": f["rating"], "remark": f["remark"] or "", "status": f["status"],
                "question": f["question"], "answer": f["answer"], "faq_id": f["faq_id"],
                "faq_category": f["faq_category"], "user_role": f["user_role"],
                "resolution_note": f["resolution_note"] or "",
            } for f in a.db.list_feedback()[:50]]
            rules = [{
                "id": r["id"], "text": r["text"], "reason": r["reason"] or "", "scope": r["scope"],
                "status": r["status"], "source": r["source"], "feedback_id": r["feedback_id"],
                "updated_at": r["updated_at"],
            } for r in a.db.list_rules()]
            self._json(200, {"stats": stats, "feedback": feedback, "rules": rules,
                             "categories": ["all"] + a.categories()})

        def _admin_add_rule(self, data: dict) -> None:
            a = portal.assistant
            text = " ".join(str(data.get("text", "")).split())
            scope = str(data.get("scope") or "all")
            if not text:
                return self._error(400, "Write the rule first")
            if len(text) > 500:
                return self._error(400, "Keep rules under 500 characters")
            if scope not in ["all"] + a.categories():
                return self._error(400, "Unknown category")
            feedback_id = data.get("feedback_id")
            if feedback_id is not None:
                try:
                    feedback_id = int(feedback_id)
                except (TypeError, ValueError):
                    return self._error(400, "feedback_id must be a number")
                if a.db.get_feedback(feedback_id) is None:
                    return self._error(404, "That feedback was not found")
            who = self._require_session()["name"]
            reason = str(data.get("reason") or "").strip()[:300] or None
            rid = a.db.add_rule(text, status="active", source="feedback" if feedback_id else "manual",
                                reason=reason, scope=scope, feedback_id=feedback_id, created_by=f"portal:{who}")
            if feedback_id:
                a.db.update_feedback(feedback_id, "rule_created", note=f"Rule #{rid} (active)")
            self._json(200, {"ok": True, "rule_id": rid})

        def _admin_rule_status(self, data: dict) -> None:
            a = portal.assistant
            try:
                rule_id = int(data.get("id"))
            except (TypeError, ValueError):
                return self._error(400, "id is required")
            status = data.get("status")
            if status not in ("active", "inactive", "rejected"):
                return self._error(400, "status must be active, inactive or rejected")
            rule = a.db.get_rule(rule_id)
            if rule is None:
                return self._error(404, "Rule not found")
            a.db.update_rule(rule_id, changed_by=f"portal:{self._require_session()['name']}", status=status)
            if rule["status"] == "pending" and rule["feedback_id"] and status == "active":
                a.db.update_feedback(rule["feedback_id"], "rule_created", note=f"Rule #{rule_id} approved")
            self._json(200, {"ok": True})

        def _admin_draft_rule(self, data: dict) -> None:
            try:
                feedback_id = int(data.get("id"))
            except (TypeError, ValueError):
                return self._error(400, "id is required")
            if portal.assistant.db.get_feedback(feedback_id) is None:
                return self._error(404, "That feedback was not found")
            with portal.lock:
                draft = portal.assistant.draft_rule(feedback_id)
            self._json(200, {"kind": draft.kind, "text": draft.text, "used_llm": draft.used_llm})

        def _admin_dismiss(self, data: dict) -> None:
            try:
                feedback_id = int(data.get("id"))
            except (TypeError, ValueError):
                return self._error(400, "id is required")
            if portal.assistant.db.get_feedback(feedback_id) is None:
                return self._error(404, "That feedback was not found")
            portal.assistant.db.update_feedback(feedback_id, "dismissed", note="Dismissed in the portal")
            self._json(200, {"ok": True})

        def _feedback(self, data: dict) -> None:
            s = self._require_session()
            if not s:
                return
            try:
                conversation_id = int(data.get("conversation_id"))
            except (TypeError, ValueError):
                return self._error(400, "conversation_id is required")
            verdict = data.get("verdict")
            if verdict not in ("good", "bad"):
                return self._error(400, "verdict must be 'good' or 'bad'")
            if portal.assistant.db.get_conversation(conversation_id) is None:
                return self._error(404, "That answer was not found")
            remark = str(data.get("remark", "")).strip()[:1000]
            fid = portal.assistant.db.add_feedback(
                conversation_id, verdict, 5 if verdict == "good" else 1, remark, tester=f"portal:{s['name']}",
            )
            self._json(200, {"ok": True, "feedback_id": fid})

    return Handler


def create_server(assistant, port: int = 8000, host: str = "127.0.0.1", admin_pin: str | None = None):
    portal = Portal(assistant, admin_pin or os.environ.get("FAQ_ADMIN_PIN", DEFAULT_ADMIN_PIN))
    httpd = ThreadingHTTPServer((host, port), make_handler(portal))
    httpd.daemon_threads = True
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(description="Student portal + API for the FAQ assistant")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on the network")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    from faqbot.config import load_settings
    from faqbot.pipeline import Assistant

    print("Loading models and knowledge base (about a minute)...")
    assistant = Assistant(load_settings())
    ready, message = assistant.llm.status()
    print(f"Knowledge base: {len(assistant.kb)} FAQs | LLM: {message}")

    httpd = create_server(assistant, args.port, args.host)
    url = f"http://localhost:{args.port}"
    print(f"\nStudent portal running at {url}  (Ctrl+C to stop)")
    print(f"Admin PIN: {os.environ.get('FAQ_ADMIN_PIN', DEFAULT_ADMIN_PIN)}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
