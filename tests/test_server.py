"""Tests for the portal API (server.py), using the same fake models as test_core.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_core as tc  # noqa: E402
from faqbot.config import load_settings  # noqa: E402
from faqbot.llm import LLMClient  # noqa: E402
from faqbot.pipeline import Assistant  # noqa: E402
from server import create_server  # noqa: E402


class PortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = HTTPServer(("127.0.0.1", 0), tc.FakeLLMHandler)
        threading.Thread(target=cls.llm.serve_forever, daemon=True).start()
        cls.tmp = Path(tempfile.mkdtemp())
        settings = load_settings(overrides={
            "storage": {"dir": str(cls.tmp)},
            "retrieval": {"similarity_threshold": 0.30},
            "llm": {"base_url": f"http://127.0.0.1:{cls.llm.server_address[1]}/v1", "model": "fake-model"},
        })
        cls.assistant = Assistant(settings, embedder=tc.FakeEmbedder(), reranker=tc.FakeReranker(),
                                  llm=LLMClient(settings.llm))
        cls.httpd = create_server(cls.assistant, port=0, admin_pin="1234")
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.llm.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, path, body=None, token=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST")
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def login(self, role="student", **extra):
        status, data = self.call("/api/login", {"role": role, **extra})
        self.assertEqual(status, 200, data)
        return data["token"]

    def test_page_and_health(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            self.assertIn(b"Admissions Assistant", r.read())
        status, data = self.call("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["faqs"], 43)
        self.assertTrue(data["llm_ready"])

    def test_ask_requires_sign_in(self):
        status, _ = self.call("/api/ask", {"question": "What is the application fee?"})
        self.assertEqual(status, 401)

    def test_student_ask_and_feedback(self):
        token = self.login(name="Priya")
        status, me = self.call("/api/me", token=token)
        self.assertEqual(me, {"role": "student", "name": "Priya"})
        status, r = self.call("/api/ask", {"question": "What is the application fee?"}, token)
        self.assertEqual(status, 200, r)
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["source"]["faq_id"], 3)
        status, fb = self.call("/api/feedback", {"conversation_id": r["conversation_id"], "verdict": "bad",
                                                 "remark": "Should mention it's per program"}, token)
        self.assertEqual(status, 200, fb)
        saved = self.assistant.db.get_feedback(fb["feedback_id"])
        self.assertEqual(saved["tester"], "portal:Priya")
        self.assertEqual(saved["rating"], 1)

    def test_role_comes_from_session_not_request(self):
        token = self.login()
        status, r = self.call("/api/ask", {"question": "What is the intake capacity for programs?", "role": "admin"},
                              token)
        self.assertEqual(status, 200)
        self.assertNotEqual((r["source"] or {}).get("faq_id"), 28)
        admin = self.login("admin", pin="1234")
        status, r = self.call("/api/ask", {"question": "What is the intake capacity for programs?"}, admin)
        self.assertEqual(r["source"]["faq_id"], 28)

    def test_wrong_pin_and_unknown_role(self):
        self.assertEqual(self.call("/api/login", {"role": "admin", "pin": "nope"})[0], 403)
        self.assertEqual(self.call("/api/login", {"role": "faculty"})[0], 400)

    def test_out_of_scope_has_no_source(self):
        token = self.login()
        status, r = self.call("/api/ask", {"question": "Who won the FIFA World Cup?"}, token)
        self.assertEqual(r["status"], "no_match")
        self.assertIsNone(r["source"])

    def test_bad_input(self):
        token = self.login()
        self.assertEqual(self.call("/api/ask", {"question": "   "}, token)[0], 400)
        self.assertEqual(self.call("/api/ask", {"question": "x" * 600}, token)[0], 400)
        self.assertEqual(self.call("/api/feedback", {"conversation_id": 99999, "verdict": "good"}, token)[0], 404)
        self.assertEqual(self.call("/api/feedback", {"conversation_id": 1, "verdict": "meh"}, token)[0], 400)

    def test_logout_ends_session(self):
        token = self.login()
        self.call("/api/logout", {}, token)
        self.assertEqual(self.call("/api/me", token=token)[0], 401)

    def stream(self, question, token):
        req = urllib.request.Request(self.base + "/api/ask/stream", data=json.dumps({"question": question}).encode(),
                                     method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=10) as r:
            return [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]

    def test_stream_endpoint(self):
        token = self.login(name="Ravi")
        events = self.stream("What is the application fee?", token)
        tokens = [e["text"] for e in events if e["type"] == "token"]
        self.assertGreater(len(tokens), 3)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual("".join(tokens), done["answer"])
        self.assertEqual(done["source"]["faq_id"], 3)
        self.assertIsNotNone(self.assistant.db.get_conversation(done["conversation_id"]))

    def test_admin_endpoints_need_admin(self):
        student = self.login()
        self.assertEqual(self.call("/api/admin/summary", token=student)[0], 403)
        self.assertEqual(self.call("/api/admin/rules", {"text": "x"}, student)[0], 403)
        self.assertEqual(self.call("/api/admin/summary")[0], 401)

    def test_admin_feedback_to_rule_flow(self):
        student = self.login(name="Asha")
        _, r = self.call("/api/ask", {"question": "What is the COMEDK code?"}, student)
        _, fb = self.call("/api/feedback", {"conversation_id": r["conversation_id"], "verdict": "bad",
                                            "remark": "Say it's from the official FAQs"}, student)
        admin = self.login("admin", pin="1234")
        status, summary = self.call("/api/admin/summary", token=admin)
        self.assertEqual(status, 200)
        item = next(f for f in summary["feedback"] if f["id"] == fb["feedback_id"])
        self.assertEqual(item["who"], "Asha")
        self.assertIn("all", summary["categories"])

        tc.FakeLLMHandler.reply = "Always say answers come from the official FAQs."
        status, draft = self.call("/api/admin/feedback/draft", {"id": fb["feedback_id"]}, admin)
        tc.FakeLLMHandler.reply = "The online application fee is Rs. 2000 per program."
        self.assertEqual(status, 200)
        self.assertEqual(draft["text"], "Always say answers come from the official FAQs.")

        status, added = self.call("/api/admin/rules", {"text": draft["text"], "feedback_id": fb["feedback_id"]}, admin)
        self.assertEqual(status, 200, added)
        rule = self.assistant.db.get_rule(added["rule_id"])
        self.assertEqual((rule["status"], rule["source"]), ("active", "feedback"))
        self.assertEqual(self.assistant.db.get_feedback(fb["feedback_id"])["status"], "rule_created")

        self.call("/api/ask", {"question": "What is the COMEDK code?"}, student)
        self.assertIn("official FAQs", tc.FakeLLMHandler.last_request["messages"][0]["content"])

        status, _ = self.call("/api/admin/rules/status", {"id": added["rule_id"], "status": "inactive"}, admin)
        self.assertEqual(status, 200)
        self.assertEqual(self.assistant.db.get_rule(added["rule_id"])["status"], "inactive")
        self.assertEqual(self.call("/api/admin/rules/status", {"id": added["rule_id"], "status": "bogus"}, admin)[0], 400)
        self.assertEqual(self.call("/api/admin/rules", {"text": "x", "scope": "Nope"}, admin)[0], 400)

    def test_admin_dismiss(self):
        student = self.login()
        _, r = self.call("/api/ask", {"question": "What is the application fee?"}, student)
        _, fb = self.call("/api/feedback", {"conversation_id": r["conversation_id"], "verdict": "good"}, student)
        admin = self.login("admin", pin="1234")
        self.assertEqual(self.call("/api/admin/feedback/dismiss", {"id": fb["feedback_id"]}, admin)[0], 200)
        self.assertEqual(self.assistant.db.get_feedback(fb["feedback_id"])["status"], "dismissed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
