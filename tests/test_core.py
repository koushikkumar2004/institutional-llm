"""Offline tests for the pipeline, database and rules.

They use tiny fake models and a fake LLM server, so they run in seconds with
no downloads and no Ollama:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import threading
import zlib
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faqbot.config import load_settings  # noqa: E402
from faqbot.knowledge_base import KnowledgeBaseError, load_knowledge_base  # noqa: E402
from faqbot.llm import LLMClient  # noqa: E402
from faqbot.pipeline import Assistant  # noqa: E402
from faqbot.prompts import NO_MATCH_MESSAGE, NOT_IN_FAQ_MESSAGE  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "is", "of", "for", "to", "at", "in", "what", "how", "do", "i", "can", "my", "and", "are"}


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


class FakeEmbedder:
    """Bag-of-words hashing 'embedding' - good enough to test the plumbing."""

    name = "fake-embedder"

    def encode(self, texts):
        out = np.zeros((len(texts), 1024), dtype="float32")
        for i, t in enumerate(texts):
            for w in words(t):
                out[i, zlib.crc32(w.encode()) % 1024] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FakeReranker:
    name = "fake-reranker"

    def score(self, query, documents):
        q = set(words(query))
        return [float(len(q & set(words(d)))) for d in documents]


class FakeLLMHandler(BaseHTTPRequestHandler):
    last_request: dict | None = None
    reply: str = "The online application fee is Rs. 2000 per program."

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        FakeLLMHandler.last_request = json.loads(self.rfile.read(length))
        if not FakeLLMHandler.last_request.get("stream"):
            self._send({"choices": [{"message": {"content": FakeLLMHandler.reply}}]})
            return
        # Streaming: send the reply a few words at a time, like Ollama does.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        words = FakeLLMHandler.reply.split(" ")
        for i, w in enumerate(words):
            piece = w if i == 0 else " " + w
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeLLMHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(ROOT / "data" / "ruas_faqs.csv", self.tmp / "faqs.csv")
        self.settings = load_settings(overrides={
            "knowledge_base": {"path": str(self.tmp / "faqs.csv")},
            "storage": {"dir": str(self.tmp / "storage")},
            "retrieval": {"similarity_threshold": 0.30},
            "llm": {"base_url": f"http://127.0.0.1:{self.port}/v1", "model": "fake-model"},
        })
        self.assistant = self.make_assistant()
        FakeLLMHandler.reply = "The online application fee is Rs. 2000 per program."

    def make_assistant(self):
        return Assistant(self.settings, embedder=FakeEmbedder(), reranker=FakeReranker(),
                         llm=LLMClient(self.settings.llm))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- knowledge base ---------------------------------------------------
    def test_knowledge_base_loads_and_fixes_applied(self):
        kb = self.assistant.kb
        self.assertEqual(len(kb), 43)
        self.assertEqual(kb.loc[kb.id == 43, "role"].item(), "student")
        self.assertEqual(kb.loc[kb.id == 28, "role"].item(), "admin")

    def test_original_csv_with_bom_still_loads(self):
        kb = load_knowledge_base(ROOT / "data" / "original" / "ramaiah_official_faq_dataset_cleaned.csv",
                                 {"student", "admin"})
        self.assertIn("id", kb.columns)
        self.assertEqual(len(kb), 43)

    def test_bad_knowledge_base_is_rejected_with_reasons(self):
        bad = self.tmp / "bad.csv"
        bad.write_text("id,category,role,question,answer\n1,Fees,student,Q?,A\n1,Fees,faculty,Q2?,\n",
                       encoding="utf-8")
        with self.assertRaises(KnowledgeBaseError) as ctx:
            load_knowledge_base(bad, {"student", "admin"})
        msg = str(ctx.exception)
        self.assertIn("Duplicate ids", msg)
        self.assertIn("faculty", msg)
        self.assertIn("empty answer", msg)

    # --- index ------------------------------------------------------------
    def test_index_is_saved_and_reused(self):
        self.assertTrue(self.assistant.index_was_rebuilt)
        second = self.make_assistant()
        self.assertFalse(second.index_was_rebuilt)

    def test_index_rebuilds_when_csv_changes(self):
        path = self.settings.knowledge_base_path
        path.write_text(path.read_text(encoding="utf-8").replace("Rs. 2000 per program", "Rs. 2500 per program"),
                        encoding="utf-8")
        again = self.make_assistant()
        self.assertTrue(again.index_was_rebuilt)

    # --- roles ------------------------------------------------------------
    def test_student_never_sees_admin_faq(self):
        r = self.assistant.retrieve("What is the intake capacity for programs?", "student")
        self.assertNotIn(28, [c.faq_id for c in r.candidates])

    def test_admin_can_see_admin_faq(self):
        r = self.assistant.retrieve("What is the intake capacity for programs?", "admin")
        self.assertEqual(r.best.faq_id, 28)

    def test_unknown_role_is_refused(self):
        with self.assertRaises(ValueError):
            self.assistant.answer("What is the application fee?", "faculty")

    # --- answering --------------------------------------------------------
    def test_generate_mode_uses_llm_and_logs(self):
        res = self.assistant.answer("What is the application fee?", "student", mode="generate", tester="t1")
        self.assertEqual(res.best.faq_id, 3)
        self.assertEqual(res.answer_mode, "generate")
        self.assertEqual(res.status, "answered")
        row = self.assistant.db.get_conversation(res.conversation_id)
        self.assertEqual(row["faq_id"], 3)
        self.assertEqual(row["tester"], "t1")

    def test_rules_are_sent_to_llm(self):
        self.assistant.answer("What is the application fee?", "student", mode="generate")
        system = FakeLLMHandler.last_request["messages"][0]["content"]
        self.assertIn("Do not expand abbreviations", system)

    def test_direct_mode_returns_stored_answer(self):
        res = self.assistant.answer("What is the COMEDK code?", "student", mode="direct")
        self.assertEqual(res.answer, "The COMEDK code for RUAS is E197.")

    def test_out_of_scope_gets_no_match_without_llm(self):
        FakeLLMHandler.last_request = None
        res = self.assistant.answer("Who won the FIFA World Cup?", "student", mode="generate")
        self.assertEqual(res.status, "no_match")
        self.assertEqual(res.answer, NO_MATCH_MESSAGE)
        self.assertIsNone(FakeLLMHandler.last_request)

    def test_not_in_faq_reply_is_detected(self):
        FakeLLMHandler.reply = NOT_IN_FAQ_MESSAGE
        res = self.assistant.answer("What is the application fee?", "student", mode="generate")
        self.assertEqual(res.status, "not_in_faq")

    def test_llm_down_falls_back_to_stored_answer(self):
        self.settings.llm.base_url = "http://127.0.0.1:1/v1"
        self.settings.llm.timeout_seconds = 2
        a = Assistant(self.settings, embedder=FakeEmbedder(), reranker=FakeReranker(),
                      llm=LLMClient(self.settings.llm))
        res = a.answer("What is the application fee?", "student", mode="generate")
        self.assertEqual(res.answer_mode, "direct_fallback")
        self.assertIn("Rs. 2000", res.answer)
        self.assertFalse(a.llm.status()[0])

    # --- feedback -> rules ------------------------------------------------
    def test_feedback_to_rule_flow(self):
        db = self.assistant.db
        res = self.assistant.answer("What is the application fee?", "student", mode="generate")
        fb_id = db.add_feedback(res.conversation_id, "bad", 2, "It should mention this is per program.")

        FakeLLMHandler.reply = "Always mention whether a fee applies per program or in total."
        draft = self.assistant.draft_rule(fb_id)
        self.assertEqual(draft.kind, "rule")
        rule_id = db.add_rule(draft.text, status="pending", source="feedback", feedback_id=fb_id)
        db.update_feedback(fb_id, "rule_drafted")

        # Pending rules are NOT used yet ("not blindly").
        self.assistant.answer("What is the application fee?", "student", mode="generate")
        self.assertNotIn("per program or in total", FakeLLMHandler.last_request["messages"][0]["content"])

        db.update_rule(rule_id, changed_by="admin", status="active")
        db.update_feedback(fb_id, "rule_created")
        self.assistant.answer("What is the application fee?", "student", mode="generate")
        self.assertIn("per program or in total", FakeLLMHandler.last_request["messages"][0]["content"])

        actions = [h["action"] for h in db.rule_history(rule_id)]
        self.assertEqual(actions, ["created", "approved"])
        self.assertEqual(db.get_feedback(fb_id)["status"], "rule_created")

    def test_kb_fix_draft(self):
        res = self.assistant.answer("What is the application fee?", "student", mode="generate")
        fb_id = self.assistant.db.add_feedback(res.conversation_id, "bad", 1, "Fee is now 2500")
        FakeLLMHandler.reply = "KB_FIX: Update the application fee in FAQ 3."
        draft = self.assistant.draft_rule(fb_id)
        self.assertEqual(draft.kind, "kb_fix")

    def test_scoped_rule_only_applies_to_its_category(self):
        db = self.assistant.db
        db.add_rule("Mention the exam is online.", status="active", source="manual", scope="RUASAT_Exam")
        self.assistant.answer("What is the application fee?", "student", mode="generate")
        self.assertNotIn("exam is online", FakeLLMHandler.last_request["messages"][0]["content"])
        self.assistant.answer("Is there negative marking in RUASAT?", "student", mode="generate")
        self.assertIn("exam is online", FakeLLMHandler.last_request["messages"][0]["content"])

    def test_rule_edit_history_and_seed_once(self):
        db = self.assistant.db
        seeded = db.list_rules()
        self.assertEqual(len(seeded), len(self.settings.seed_rules))
        rid = seeded[0]["id"]
        db.update_rule(rid, changed_by="admin", text="Only state facts from the FAQ entry.")
        db.update_rule(rid, changed_by="admin", status="inactive")
        self.assertEqual([h["action"] for h in db.rule_history(rid)], ["created", "edited", "deactivated"])
        self.assertEqual(db.seed_rules_if_empty(self.settings.seed_rules), 0)  # never re-seeds

    def test_stats(self):
        res = self.assistant.answer("What is the application fee?", "student", mode="direct")
        self.assistant.db.add_feedback(res.conversation_id, "good", 5, "Perfect")
        s = self.assistant.db.stats()
        self.assertEqual(s["questions"], 1)
        self.assertEqual(s["avg_rating"], 5)

    # --- streaming ---------------------------------------------------------
    def test_stream_matches_answer_and_logs(self):
        events = list(self.assistant.answer_stream("What is the application fee?", "student", mode="generate"))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "meta")
        self.assertEqual(kinds[-1], "done")
        text = "".join(v for k, v in events if k == "token")
        self.assertEqual(text, "The online application fee is Rs. 2000 per program.")
        self.assertGreater(kinds.count("token"), 3)  # really arrived in pieces
        result = events[-1][1]
        self.assertEqual(result.answer, text)
        self.assertEqual(self.assistant.db.get_conversation(result.conversation_id)["faq_id"], 3)

    def test_stream_no_match_and_fallback(self):
        events = list(self.assistant.answer_stream("Who won the FIFA World Cup?", "student"))
        self.assertEqual(events[-1][1].status, "no_match")
        self.settings.llm.base_url = "http://127.0.0.1:1/v1"
        self.settings.llm.timeout_seconds = 2
        a = Assistant(self.settings, embedder=FakeEmbedder(), reranker=FakeReranker(),
                      llm=LLMClient(self.settings.llm))
        events = list(a.answer_stream("What is the application fee?", "student", mode="generate"))
        self.assertEqual(events[-1][1].answer_mode, "direct_fallback")
        self.assertIn("Rs. 2000", "".join(v for k, v in events if k == "token"))

    def test_stream_hides_think_block(self):
        FakeLLMHandler.reply = "<think> reasoning here </think> The fee is Rs. 2000."
        text = "".join(v for k, v in self.assistant.answer_stream("What is the application fee?", "student")
                       if k == "token")
        self.assertEqual(text.strip(), "The fee is Rs. 2000.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
