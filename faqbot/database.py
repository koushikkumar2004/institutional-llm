"""SQLite storage for conversations, feedback and rules.

SQLite is a single file (storage/<tenant>.sqlite3) and needs no server, which is
right for development and the demo. Every table carries a tenant_id, so moving
to PostgreSQL for multiple universities later is a connection change, not a
redesign.

Tables
  conversations - every question asked, what the system picked and answered
  feedback      - ratings and remarks, linked to a conversation
  rules         - editable behaviour rules (pending / active / inactive / rejected)
  rule_history  - every change to a rule, so you can see how rules evolved
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    tester        TEXT,
    user_role     TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    status        TEXT NOT NULL,      -- answered | no_match | not_in_faq
    answer_mode   TEXT NOT NULL,      -- generate | direct | direct_fallback
    faq_id        INTEGER,
    faq_category  TEXT,
    similarity    REAL,
    rerank_score  REAL,
    candidates_json TEXT,
    rules_json    TEXT,
    llm_model     TEXT,
    latency_ms    INTEGER,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    tester          TEXT,
    verdict         TEXT NOT NULL,    -- good | partly | bad
    rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    remark          TEXT,
    issue_type      TEXT,             -- wrong_fact | wrong_faq | bad_behaviour | missing_info | praise | other
    status          TEXT NOT NULL DEFAULT 'new',  -- new | rule_drafted | rule_created | kb_fix | dismissed
    resolution_note TEXT,
    resolved_at     TEXT
);

CREATE TABLE IF NOT EXISTS rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL,
    text        TEXT NOT NULL,
    reason      TEXT,
    scope       TEXT NOT NULL DEFAULT 'all',   -- 'all' or an FAQ category
    status      TEXT NOT NULL,                 -- pending | active | inactive | rejected
    source      TEXT NOT NULL,                 -- seed | feedback | manual
    feedback_id INTEGER REFERENCES feedback(id) ON DELETE SET NULL,
    created_by  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id    INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    changed_at TEXT NOT NULL,
    changed_by TEXT,
    action     TEXT NOT NULL,      -- created | edited | approved | rejected | activated | deactivated
    old_text   TEXT,
    new_text   TEXT,
    old_status TEXT,
    new_status TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_tenant ON conversations(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fb_tenant ON feedback(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_rules_tenant ON rules(tenant_id, status);
"""

RULE_STATUSES = ("pending", "active", "inactive", "rejected")
FEEDBACK_STATUSES = ("new", "rule_drafted", "rule_created", "kb_fix", "dismissed")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path, tenant_id: str):
        self.path = Path(path)
        self.tenant_id = tenant_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ----------------------------------------------------------- conversations
    def log_conversation(self, **fields: Any) -> int:
        fields = dict(fields)
        fields.setdefault("created_at", _now())
        fields["tenant_id"] = self.tenant_id
        for key in ("candidates_json", "rules_json"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], ensure_ascii=False)
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self._connect() as conn:
            cur = conn.execute(f"INSERT INTO conversations ({cols}) VALUES ({marks})", tuple(fields.values()))
            return int(cur.lastrowid)

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT * FROM conversations WHERE id = ? AND tenant_id = ?", (conversation_id, self.tenant_id)
        )
        return rows[0] if rows else None

    def recent_conversations(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM conversations WHERE tenant_id = ? ORDER BY id DESC LIMIT ?", (self.tenant_id, limit)
        )

    # ---------------------------------------------------------------- feedback
    def add_feedback(
        self, conversation_id: int, verdict: str, rating: int, remark: str = "",
        tester: str | None = None, issue_type: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO feedback (conversation_id, tenant_id, created_at, tester, verdict, rating, remark, issue_type)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, self.tenant_id, _now(), tester, verdict, int(rating), remark.strip(), issue_type),
            )
            return int(cur.lastrowid)

    def list_feedback(self, statuses: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT f.*, c.question, c.answer, c.user_role, c.faq_id, c.faq_category, c.status AS answer_status,"
            " c.answer_mode, c.rules_json, c.candidates_json"
            " FROM feedback f JOIN conversations c ON c.id = f.conversation_id"
            " WHERE f.tenant_id = ?"
        )
        params: list[Any] = [self.tenant_id]
        if statuses:
            sql += f" AND f.status IN ({', '.join('?' for _ in statuses)})"
            params += list(statuses)
        return self._rows(sql + " ORDER BY f.id DESC", tuple(params))

    def get_feedback(self, feedback_id: int) -> dict[str, Any] | None:
        rows = [f for f in self.list_feedback() if f["id"] == feedback_id]
        return rows[0] if rows else None

    def update_feedback(self, feedback_id: int, status: str, note: str | None = None,
                        issue_type: str | None = None) -> None:
        if status not in FEEDBACK_STATUSES:
            raise ValueError(f"Unknown feedback status {status}")
        resolved = None if status in ("new", "rule_drafted") else _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE feedback SET status = ?, resolution_note = COALESCE(?, resolution_note),"
                " issue_type = COALESCE(?, issue_type), resolved_at = ? WHERE id = ? AND tenant_id = ?",
                (status, note, issue_type, resolved, feedback_id, self.tenant_id),
            )

    # ------------------------------------------------------------------- rules
    def add_rule(
        self, text: str, status: str, source: str, reason: str | None = None, scope: str = "all",
        feedback_id: int | None = None, created_by: str | None = None,
    ) -> int:
        text = " ".join(text.split())
        if not text:
            raise ValueError("Rule text cannot be empty")
        if status not in RULE_STATUSES:
            raise ValueError(f"Unknown rule status {status}")
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO rules (tenant_id, text, reason, scope, status, source, feedback_id, created_by,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self.tenant_id, text, reason, scope or "all", status, source, feedback_id, created_by, now, now),
            )
            rule_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO rule_history (rule_id, changed_at, changed_by, action, new_text, new_status)"
                " VALUES (?, ?, ?, 'created', ?, ?)",
                (rule_id, now, created_by, text, status),
            )
        return rule_id

    def update_rule(
        self, rule_id: int, changed_by: str | None = None, text: str | None = None,
        status: str | None = None, scope: str | None = None, reason: str | None = None,
    ) -> None:
        current = self.get_rule(rule_id)
        if current is None:
            raise ValueError(f"Rule {rule_id} not found")
        new_text = " ".join(text.split()) if text is not None else current["text"]
        if not new_text:
            raise ValueError("Rule text cannot be empty")
        new_status = status or current["status"]
        if new_status not in RULE_STATUSES:
            raise ValueError(f"Unknown rule status {new_status}")
        new_scope = scope or current["scope"]
        new_reason = reason if reason is not None else current["reason"]

        old_status = current["status"]
        if new_status == old_status:
            action = "edited"
        elif new_status == "rejected":
            action = "rejected"
        elif new_status == "active":
            action = "approved" if old_status == "pending" else "activated"
        else:
            action = "deactivated"
        if (new_text, new_status, new_scope, new_reason) == (
            current["text"], current["status"], current["scope"], current["reason"]
        ):
            return

        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE rules SET text = ?, status = ?, scope = ?, reason = ?, updated_at = ?"
                " WHERE id = ? AND tenant_id = ?",
                (new_text, new_status, new_scope, new_reason, now, rule_id, self.tenant_id),
            )
            conn.execute(
                "INSERT INTO rule_history (rule_id, changed_at, changed_by, action, old_text, new_text,"
                " old_status, new_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rule_id, now, changed_by, action, current["text"], new_text, current["status"], new_status),
            )

    def delete_rule(self, rule_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM rules WHERE id = ? AND tenant_id = ?", (rule_id, self.tenant_id))

    def get_rule(self, rule_id: int) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM rules WHERE id = ? AND tenant_id = ?", (rule_id, self.tenant_id))
        return rows[0] if rows else None

    def list_rules(self, statuses: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM rules WHERE tenant_id = ?"
        params: list[Any] = [self.tenant_id]
        if statuses:
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            params += list(statuses)
        return self._rows(sql + " ORDER BY id", tuple(params))

    def active_rules_for(self, category: str | None) -> list[dict[str, Any]]:
        """Active rules that apply to an answer from the given FAQ category."""
        return self._rows(
            "SELECT * FROM rules WHERE tenant_id = ? AND status = 'active' AND (scope = 'all' OR scope = ?)"
            " ORDER BY id",
            (self.tenant_id, category or ""),
        )

    def rule_history(self, rule_id: int) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM rule_history WHERE rule_id = ? ORDER BY id", (rule_id,))

    def seed_rules_if_empty(self, seed_rules: list[dict[str, str]]) -> int:
        """Load starter rules the first time only, so later edits are never overwritten."""
        if self._rows("SELECT id FROM rules WHERE tenant_id = ? LIMIT 1", (self.tenant_id,)):
            return 0
        for rule in seed_rules:
            self.add_rule(
                text=rule["text"], reason=rule.get("reason"), scope=rule.get("scope", "all"),
                status=rule.get("status", "active"), source="seed", created_by="setup",
            )
        return len(seed_rules)

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            q = lambda sql: conn.execute(sql, (self.tenant_id,)).fetchone()[0]  # noqa: E731
            return {
                "questions": q("SELECT COUNT(*) FROM conversations WHERE tenant_id = ?"),
                "answered": q("SELECT COUNT(*) FROM conversations WHERE tenant_id = ? AND status = 'answered'"),
                "no_match": q("SELECT COUNT(*) FROM conversations WHERE tenant_id = ? AND status != 'answered'"),
                "feedback": q("SELECT COUNT(*) FROM feedback WHERE tenant_id = ?"),
                "avg_rating": q("SELECT AVG(rating) FROM feedback WHERE tenant_id = ?"),
                "open_feedback": q("SELECT COUNT(*) FROM feedback WHERE tenant_id = ? AND status = 'new'"),
                "active_rules": q("SELECT COUNT(*) FROM rules WHERE tenant_id = ? AND status = 'active'"),
                "pending_rules": q("SELECT COUNT(*) FROM rules WHERE tenant_id = ? AND status = 'pending'"),
            }
