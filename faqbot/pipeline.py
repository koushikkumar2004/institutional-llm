"""The full question-answering pipeline.

    question + role
        -> role check                      (unknown roles are refused)
        -> FAISS search (role's own index) (top_k candidates, cosine similarity)
        -> similarity threshold            (nothing close enough -> "not in the FAQs")
        -> CrossEncoder rerank             (question + answer text, best one wins)
        -> active rules for that FAQ       (from the database, editable in the app)
        -> LLM writes the answer           (or the stored answer in "direct" mode)
        -> conversation logged             (so it can be rated and reviewed)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import pandas as pd

from .config import Settings
from .database import Database
from .index import FAQIndex, StaleIndexError
from .knowledge_base import load_knowledge_base
from .llm import LLMClient, LLMError
from .models import Embedder, EmbedderProtocol, Reranker, RerankerProtocol
from .prompts import (
    KB_FIX_PREFIX, NO_MATCH_MESSAGE, NOT_IN_FAQ_MESSAGE, answer_messages, rule_draft_messages,
)


@dataclass
class Candidate:
    faq_id: int
    category: str
    role: str
    question: str
    answer: str
    similarity: float
    passed_threshold: bool = False
    rerank_score: float | None = None


@dataclass
class Retrieval:
    candidates: list[Candidate]           # FAISS shortlist, in FAISS order
    reranked: list[Candidate]             # those above the threshold, in reranker order
    best: Candidate | None                # what the answer will be based on
    rejected_reason: str | None = None    # why nothing was chosen


@dataclass
class AnswerResult:
    conversation_id: int | None
    question: str
    role: str
    answer: str
    status: str                 # answered | no_match | not_in_faq
    answer_mode: str            # generate | direct | direct_fallback
    best: Candidate | None
    candidates: list[Candidate]
    rules_applied: list[dict[str, Any]] = field(default_factory=list)
    llm_model: str | None = None
    latency_ms: int = 0
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleDraft:
    kind: str        # rule | kb_fix
    text: str
    used_llm: bool


class Assistant:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderProtocol | None = None,
        reranker: RerankerProtocol | None = None,
        llm: LLMClient | None = None,
        database: Database | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or Embedder(settings.embedding_model)
        self.reranker = reranker or Reranker(settings.reranker_model)
        self.llm = llm or LLMClient(settings.llm)
        self.db = database or Database(settings.database_path, settings.tenant_id)
        self.db.seed_rules_if_empty(settings.seed_rules)
        self.kb: pd.DataFrame = pd.DataFrame()
        self.index: FAQIndex | None = None
        self.index_was_rebuilt = False
        self.load_knowledge_base()

    # ---------------------------------------------------------- knowledge base
    def load_knowledge_base(self, force_rebuild: bool = False) -> None:
        """Load the CSV and the saved index; rebuild the index only if it is missing or out of date."""
        s = self.settings
        self.kb = load_knowledge_base(s.knowledge_base_path, s.faq_roles)
        self.index_was_rebuilt = False
        if not force_rebuild:
            try:
                self.index = FAQIndex.load(s.index_dir, self.kb, self.embedder.name, s.roles)
                return
            except (StaleIndexError, FileNotFoundError, OSError, ValueError, KeyError):
                pass
        self.index = FAQIndex.build(self.kb, self.embedder, s.roles)
        self.index.save(s.index_dir)
        self.index_was_rebuilt = True

    # --------------------------------------------------------------- retrieval
    def check_role(self, role: str) -> None:
        if role not in self.settings.roles:
            raise ValueError(f"Unknown role '{role}'. Allowed: {sorted(self.settings.roles)}")

    def retrieve(self, question: str, role: str) -> Retrieval:
        self.check_role(role)
        assert self.index is not None
        s = self.settings
        query_vec = self.embedder.encode([question])[0]
        hits = self.index.search(query_vec, role, s.top_k)

        candidates = []
        for hit in hits:
            row = self.kb.iloc[hit.row]
            candidates.append(Candidate(
                faq_id=int(row["id"]), category=row["category"], role=row["role"],
                question=row["question"], answer=row["answer"], similarity=round(hit.similarity, 4),
                passed_threshold=hit.similarity >= s.similarity_threshold,
            ))

        relevant = [c for c in candidates if c.passed_threshold]
        if not relevant:
            return Retrieval(candidates, [], None, "no FAQ above the similarity threshold")

        scores = self.reranker.score(question, [f"Question: {c.question} Answer: {c.answer}" for c in relevant])
        for c, score in zip(relevant, scores):
            c.rerank_score = round(float(score), 4)
        reranked = sorted(relevant, key=lambda c: c.rerank_score, reverse=True)
        best = reranked[0]

        if s.rerank_min_score is not None and best.rerank_score < s.rerank_min_score:
            return Retrieval(candidates, reranked, None, "best reranker score below rerank_min_score")
        return Retrieval(candidates, reranked, best)

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, role: str, mode: str | None = None,
               tester: str | None = None, log: bool = True) -> AnswerResult:
        started = time.perf_counter()
        question = question.strip()
        mode = mode or self.settings.answer_mode
        if mode not in ("generate", "direct"):
            raise ValueError("mode must be 'generate' or 'direct'")

        retrieval = self.retrieve(question, role)
        best = retrieval.best
        rules: list[dict[str, Any]] = []
        llm_model = None
        note = None

        if best is None:
            status, text, used_mode = "no_match", NO_MATCH_MESSAGE, mode
            note = retrieval.rejected_reason
        elif mode == "direct":
            status, text, used_mode = "answered", best.answer, "direct"
        else:
            rules = self.db.active_rules_for(best.category)
            messages = answer_messages(
                self.settings.tenant_name, question, best.question, best.answer, [r["text"] for r in rules]
            )
            try:
                text = self.llm.chat(messages)
                llm_model = self.llm.model
                used_mode = "generate"
                if not text:
                    text, used_mode = best.answer, "direct_fallback"
                    note = "LLM returned an empty reply; showing the stored FAQ answer."
                status = "not_in_faq" if NOT_IN_FAQ_MESSAGE.rstrip(".").lower() in text.lower() else "answered"
            except LLMError as exc:
                text, status, used_mode = best.answer, "answered", "direct_fallback"
                note = f"LLM unavailable, showing the stored FAQ answer. ({exc})"

        return self._finalize(started, question, role, retrieval, text, status, used_mode, rules,
                              llm_model, note, tester, log)

    def answer_stream(self, question: str, role: str, mode: str | None = None,
                      tester: str | None = None, log: bool = True) -> Iterator[tuple[str, Any]]:
        """Same as answer(), but yields the answer as it is written.

        Yields ("meta", Candidate | None) once retrieval is done, then ("token", text) pieces,
        then ("done", AnswerResult) after the conversation is logged.
        """
        started = time.perf_counter()
        question = question.strip()
        mode = mode or self.settings.answer_mode
        if mode not in ("generate", "direct"):
            raise ValueError("mode must be 'generate' or 'direct'")

        retrieval = self.retrieve(question, role)
        best = retrieval.best
        yield ("meta", best)
        rules: list[dict[str, Any]] = []
        llm_model = None
        note = None

        if best is None:
            status, text, used_mode = "no_match", NO_MATCH_MESSAGE, mode
            note = retrieval.rejected_reason
            yield ("token", text)
        elif mode == "direct":
            status, text, used_mode = "answered", best.answer, "direct"
            yield ("token", text)
        else:
            rules = self.db.active_rules_for(best.category)
            messages = answer_messages(
                self.settings.tenant_name, question, best.question, best.answer, [r["text"] for r in rules]
            )
            parts: list[str] = []
            try:
                for piece in self.llm.chat_stream(messages):
                    parts.append(piece)
                    yield ("token", piece)
                text, used_mode, llm_model = "".join(parts).strip(), "generate", self.llm.model
                if not text:
                    text, used_mode = best.answer, "direct_fallback"
                    note = "LLM returned an empty reply; showing the stored FAQ answer."
                    yield ("token", text)
            except LLMError as exc:
                if parts:
                    text, used_mode, llm_model = "".join(parts).strip(), "generate", self.llm.model
                    note = f"LLM stopped early. ({exc})"
                else:
                    text, used_mode = best.answer, "direct_fallback"
                    note = f"LLM unavailable, showing the stored FAQ answer. ({exc})"
                    yield ("token", text)
            status = "not_in_faq" if NOT_IN_FAQ_MESSAGE.rstrip(".").lower() in text.lower() else "answered"

        yield ("done", self._finalize(started, question, role, retrieval, text, status, used_mode, rules,
                                      llm_model, note, tester, log))

    def _finalize(self, started: float, question: str, role: str, retrieval: Retrieval, text: str,
                  status: str, used_mode: str, rules: list[dict[str, Any]], llm_model: str | None,
                  note: str | None, tester: str | None, log: bool) -> AnswerResult:
        best = retrieval.best
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = AnswerResult(
            conversation_id=None, question=question, role=role, answer=text, status=status,
            answer_mode=used_mode, best=best, candidates=retrieval.candidates,
            rules_applied=[{"id": r["id"], "text": r["text"], "scope": r["scope"]} for r in rules],
            llm_model=llm_model, latency_ms=latency_ms, note=note,
        )
        if log:
            result.conversation_id = self.db.log_conversation(
                tester=tester, user_role=role, question=question, answer=text, status=status,
                answer_mode=used_mode, faq_id=best.faq_id if best else None,
                faq_category=best.category if best else None,
                similarity=best.similarity if best else None,
                rerank_score=best.rerank_score if best else None,
                candidates_json=[asdict(c) for c in retrieval.candidates],
                rules_json=result.rules_applied, llm_model=llm_model, latency_ms=latency_ms, note=note,
            )
        return result

    # ------------------------------------------------- feedback -> rule drafts
    def draft_rule(self, feedback_id: int) -> RuleDraft:
        """Suggest a rule from one piece of feedback. Nothing changes until a human approves it."""
        fb = self.db.get_feedback(feedback_id)
        if fb is None:
            raise ValueError(f"Feedback {feedback_id} not found")
        faq = None
        if fb.get("faq_id") is not None:
            match = self.kb[self.kb["id"] == fb["faq_id"]]
            faq = match.iloc[0] if not match.empty else None
        messages = rule_draft_messages(
            question=fb["question"], bot_answer=fb["answer"],
            faq_question=None if faq is None else faq["question"],
            faq_answer=None if faq is None else faq["answer"],
            verdict=fb["verdict"], remark=fb.get("remark") or "",
        )
        try:
            text = self.llm.chat(messages, max_tokens=120).strip().strip('"')
        except LLMError:
            return RuleDraft(kind="rule", text=(fb.get("remark") or "").strip(), used_llm=False)
        if text.upper().startswith(KB_FIX_PREFIX):
            return RuleDraft(kind="kb_fix", text=text[len(KB_FIX_PREFIX):].strip(), used_llm=True)
        return RuleDraft(kind="rule", text=text.splitlines()[0].strip() if text else "", used_llm=True)

    def categories(self) -> list[str]:
        return sorted(self.kb["category"].unique().tolist())
