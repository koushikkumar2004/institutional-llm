"""Testing console for the University FAQ Assistant.

Run:  streamlit run app.py

Pages
  Chat             - ask questions as a student or admin, see what happened behind the scenes, rate answers
  Feedback review  - turn ratings and remarks into rules (a human always approves)
  Rules            - add, edit, switch off and review the history of rules
  Knowledge base   - browse the FAQs, rebuild the index after editing the CSV
  Activity         - numbers, conversation log and CSV exports

This is the internal "dummy UI" for testing. The real product frontend will talk
to the same pipeline through an API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from faqbot.config import load_settings  # noqa: E402
from faqbot.knowledge_base import KnowledgeBaseError  # noqa: E402
from faqbot.pipeline import Assistant  # noqa: E402

st.set_page_config(page_title="University FAQ Assistant", page_icon="🎓", layout="wide")

PAGES = ["💬 Chat", "📝 Feedback review", "📏 Rules", "📚 Knowledge base", "📊 Activity"]
VERDICTS = {"👍 Good": "good", "➖ Partly right": "partly", "👎 Bad": "bad"}
VERDICT_ICON = {"good": "👍", "partly": "➖", "bad": "👎"}
ISSUE_TYPES = {
    "(not sure)": None,
    "Wrong or outdated fact in the FAQ": "wrong_fact",
    "Picked the wrong FAQ": "wrong_faq",
    "Bad behaviour (invented details, tone, length)": "bad_behaviour",
    "Information missing from the FAQs": "missing_info",
    "Praise: keep doing this": "praise",
    "Other": "other",
}
ISSUE_LABEL = {v: k for k, v in ISSUE_TYPES.items()}
STATUS_LABEL = {
    "answered": "✅ answered",
    "no_match": "🚫 no matching FAQ",
    "not_in_faq": "🤷 FAQ found but it doesn't answer this",
}
MODE_LABEL = {
    "generate": "LLM-written",
    "direct": "stored FAQ answer",
    "direct_fallback": "stored FAQ answer (LLM offline)",
}
EXAMPLES = [
    "How much money do I need to pay to apply?",
    "What is the COMEDK code?",
    "Can I change my program after admission?",
    "What is the intake capacity?",
    "Who won the FIFA World Cup?",
    "ಅರ್ಜಿ ಶುಲ್ಕ ಎಷ್ಟು?",
]


# --------------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading models and knowledge base (the first start takes a minute)...")
def get_assistant() -> Assistant:
    return Assistant(load_settings())


@st.cache_data(ttl=15, show_spinner=False)
def llm_status() -> tuple[bool, str]:
    return get_assistant().llm.status()


try:
    assistant = get_assistant()
except KnowledgeBaseError as exc:
    st.error(f"The knowledge base has problems. Fix the CSV and reload.\n\n```\n{exc}\n```")
    st.stop()
except Exception as exc:  # noqa: BLE001 - show any start-up problem clearly
    st.error(
        "The assistant could not start. Run `python scripts/prepare.py` in the terminal to see what is "
        f"missing.\n\n`{type(exc).__name__}: {exc}`"
    )
    st.stop()

settings = assistant.settings
db = assistant.db
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("rated", set())

flash = ss.pop("flash", None)
if flash:
    st.toast(flash)


def set_flash(message: str) -> None:
    ss["flash"] = message


def fmt(value, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### 🎓 {settings.tenant_name}")
    st.caption("FAQ assistant · testing console")
    page = st.radio("Page", PAGES, label_visibility="collapsed")
    st.divider()
    role = st.selectbox("Ask as", list(settings.roles), help="Admins can also see admin-only FAQs.")
    mode_label = st.radio(
        "Answer mode",
        ["Generate with LLM", "Stored FAQ answer"],
        index=0 if settings.answer_mode == "generate" else 1,
        help="Generate: the LLM rewrites the chosen FAQ and follows the rules. "
        "Stored: the FAQ's own answer, word for word (no LLM).",
    )
    mode = "generate" if mode_label == "Generate with LLM" else "direct"
    tester = st.text_input("Your name (saved with feedback)", key="tester").strip() or None
    ok, message = llm_status()
    (st.success if ok else st.warning)(f"LLM: {message}")
    stats = db.stats()
    st.caption(
        f"{stats['open_feedback']} feedback to review · {stats['pending_rules']} rules pending · "
        f"{stats['active_rules']} active"
    )
    if st.button("Clear chat"):
        ss["chat"] = []
        st.rerun()


# ---------------------------------------------------------------- chat page
def render_details(r: dict) -> None:
    best = r["best"]
    if best:
        st.markdown(f"**Chosen FAQ #{best['faq_id']}** · {best['category']}  \n*{best['question']}*")
        st.markdown(f"> {best['answer']}")
    else:
        st.markdown(f"**No FAQ chosen** · {r.get('note') or ''}")
    st.caption(
        f"Similarity threshold {settings.similarity_threshold:.2f} · FAISS shortlist of {settings.top_k} · "
        "reranker scores are relevance scores, not probabilities"
    )
    if r["candidates"]:
        table = pd.DataFrame(r["candidates"])[
            ["faq_id", "similarity", "passed_threshold", "rerank_score", "question"]
        ].rename(columns={
            "faq_id": "FAQ", "similarity": "FAISS similarity", "passed_threshold": "above threshold",
            "rerank_score": "reranker score", "question": "FAQ question",
        })
        st.dataframe(table, hide_index=True)
    if r["rules_applied"]:
        st.markdown("**Rules sent to the LLM**")
        for rule in r["rules_applied"]:
            st.markdown(f"- #{rule['id']} {rule['text']}")
    elif r["answer_mode"] == "generate":
        st.caption("No active rules applied.")
    if r.get("llm_model"):
        st.caption(f"LLM: {r['llm_model']}")


def render_feedback(r: dict, expanded: bool) -> None:
    cid = r["conversation_id"]
    if cid in ss["rated"]:
        st.caption("✅ Feedback saved, thanks.")
        return
    with st.expander("Rate this answer", expanded=expanded):
        with st.form(f"fb_{cid}"):
            c1, c2 = st.columns([3, 2])
            verdict = c1.radio("How was this answer?", list(VERDICTS), horizontal=True, index=None)
            rating = c2.select_slider("Rating", options=[1, 2, 3, 4, 5], value=3)
            issue = st.selectbox("What kind of issue? (optional)", list(ISSUE_TYPES))
            remark = st.text_area(
                "Remark",
                placeholder="e.g. It invented an admissions committee; the FAQ only says it depends on "
                "program, timing and availability.",
            )
            if st.form_submit_button("Save feedback"):
                if verdict is None:
                    st.warning("Pick Good, Partly right or Bad first.")
                else:
                    db.add_feedback(cid, VERDICTS[verdict], rating, remark, tester, ISSUE_TYPES[issue])
                    ss["rated"].add(cid)
                    set_flash("Feedback saved. Review it on the Feedback review page.")
                    st.rerun()


def page_chat() -> None:
    st.title("💬 Ask a question")
    st.caption(
        f"Asking as **{role}** · answers are **{mode_label.lower()}** · "
        "open *Behind the scenes* to see which FAQ was picked and why"
    )
    if not ss["chat"]:
        st.markdown("**Try one of these:**")
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 3].button(example, key=f"ex_{i}"):
                ss["pending_question"] = example
                st.rerun()

    for i, r in enumerate(ss["chat"]):
        with st.chat_message("user"):
            st.markdown(r["question"])
            st.caption(f"asked as {r['role']}")
        with st.chat_message("assistant"):
            st.markdown(r["answer"])
            parts = [STATUS_LABEL.get(r["status"], r["status"])]
            if r["status"] == "no_match":
                parts.append("LLM not called")
            else:
                parts.append(MODE_LABEL.get(r["answer_mode"], r["answer_mode"]))
            if r["best"]:
                parts.insert(1, f"FAQ #{r['best']['faq_id']}")
            parts.append(f"{r['latency_ms'] / 1000:.1f}s")
            st.caption(" · ".join(parts))
            if r.get("note") and r["answer_mode"] == "direct_fallback":
                st.caption(f"⚠️ {r['note']}")
            with st.expander("Behind the scenes"):
                render_details(r)
            render_feedback(r, expanded=(i == len(ss["chat"]) - 1))

    typed = st.chat_input("Ask about admissions, fees, RUASAT, programs...")
    question = typed or ss.pop("pending_question", None)
    if question:
        with st.spinner("Finding the right FAQ and writing the answer..."):
            try:
                result = assistant.answer(question, role, mode=mode, tester=tester)
            except ValueError as exc:
                st.error(str(exc))
                return
        ss["chat"].append(result.to_dict())
        st.rerun()


# ---------------------------------------------------------- feedback review
def page_feedback() -> None:
    st.title("📝 Feedback review")
    st.caption(
        "Each remark is only a suggestion. It changes nothing until a person turns it into a rule and "
        "approves it, or fixes the knowledge base. That is the \"learn, but not blindly\" step."
    )
    view = st.radio("Show", ["Needs review", "Rule drafted", "Done", "All"], horizontal=True)
    statuses = {
        "Needs review": ("new",),
        "Rule drafted": ("rule_drafted",),
        "Done": ("rule_created", "kb_fix", "dismissed"),
        "All": None,
    }[view]
    items = db.list_feedback(statuses)
    if not items:
        st.info("Nothing here yet. Rate some answers on the Chat page.")
        return

    categories = ["all"] + assistant.categories()
    for fb in items:
        fid = fb["id"]
        with st.container(border=True):
            st.markdown(
                f"**#{fid}** · {VERDICT_ICON.get(fb['verdict'], '')} {fb['verdict']} · {fb['rating']}/5 · "
                f"{fb['tester'] or 'anonymous'} · {fb['created_at']} · status **{fb['status']}**"
            )
            st.markdown(f"**Q ({fb['user_role']}):** {fb['question']}")
            st.markdown(f"**Bot:** {fb['answer']}")
            faq_txt = f"FAQ #{fb['faq_id']} ({fb['faq_category']})" if fb["faq_id"] else "no FAQ matched"
            st.caption(f"{faq_txt} · {MODE_LABEL.get(fb['answer_mode'], fb['answer_mode'])}")
            if fb["remark"]:
                st.markdown(f"**Remark:** {fb['remark']}")
            if fb["issue_type"]:
                st.caption(f"Issue type: {ISSUE_LABEL.get(fb['issue_type'], fb['issue_type'])}")
            if fb["resolution_note"]:
                st.caption(f"Resolution: {fb['resolution_note']}")

            if fb["status"] not in ("new", "rule_drafted"):
                continue

            draft_key = f"draft_{fid}"
            c1, c2, c3 = st.columns(3)
            if c1.button("✨ Draft a rule from this remark", key=f"d_{fid}"):
                with st.spinner("Asking the LLM to draft a rule..."):
                    d = assistant.draft_rule(fid)
                ss[draft_key] = {"kind": d.kind, "text": d.text, "used_llm": d.used_llm}
                st.rerun()
            if c2.button("📚 Needs a knowledge-base fix", key=f"k_{fid}"):
                db.update_feedback(fid, "kb_fix", note="Fix the FAQ in the knowledge base, then rebuild the index.")
                set_flash(f"Feedback #{fid} marked for a knowledge-base fix.")
                st.rerun()
            if c3.button("Dismiss", key=f"x_{fid}"):
                db.update_feedback(fid, "dismissed", note="Dismissed by reviewer")
                st.rerun()

            draft = ss.get(draft_key)
            if draft and draft["kind"] == "kb_fix":
                st.info(f"The LLM thinks this is a knowledge-base problem, not a behaviour one: {draft['text']}")
            if draft and not draft["used_llm"]:
                st.caption("LLM offline: the box is prefilled with the remark. Rewrite it as an instruction.")

            with st.form(f"rule_from_fb_{fid}"):
                text = st.text_area(
                    "Rule text (an instruction to the bot)",
                    value=draft["text"] if draft and draft["kind"] == "rule" else "",
                    placeholder="e.g. For policy questions, state only what the FAQ says; do not describe "
                    "decision-makers or processes it doesn't mention.",
                )
                s1, s2 = st.columns(2)
                scope = s1.selectbox("Applies to", categories, index=0,
                                     help=f"'all' or one FAQ category (this answer used: {fb['faq_category']})")
                reason = s2.text_input("Why", value=f"Feedback #{fid}: {(fb['remark'] or '')[:150]}")
                b1, b2 = st.columns(2)
                as_pending = b1.form_submit_button("Save as pending")
                as_active = b2.form_submit_button("Approve and activate now")
                if as_pending or as_active:
                    if not text.strip():
                        st.warning("Write the rule first.")
                    else:
                        status = "active" if as_active else "pending"
                        rid = db.add_rule(text, status=status, source="feedback", reason=reason, scope=scope,
                                          feedback_id=fid, created_by=tester or "reviewer")
                        db.update_feedback(fid, "rule_created" if as_active else "rule_drafted",
                                           note=f"Rule #{rid} ({status})")
                        ss.pop(draft_key, None)
                        set_flash(f"Rule #{rid} saved as {status}.")
                        st.rerun()


# ---------------------------------------------------------------- rules page
def page_rules() -> None:
    st.title("📏 Rules")
    st.caption(
        "Active rules are sent to the LLM with every answer (scope *all*) or with answers from one FAQ "
        "category. Edit or switch them off any time; every change is kept in the history."
    )
    categories = ["all"] + assistant.categories()

    with st.expander("➕ Add a rule"):
        with st.form("new_rule", clear_on_submit=True):
            text = st.text_area("Rule text")
            c1, c2 = st.columns(2)
            scope = c1.selectbox("Applies to", categories)
            reason = c2.text_input("Why")
            if st.form_submit_button("Add as active rule"):
                if text.strip():
                    rid = db.add_rule(text, status="active", source="manual", reason=reason, scope=scope,
                                      created_by=tester or "admin")
                    set_flash(f"Rule #{rid} added.")
                    st.rerun()
                else:
                    st.warning("Write the rule first.")

    pending = db.list_rules(("pending",))
    st.subheader(f"Waiting for approval ({len(pending)})")
    if not pending:
        st.caption("No pending rules.")
    for rule in pending:
        with st.container(border=True):
            key = f"{rule['id']}_{rule['updated_at']}"
            text = st.text_area(f"Rule #{rule['id']} · from {rule['source']}", value=rule["text"], key=f"pt_{key}")
            st.caption(f"Why: {rule['reason'] or '–'}")
            c1, c2, c3 = st.columns([2, 1, 1])
            scope = c1.selectbox("Applies to", categories, key=f"ps_{key}",
                                 index=categories.index(rule["scope"]) if rule["scope"] in categories else 0)
            if c2.button("✅ Approve", key=f"pa_{key}"):
                db.update_rule(rule["id"], changed_by=tester or "admin", text=text, scope=scope, status="active")
                if rule["feedback_id"]:
                    db.update_feedback(rule["feedback_id"], "rule_created", note=f"Rule #{rule['id']} approved")
                set_flash(f"Rule #{rule['id']} is now active.")
                st.rerun()
            if c3.button("❌ Reject", key=f"pr_{key}"):
                db.update_rule(rule["id"], changed_by=tester or "admin", status="rejected")
                if rule["feedback_id"]:
                    db.update_feedback(rule["feedback_id"], "new", note=f"Drafted rule #{rule['id']} was rejected")
                st.rerun()

    current = db.list_rules(("active", "inactive"))
    st.subheader(f"Rules in use ({sum(r['status'] == 'active' for r in current)} active)")
    for rule in current:
        key = f"{rule['id']}_{rule['updated_at']}"
        icon = "🟢" if rule["status"] == "active" else "⚪"
        with st.container(border=True):
            st.markdown(f"{icon} **#{rule['id']}** {rule['text']}")
            st.caption(
                f"scope: {rule['scope']} · source: {rule['source']} · updated {rule['updated_at']}"
                + (f" · from feedback #{rule['feedback_id']}" if rule["feedback_id"] else "")
            )
            if rule["reason"]:
                st.caption(f"Why: {rule['reason']}")
            with st.expander("Edit"):
                with st.form(f"edit_{key}"):
                    new_text = st.text_area("Rule text", value=rule["text"])
                    c1, c2 = st.columns(2)
                    new_scope = c1.selectbox("Applies to", categories,
                                             index=categories.index(rule["scope"]) if rule["scope"] in categories else 0)
                    new_status = c2.selectbox("Status", ["active", "inactive"],
                                              index=0 if rule["status"] == "active" else 1)
                    new_reason = st.text_input("Why", value=rule["reason"] or "")
                    if st.form_submit_button("Save changes"):
                        db.update_rule(rule["id"], changed_by=tester or "admin", text=new_text,
                                       scope=new_scope, status=new_status, reason=new_reason)
                        set_flash(f"Rule #{rule['id']} updated.")
                        st.rerun()
                confirm = st.checkbox("I want to delete this rule permanently", key=f"dc_{key}")
                if st.button("Delete", key=f"del_{key}", disabled=not confirm):
                    db.delete_rule(rule["id"])
                    set_flash(f"Rule #{rule['id']} deleted.")
                    st.rerun()
            with st.expander("History"):
                hist = pd.DataFrame(db.rule_history(rule["id"]))
                if not hist.empty:
                    st.dataframe(hist[["changed_at", "changed_by", "action", "new_status", "new_text"]],
                                 hide_index=True)

    rejected = db.list_rules(("rejected",))
    if rejected:
        with st.expander(f"Rejected rules ({len(rejected)})"):
            for rule in rejected:
                st.markdown(f"- #{rule['id']} {rule['text']}")


# ------------------------------------------------------- knowledge base page
def page_knowledge_base() -> None:
    st.title("📚 Knowledge base")
    kb = assistant.kb
    manifest = assistant.index.manifest if assistant.index else {}
    c1, c2, c3 = st.columns(3)
    c1.metric("FAQs", len(kb))
    c2.metric("Student-visible", int((kb["role"] == "student").sum()))
    c3.metric("Admin-only", int((kb["role"] == "admin").sum()))
    st.caption(
        f"File: `{settings.knowledge_base_path}` · index built {manifest.get('built_at', '?')} · "
        f"backend {manifest.get('backend', '?')} · model {manifest.get('embedding_model', '?')} · "
        f"fingerprint {manifest.get('kb_fingerprint', '?')}"
    )

    f1, f2, f3 = st.columns([1, 2, 2])
    roles = f1.multiselect("Role", sorted(kb["role"].unique()))
    cats = f2.multiselect("Category", assistant.categories())
    search = f3.text_input("Search text")
    view = kb
    if roles:
        view = view[view["role"].isin(roles)]
    if cats:
        view = view[view["category"].isin(cats)]
    if search:
        mask = view["question"].str.contains(search, case=False, regex=False) | view["answer"].str.contains(
            search, case=False, regex=False)
        view = view[mask]
    st.dataframe(view, hide_index=True)

    st.markdown(
        "**To change FAQs:** edit the CSV in VS Code or Excel (keep the five columns; in Excel use "
        "*Save As → CSV UTF-8*), save it, then click below. The index also rebuilds automatically on the "
        "next start whenever the file has changed."
    )
    if st.button("🔄 Reload knowledge base and rebuild index"):
        try:
            with st.spinner("Re-embedding FAQs..."):
                assistant.load_knowledge_base(force_rebuild=True)
            set_flash(f"Reloaded {len(assistant.kb)} FAQs and rebuilt the index.")
            st.rerun()
        except KnowledgeBaseError as exc:
            st.error(f"```\n{exc}\n```")


# ------------------------------------------------------------- activity page
def page_activity() -> None:
    st.title("📊 Activity")
    s = db.stats()
    cols = st.columns(6)
    cols[0].metric("Questions", s["questions"])
    cols[1].metric("Answered", s["answered"])
    cols[2].metric("Not answered", s["no_match"])
    cols[3].metric("Feedback", s["feedback"])
    cols[4].metric("Avg rating", "–" if s["avg_rating"] is None else f"{s['avg_rating']:.2f} / 5")
    cols[5].metric("Active rules", s["active_rules"])

    conversations = pd.DataFrame(db.recent_conversations(500))
    if conversations.empty:
        st.info("No questions asked yet.")
        return
    st.subheader("Recent questions")
    st.dataframe(
        conversations[["id", "created_at", "tester", "user_role", "question", "status", "answer_mode",
                       "faq_id", "similarity", "rerank_score", "latency_ms", "answer"]],
        hide_index=True,
    )
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Conversations (CSV)", conversations.to_csv(index=False).encode("utf-8-sig"),
                       "conversations.csv", "text/csv")
    feedback = pd.DataFrame(db.list_feedback())
    d2.download_button("⬇️ Feedback (CSV)", feedback.to_csv(index=False).encode("utf-8-sig"),
                       "feedback.csv", "text/csv", disabled=feedback.empty)
    rules = pd.DataFrame(db.list_rules())
    d3.download_button("⬇️ Rules (CSV)", rules.to_csv(index=False).encode("utf-8-sig"),
                       "rules.csv", "text/csv", disabled=rules.empty)


{
    PAGES[0]: page_chat,
    PAGES[1]: page_feedback,
    PAGES[2]: page_rules,
    PAGES[3]: page_knowledge_base,
    PAGES[4]: page_activity,
}[page]()
