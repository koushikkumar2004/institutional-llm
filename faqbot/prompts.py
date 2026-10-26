"""Prompts sent to the LLM. Rules from the database are inserted into the answer prompt."""

from __future__ import annotations

NO_MATCH_MESSAGE = "I couldn't find that information in the available FAQs."
NOT_IN_FAQ_MESSAGE = "The available FAQs do not contain that specific information."
KB_FIX_PREFIX = "KB_FIX:"


def answer_messages(
    tenant_name: str, question: str, faq_question: str, faq_answer: str, rules: list[str]
) -> list[dict[str, str]]:
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, start=1)) or "(no extra rules)"
    system = (
        f"You are the FAQ assistant for {tenant_name}.\n"
        "You answer the user's question using ONLY the single FAQ entry you are given. "
        "You may rephrase it, but you must not change its meaning or add information.\n"
        "If the FAQ entry does not answer the user's question, reply with exactly this "
        f"sentence and nothing else:\n{NOT_IN_FAQ_MESSAGE}\n\n"
        f"Rules you must follow:\n{rules_block}"
    )
    user = (
        "FAQ entry\n"
        f"Question: {faq_question}\n"
        f"Answer: {faq_answer}\n\n"
        f"User question: {question}\n\n"
        f"Follow every one of these rules in your answer:\n{rules_block}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rule_draft_messages(
    question: str, bot_answer: str, faq_question: str | None, faq_answer: str | None,
    verdict: str, remark: str,
) -> list[dict[str, str]]:
    system = (
        "You help maintain a university FAQ chatbot. A human reviewer rated one of its answers "
        "and wrote a remark. Turn the remark into ONE general behaviour rule for the chatbot.\n"
        "Requirements:\n"
        "- One sentence, written as an instruction to the chatbot (e.g. 'Do not ...', 'Always ...').\n"
        "- General enough to apply to future questions, not only this one.\n"
        "- Do not put facts (numbers, dates, names, fees) into the rule.\n"
        "- If the remark praises the answer, write a rule that keeps the praised behaviour.\n"
        f"- If the remark says the FAQ data itself is wrong or missing, reply exactly '{KB_FIX_PREFIX} ' "
        "followed by a short note of what must change in the knowledge base.\n"
        "Reply with the rule only."
    )
    faq_part = (
        f"FAQ used: {faq_question}\nFAQ answer: {faq_answer}" if faq_question else "FAQ used: none (no match)"
    )
    user = (
        f"User question: {question}\n"
        f"{faq_part}\n"
        f"Chatbot answer: {bot_answer}\n"
        f"Reviewer verdict: {verdict}\n"
        f"Reviewer remark: {remark}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
