import re

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def _strip_think_block(text: str) -> str:
    """Remove internal reasoning blocks before returning or evaluating the content."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)


def _normalize_guardrail_text(text: str) -> str:
    """Normalize model output so refusal checks are less brittle."""
    cleaned = _strip_think_block(text)
    if not cleaned:
        return ""

    cleaned = cleaned.replace("’", "'").replace("“", '"').replace("”", '"')
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _did_guardrail_fire(content: str) -> bool:
    """Return True when the generated text is a refusal / guardrail response."""
    normalized = _normalize_guardrail_text(content)
    if not normalized:
        return False

    direct_indicators = (indicator.lower() for indicator in RAIL_INDICATORS)
    if any(indicator in normalized for indicator in direct_indicators):
        return True

    broader_refusal_phrases = (
        "can't help with that",
        "cannot help with that",
        "can't assist with that",
        "cannot assist with that",
        "not in scope",
        "out of scope",
        "not allowed",
        "i'm sorry, but i can't",
        "i am sorry, but i cannot",
        "i can't help with that",
        "i cannot help with that",
        "refuse",
        "policy",
        "not about the allowed topics",
        "i am unable to assist",
    )
    return any(phrase in normalized for phrase in broader_refusal_phrases)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-safeguard-20b",
        temperature=0,
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])
        logfire.info(f"Guardrails result: {result}")
        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        clean_content = _strip_think_block(content)

        fired = _did_guardrail_fire(clean_content)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, clean_content

        logfire.info("✅ Guardrails passed.")
        return False, None
