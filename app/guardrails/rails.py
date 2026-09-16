import re

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails: LLMRails | None = None

_OFF_TOPIC_MESSAGES = frozenset(
    {
        "tell me a joke",
        "what is the capital of france",
        "write me a poem",
        "what is 2 plus 2",
        "what should i eat for dinner",
        "who won the game yesterday",
        "recommend a movie",
        "what is the weather today",
        "can you help me with math homework",
        "tell me about world history",
        "what is the best restaurant near me",
    }
)

_OFF_TOPIC_RESPONSE = (
    "I'm an Enterprise IT Assistant focussed on Kubernetes, Intel hardware "
    "and networking. I can't help with that - but ask me anything technical!"
)


def _normalize_message(message: str) -> str:
    return re.sub(r"[^\w\s]", "", message.casefold()).strip()


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llma-3.1-8b-instant for fast intent classification at the gate
    """
    global _rails
    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY, model=settings.GROQ_GUARD_MODEL, temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("NeMo Guardrails initialized")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate
    Returns:
    (True,rail_response) -  a rail fired;return this response immediately,skip the RAG pipeline entirely
    (False,None) - message is clean;proceed to LangGraph
    """
    normalized_message = _normalize_message(message)
    if normalized_message in _OFF_TOPIC_MESSAGES:
        logfire.info(f"Deterministic off-topic rail fired | query='{message[:80]}'")
        return True, _OFF_TOPIC_RESPONSE

    if _rails is None:
        logfire.warning("Guardrails not initialized - skipping gate")
        return False, None
    with logfire.span("Guardrails check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        normalized_content = content.casefold()
        fired = any(
            indicator.casefold() in normalized_content
            for indicator in RAIL_INDICATORS
        )

        if fired:
            logfire.info(f"Guardrails fired | query='{message[:80]}'")
            return True, content
        logfire.info("Guardrails passed.")
        return False, None
