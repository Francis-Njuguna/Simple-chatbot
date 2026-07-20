"""LLM generation service — Anthropic (primary) | OpenAI | Ollama.

Performance notes
-----------------
* The underlying LangChain chat client (and its HTTP connection pool) is built
  **once** and reused for the life of the process via ``get_llm_service`` — it
  is no longer reconstructed on every request.
* ``max_tokens`` is configurable (default reduced to 1024) so Claude does not
  spend time generating far more tokens than a help-desk answer needs.
"""

from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.config import get_settings
from backend.app.prompts.templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Configurable LLM provider for answer generation.

    Provider priority (set via LLM_PROVIDER env var):
        anthropic  → haiku-4-5 / Sonnet  (default)
        openai     → GPT-4o (fallback)
        ollama     → local Llama / Mistral etc.  (offline fallback)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = self._build_llm()

    def _build_llm(self) -> Any:
        provider = self.settings.llm_provider
        max_tokens = self.settings.llm_max_tokens

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # lazy import

            logger.info("Using Anthropic model: %s", self.settings.anthropic_model)
            return ChatAnthropic(
                model=self.settings.anthropic_model,
                api_key=self.settings.anthropic_api_key,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=self.settings.llm_timeout,
                max_retries=self.settings.llm_max_retries,
            )

        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama  # lazy import

            logger.info("Using Ollama model: %s", self.settings.ollama_model)
            return ChatOllama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=0.0,
            )

        # default / "openai"
        from langchain_openai import ChatOpenAI  # lazy import

        logger.info("Using OpenAI model: %s", self.settings.openai_model)
        return ChatOpenAI(
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
        )

    def _build_messages(
        self, question: str, context: str, history: str
    ) -> list[Any]:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            history=history,
            question=question,
        )
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        # Anthropic can return a list of content blocks — extract text
        if isinstance(content, list):
            return " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        return str(content).strip()

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
    ) -> str:
        messages = self._build_messages(question, context, history)
        try:
            response = await self._llm.ainvoke(messages)
            return self._extract_text(response.content)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return (
                "I could not find that information in the Amref Help Desk knowledge base."
            )

    async def stream_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
    ):
        """Yield answer chunks as they arrive (for streaming responses)."""
        messages = self._build_messages(question, context, history)
        try:
            async for chunk in self._llm.astream(messages):
                text = self._extract_text(chunk.content)
                if text:
                    yield text
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc)
            yield (
                "I could not find that information in the Amref Help Desk knowledge base."
            )


# ---------------------------------------------------------------------------
# Process-wide singleton — the chat client / HTTP pool is built once.
# ---------------------------------------------------------------------------

@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
