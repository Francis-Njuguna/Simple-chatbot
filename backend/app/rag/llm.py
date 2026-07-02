"""LLM generation service — Anthropic (primary) | OpenAI | Ollama."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.config import get_settings
from backend.app.prompts.templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Configurable LLM provider for answer generation.

    Provider priority (set via LLM_PROVIDER env var):
        anthropic  → haiku-4-5/ Sonnet  (default)
        openai     → GPT-4o (fallback)
        ollama     → local Llama / Mistral etc.  (offline fallback)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = self._build_llm()

    def _build_llm(self) -> Any:
        provider = self.settings.llm_provider

        if provider == "anthropic":
            # langchain-anthropic ships ChatAnthropic
            from langchain_anthropic import ChatAnthropic  # lazy import

            logger.info(
                "Using Anthropic model: %s", self.settings.anthropic_model
            )
            return ChatAnthropic(
                model=self.settings.anthropic_model,
                api_key=self.settings.anthropic_api_key,
                temperature=0.0,
                max_tokens=2048,
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
        )

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
    ) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            history=history,
            question=question,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            content = response.content
            if isinstance(content, str):
                return content.strip()
            # Anthropic can return a list of content blocks — extract text
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            return str(content).strip()
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return (
                "I could not find that information in the Amref Help Desk knowledge base."
            )
