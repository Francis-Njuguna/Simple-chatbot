"""LLM generation service — OpenAI-compatible (primary) | Anthropic | Ollama.

Performance notes
-----------------
* The underlying LangChain chat client (and its HTTP connection pool) is built
  **once** and reused for the life of the process via ``get_llm_service`` — it
  is no longer reconstructed on every request.
* ``max_tokens`` is configurable (default reduced to 1024) so the LLM does not
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
        openai     → Qwen/OpenAI (default)
        anthropic  → haiku-4-5 / Sonnet  (optional fallback)
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
 
        if self.settings.openai_api_base and provider != "openai":
            logger.warning(
                "OPENAI_API_BASE is set but LLM_PROVIDER=%s. "
                "That endpoint will be ignored unless LLM_PROVIDER=openai.",
                provider,
            )
 
        if provider == "ollama":
            llm_class = None
            try:
                from langchain_ollama.llms import OllamaLLM  # lazy import
                llm_class = OllamaLLM
            except ImportError:
                try:
                    from langchain_community.llms import Ollama as OllamaLLM  # lazy import
                    llm_class = OllamaLLM
                except ImportError as exc:
                    raise ImportError(
                        "Ollama support is not available in the installed packages. "
                        "Install `langchain-ollama` or a compatible `langchain_community` version, or "
                        "use LLM_PROVIDER=openai with OPENAI_API_BASE for Qwen/OpenAI-compatible endpoints. "
                        f"Original import error: {exc}"
                    ) from exc
 
            logger.info("Using Ollama model: %s", self.settings.ollama_model)
            return llm_class(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=0.0,
                num_predict=max_tokens,
                timeout=self.settings.ollama_timeout,
                keep_alive=self.settings.ollama_keep_alive,
            )
 
        # default / "openai"
        from langchain_openai import ChatOpenAI  # lazy import
        import os

        # Basic validation for OpenAI-compatible configuration to avoid
        # silently using placeholder values and then returning the generic
        # fallback answer on every request.
        openai_api_key = self.settings.openai_api_key
        if self.settings.openai_api_base:
            if "<qwen-openai-compatible-endpoint>" in self.settings.openai_api_base:
                raise RuntimeError(
                    "OPENAI_API_BASE is a placeholder. "
                    "Set it to your Qwen/OpenAI-compatible endpoint URL."
                )
            os.environ.setdefault("OPENAI_API_BASE", self.settings.openai_api_base)
            logger.info("OpenAI API base overridden: %s", self.settings.openai_api_base)
            if not openai_api_key:
                logger.warning(
                    "OPENAI_API_KEY is not set. Using local OpenAI-compatible endpoint without authentication. "
                    "If your endpoint requires a key, set OPENAI_API_KEY accordingly."
                )
                openai_api_key = "unused"
 
        if not self.settings.openai_api_base:
            if not openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Set a valid OpenAI or Qwen API key."
                )
 
        if openai_api_key and (
            "your-qwen-key-here" in openai_api_key.lower()
            or "your-openai-key-here" in openai_api_key.lower()
        ):
            raise RuntimeError(
               "OPENAI_API_KEY is still a placeholder. "
               "Replace it with a real API key for Qwen or OpenAI."
            )
 
        logger.info("Using OpenAI model: %s", self.settings.openai_model)
        return ChatOpenAI(
            model=self.settings.openai_model,
            api_key=openai_api_key,
            base_url=self.settings.openai_api_base,
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
 
    def _build_prompt(self, question: str, context: str, history: str) -> str:
        return "\n\n".join(
            [SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(
                context=context,
                history=history,
                question=question,
            )]
        )
 
    def _use_chat_model(self) -> bool:
        return self.settings.llm_provider in {"openai", "anthropic"}
 
    @staticmethod
    def _extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if hasattr(content, "content"):
            return LLMService._extract_text(content.content)
        if hasattr(content, "text"):
            return str(content.text).strip()
        if hasattr(content, "generations"):
            return LLMService._extract_text(getattr(content, "generations"))
        if isinstance(content, list):
            return " ".join(
                LLMService._extract_text(block) if not isinstance(block, dict) else block.get("text", "")
                for block in content
            ).strip()
        return str(content).strip()
 
    async def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
    ) -> str:
        if self._use_chat_model():
            prompt = self._build_messages(question, context, history)
        else:
            prompt = self._build_prompt(question, context, history)
 
        try:
            response = await self._llm.ainvoke(prompt)
            return self._extract_text(response)
        except Exception:
            logger.exception("LLM generation failed")
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
        if self._use_chat_model():
            prompt = self._build_messages(question, context, history)
        else:
            prompt = self._build_prompt(question, context, history)
 
        try:
            async for chunk in self._llm.astream(prompt):
                text = self._extract_text(chunk)
                if text:
                    yield text
        except Exception:
            logger.exception("LLM streaming failed")
            yield (
                "I could not find that information in the Amref Help Desk knowledge base!."
            )


# ---------------------------------------------------------------------------
# Process-wide singleton — the chat client / HTTP pool is built once.
# ---------------------------------------------------------------------------

@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
