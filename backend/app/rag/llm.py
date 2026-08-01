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
from backend.app.prompts.templates import NO_IMAGES_NOTE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Configurable LLM provider for answer generation.

    Provider priority (set via LLM_PROVIDER env var):
        grok ->
        openai     → Qwen/OpenAI (default)
        gemini     → Google Gemini (fast & capable)
        anthropic  → haiku-4-5 / Sonnet  (optional fallback)
        ollama     → local Llama / Mistral etc.  (offline fallback)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = self._build_llm()

    def _build_llm(self) -> Any:
        provider = self.settings.llm_provider
        max_tokens = self.settings.llm_max_tokens

        
 
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

            if not self.settings.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Set a valid Google Gemini API key."
                )

            logger.info("Using Google Gemini model: %s", self.settings.gemini_model)
            return ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                api_key=self.settings.gemini_api_key,
                temperature=0.0,
                max_output_tokens=max_tokens,
                timeout=self.settings.llm_timeout,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # lazy import

            api_key = self.settings.anthropic_auth_key
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_AUTH_KEY is not set. Set a valid Anthropic auth key."
                )

            base_url = self.settings.anthropic_base_url
            logger.info(
                "Using Anthropic model: %s (base_url=%s)",
                self.settings.anthropic_model,
                base_url or "default (api.anthropic.com)",
            )
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            return ChatAnthropic(
                model=self.settings.anthropic_model,
                api_key=api_key,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=self.settings.llm_timeout,
                max_retries=self.settings.llm_max_retries,
                **kwargs,
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
        self, question: str, context: str, history: str, images: str
    ) -> list[Any]:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            images=images,
            history=history,
            question=question,
        )
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

    def _build_prompt(self, question: str, context: str, history: str, images: str) -> str:
        return "\n\n".join(
            [SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(
                context=context,
                images=images,
                history=history,
                question=question,
            )]
        )
 
    def _use_chat_model(self) -> bool:
        return self.settings.llm_provider in {"openai", "gemini", "anthropic"}
 
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
 
    def _error_message(self, exc: Exception) -> str:
        """Return a user-facing message describing an LLM *failure*.

        Previously any exception here returned "I could not find that
        information in the knowledge base", which made an auth error,
        a timeout and a genuine knowledge-base miss indistinguishable —
        both to users and to anyone reading the analytics logs.  Retrieval
        may have worked perfectly; only the generation call failed.
        """
        provider = self.settings.llm_provider
        name = type(exc).__name__

        if "Authentication" in name or "PermissionDenied" in name:
            detail = (
                f"the {provider} endpoint rejected the configured credentials"
            )
        elif "NotFound" in name:
            detail = (
                f"the {provider} endpoint does not recognise the configured model "
                f"({self.settings.anthropic_model if provider == 'anthropic' else 'see config'})"
            )
        elif "RateLimit" in name:
            detail = f"the {provider} endpoint is rate limiting requests"
        elif "Timeout" in name or "Connection" in name or "APIConnection" in name:
            detail = f"the {provider} endpoint could not be reached"
        else:
            detail = f"the {provider} request failed ({name})"

        return (
            "I found relevant knowledge-base material but could not generate an "
            f"answer: {detail}. This is a configuration or service problem, not a "
            "gap in the knowledge base — please check the server logs."
        )

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
        images: str = NO_IMAGES_NOTE,
    ) -> str:
        if self._use_chat_model():
            prompt = self._build_messages(question, context, history, images)
        else:
            prompt = self._build_prompt(question, context, history, images)

        try:
            response = await self._llm.ainvoke(prompt)
            return self._extract_text(response)
        except Exception as exc:
            logger.exception("LLM generation failed")
            return self._error_message(exc)

    async def complete(self, system: str, user: str) -> str:
        """Run a one-off prompt with no chat scaffolding.

        Unlike :meth:`generate_answer`, exceptions propagate — background jobs
        need to distinguish a failed call from a successful empty answer, and
        must not persist an error string as if it were content.
        """
        if self._use_chat_model():
            prompt: Any = [SystemMessage(content=system), HumanMessage(content=user)]
        else:
            prompt = f"{system}\n\n{user}"
        response = await self._llm.ainvoke(prompt)
        return self._extract_text(response)

    async def stream_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
        images: str = NO_IMAGES_NOTE,
    ):
        """Yield answer chunks as they arrive (for streaming responses)."""
        if self._use_chat_model():
            prompt = self._build_messages(question, context, history, images)
        else:
            prompt = self._build_prompt(question, context, history, images)
 
        try:
            async for chunk in self._llm.astream(prompt):
                text = self._extract_text(chunk)
                if text:
                    yield text
        except Exception as exc:
            logger.exception("LLM streaming failed")
            yield self._error_message(exc)


# ---------------------------------------------------------------------------
# Process-wide singleton — the chat client / HTTP pool is built once.
# ---------------------------------------------------------------------------

@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
