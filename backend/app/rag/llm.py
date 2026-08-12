"""LLM generation service — OpenAI-compatible (primary) | Anthropic | Ollama.

Performance notes
-----------------
* The underlying LangChain chat client (and its HTTP connection pool) is built
  **once** and reused for the life of the process via ``get_llm_service`` — it
  is no longer reconstructed on every request.
* ``max_tokens`` is configurable (default 2048) so the LLM has room for a
  complete numbered procedure plus caveats without truncating mid-answer.
"""


import time
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

        
 
        if provider == "agentrouter":
            # AgentRouter fronts Anthropic Claude with an OpenAI-compatible
            # /v1/chat/completions endpoint, so ChatOpenAI is the client — no
            # Anthropic SDK involved.
            from langchain_openai import ChatOpenAI  # lazy import

            api_key = self.settings.agentrouter_api_key
            if not api_key:
                raise RuntimeError(
                    "AGENTROUTER_API_KEY is not set. Set it (or ANTHROPIC_AUTH_KEY) "
                    "to the Anthropic key issued for AgentRouter."
                )
            if "your-" in api_key.lower() or "-here" in api_key.lower():
                raise RuntimeError(
                    "AGENTROUTER_API_KEY is still a placeholder. Set a real key."
                )
            base_url = self.settings.agentrouter_base_url
            if not base_url:
                raise RuntimeError(
                    "AGENTROUTER_BASE_URL is empty. Set the OpenAI-compatible base "
                    "URL, e.g. https://agentrouter.org/v1"
                )

            kwargs: dict[str, Any] = {}
            # See config.agentrouter_user_agent: the gateway 401s any client it
            # does not recognise, so the user-agent is part of the contract.
            if self.settings.agentrouter_user_agent:
                kwargs["default_headers"] = {
                    "User-Agent": self.settings.agentrouter_user_agent
                }

            # AgentRouter emits `data: null` SSE frames that crash the langchain
            # streaming adapter. See rag.sse_repair for the frame dump and why
            # the repair belongs at the transport layer.
            from backend.app.rag.sse_repair import build_repaired_async_client

            kwargs["http_async_client"] = build_repaired_async_client(
                timeout=self.settings.llm_timeout,
                max_connections=self.settings.llm_max_connections,
                max_keepalive=self.settings.llm_max_keepalive_connections,
            )

            logger.info(
                "Using AgentRouter model %s via %s",
                self.settings.agentrouter_model,
                base_url,
            )
            return ChatOpenAI(
                model=self.settings.agentrouter_model,
                api_key=api_key,
                base_url=base_url,
                # None => the parameter is not sent at all. See config.
                temperature=self.settings.agentrouter_temperature,
                max_tokens=max_tokens,
                timeout=self.settings.llm_timeout,
                max_retries=self.settings.llm_max_retries,
                **kwargs,
            )

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
                temperature=0.5,
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
        return self.settings.llm_provider in {"openai", "agentrouter", "gemini", "anthropic"}
 
    @staticmethod
    def _extract_text(content: Any, *, strip: bool = True) -> str:
        """Pull plain text out of whatever shape the provider returned.

        ``strip`` is False when consuming a token stream. Stripping a whole
        answer is right; stripping each delta is not, because the space between
        two words routinely arrives at the head of a chunk — the frames
        ``" c"`` and ``"an re"`` are ``" can re"``, and stripping them yields
        ``"can re"``, silently deleting the word break.
        """
        def _clean(value: str) -> str:
            return value.strip() if strip else value

        if content is None:
            return ""
        if isinstance(content, str):
            return _clean(content)
        if hasattr(content, "content"):
            return LLMService._extract_text(content.content, strip=strip)
        if hasattr(content, "text"):
            return _clean(str(content.text))
        if hasattr(content, "generations"):
            return LLMService._extract_text(getattr(content, "generations"), strip=strip)
        if isinstance(content, list):
            parts = [
                LLMService._extract_text(block, strip=strip)
                if not isinstance(block, dict)
                else block.get("text", "")
                for block in content
            ]
            # Streaming concatenates: a chunk's blocks are consecutive slices of
            # one answer, so any separator would insert text the model did not
            # emit. A complete response's blocks are separate spans, which is
            # why the non-streaming path keeps the space.
            return _clean(("" if not strip else " ").join(parts))
        return _clean(str(content))
 
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
            if provider == "agentrouter":
                # AgentRouter returns 401 for two very different reasons and the
                # log is the only place the difference is visible: a bad key, or
                # "unauthorized client detected" when it does not recognise the
                # HTTP client (see settings.agentrouter_user_agent).
                detail += (
                    " (a 401 from AgentRouter means either an invalid "
                    "AGENTROUTER_API_KEY or a rejected client user-agent)"
                )
        elif "NotFound" in name:
            detail = (
                f"the {provider} endpoint does not recognise the configured model "
                f"({self._configured_model()})"
            )
        elif "BadRequest" in name and provider == "agentrouter":
            # Claude models served through AgentRouter reject `temperature` with
            # 400 "`temperature` is deprecated for this model", and the router
            # fans out across backends so the same model may accept it on one
            # request and reject it on the next.
            detail = (
                "the AgentRouter endpoint rejected the request (400). If the log "
                "mentions `temperature`, set AGENTROUTER_TEMPERATURE to empty in "
                ".env so the parameter is not sent"
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

    def _configured_model(self) -> str:
        """The model name for the active provider, for diagnostics."""
        provider = self.settings.llm_provider
        return {
            "agentrouter": self.settings.agentrouter_model,
            "anthropic": self.settings.anthropic_model,
            "openai": self.settings.openai_model,
            "gemini": self.settings.gemini_model,
            "ollama": self.settings.ollama_model,
        }.get(provider, "see config")

    def describe(self) -> dict[str, Any]:
        """The request parameters actually in effect, for diagnostics.

        Read off the constructed client rather than off settings, so what is
        reported is what will really be sent — a provider branch that ignores or
        overrides a setting cannot hide behind this.
        """
        llm = self._llm
        endpoint = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
        info: dict[str, Any] = {
            "provider": self.settings.llm_provider,
            "model": self._configured_model(),
            "endpoint": str(endpoint) if endpoint else "provider default",
            "max_tokens": getattr(llm, "max_tokens", None),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "request_timeout", None) or getattr(llm, "timeout", None),
            "max_retries": getattr(llm, "max_retries", None),
            "streaming": getattr(llm, "streaming", None),
        }
        client = getattr(llm, "root_async_client", None)
        http_client = getattr(client, "_client", None) if client is not None else None
        limits = getattr(getattr(http_client, "_transport", None), "_pool", None)
        if limits is not None:
            info["http_max_connections"] = getattr(limits, "_max_connections", None)
            info["http_max_keepalive"] = getattr(limits, "_max_keepalive_connections", None)
        return info

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        """Input/output token counts, when the provider reports them."""
        meta = getattr(response, "usage_metadata", None) or {}
        if not meta:
            raw = getattr(response, "response_metadata", {}) or {}
            meta = raw.get("token_usage") or raw.get("usage") or {}
        if not meta:
            return {}
        return {
            "input_tokens": meta.get("input_tokens") or meta.get("prompt_tokens") or 0,
            "output_tokens": meta.get("output_tokens") or meta.get("completion_tokens") or 0,
        }

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "No prior conversation.",
        images: str = NO_IMAGES_NOTE,
        stats: dict[str, Any] | None = None,
    ) -> str:
        """Generate an answer. Exceptions are returned as text, not raised.

        ``stats``, when supplied, is filled in with prompt size, token usage and
        call duration. It is a caller-owned dict on purpose: this service is a
        process-wide singleton, so recording per-request numbers on ``self``
        would hand every concurrent request whichever value finished last.
        """
        if self._use_chat_model():
            prompt = self._build_messages(question, context, history, images)
        else:
            prompt = self._build_prompt(question, context, history, images)

        if stats is not None:
            stats["prompt_chars"] = (
                sum(len(str(m.content)) for m in prompt)
                if isinstance(prompt, list)
                else len(prompt)
            )

        started = time.perf_counter()
        try:
            response = await self._llm.ainvoke(prompt)
            if stats is not None:
                stats["llm_call_ms"] = (time.perf_counter() - started) * 1000.0
                stats.update(self._usage(response))
                stats["ok"] = True
            return self._extract_text(response)
        except Exception as exc:
            if stats is not None:
                stats["llm_call_ms"] = (time.perf_counter() - started) * 1000.0
                stats["ok"] = False
                stats["error"] = type(exc).__name__
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
                # strip=False: leading/trailing spaces inside a delta are part
                # of the answer. See _extract_text.
                text = self._extract_text(chunk, strip=False)
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
