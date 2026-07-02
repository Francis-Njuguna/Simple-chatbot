"""Shared SlowAPI rate-limiter instance.

Why a dedicated module?
-----------------------
``slowapi``'s ``Limiter`` must be attached to ``app.state`` in ``main.py``
**and** referenced by individual route handlers via the ``@limiter.limit``
decorator.  If the limiter were defined inside ``main.py``, any route module
that imports it would create a circular dependency:

    main.py  →  routes/chat.py  →  main.py   # ImportError!

By placing the limiter here – in a leaf module that imports nothing from the
rest of the application – both ``main.py`` and every route module can import
it without any cycle.

Usage
-----
In route handlers::

    from backend.app.core.limiter import limiter

    @router.post("/chat")
    @limiter.limit("20/minute")
    async def chat(request: Request, ...): ...

In ``main.py``::

    from backend.app.core.limiter import limiter

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.config import get_settings

# ---------------------------------------------------------------------------
# Build the limiter exactly once.
# ``get_settings()`` is cached via ``@lru_cache``, so this is effectively
# free after the first call.
# ---------------------------------------------------------------------------
_settings = get_settings()

#: Module-level singleton consumed by ``main.py`` and all route decorators.
limiter: Limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_settings.rate_limit],
)
