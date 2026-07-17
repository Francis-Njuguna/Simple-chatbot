"""TLS / SSL helpers for outbound HTTP clients.

Why this exists
---------------
Hitting ``helpdesk.amref.ac.ke`` raises:

    [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate

We PROVED (via hardcoded ``verify=certifi.where()`` + diagnostics) that this
is NOT a client-configuration problem: certifi is loaded, ``SSL_CERT_FILE``
points at it, and verification still fails. The cause is server-side — the
host serves an **incomplete certificate chain** (leaf only, missing the
intermediate CA). certifi ships trusted *roots*, not this site's specific
*intermediate*, so no static bundle can bridge the gap.

Fix strategy (``build_kb_ssl_context``)
---------------------------------------
1. If ``KB_VERIFY_SSL=false`` → return ``False`` (skip verification entirely).
2. Start from certifi's roots (+ VERIFY_X509_PARTIAL_CHAIN so any cert in the
   store may act as a trust anchor).
3. If ``KB_CA_BUNDLE`` is set → load it (a PEM containing the intermediate).
4. Otherwise, **auto-recover**: open a raw, unverified TLS connection to the
   host, capture whatever certificates it presents (leaf + any intermediates),
   and load them into the verifying context. Verification then stays ON while
   the missing link is supplied at runtime.
"""

import socket
import ssl
from typing import Union
from urllib.parse import urlsplit

import certifi

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


def _host_port(url: str, default_port: int = 443) -> tuple[str, int]:
    parts = urlsplit(url if "://" in url else f"https://{url}")
    return (parts.hostname or url), (parts.port or default_port)


def _capture_server_chain_pem(host: str, port: int) -> list[str]:
    """Return PEM certs the server presents, WITHOUT verifying them.

    Grabs the intermediate the server forgot to send so it can be added to a
    *verifying* context afterwards. Returns PEM strings (deduplicated).
    """
    sniff_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    sniff_ctx.check_hostname = False
    sniff_ctx.verify_mode = ssl.CERT_NONE

    pems: list[str] = []
    with socket.create_connection((host, port), timeout=15) as sock:
        with sniff_ctx.wrap_socket(sock, server_hostname=host) as tls:
            # Full chain — available on Python 3.10+ (OpenSSL 1.1.1+).
            get_chain = getattr(tls, "get_unverified_chain", None)
            if get_chain is not None:
                try:
                    for cert in get_chain():
                        pem = ssl.DER_cert_to_PEM_cert(cert.public_bytes(_DER))
                        if pem not in pems:
                            pems.append(pem)
                except Exception:  # pragma: no cover - best effort
                    pems = []

            # Fallback: at least the leaf certificate.
            if not pems:
                der_leaf = tls.getpeercert(binary_form=True)
                if der_leaf:
                    pems.append(ssl.DER_cert_to_PEM_cert(der_leaf))

    return pems


# ssl._ssl.Certificate.public_bytes() wants an ssl.Encoding-like value.
# ssl.DER is exposed on 3.10+; guard for older builds.
_DER = getattr(ssl, "DER", 1)


def build_kb_ssl_context() -> Union[ssl.SSLContext, bool]:
    """Build the ``verify`` value for the KB crawler's httpx client."""
    settings = get_settings()

    if not settings.kb_verify_ssl:
        logger.warning("KB_VERIFY_SSL=false — crawler TLS verification DISABLED (insecure).")
        return False

    context = ssl.create_default_context(cafile=certifi.where())
    try:
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    except AttributeError:  # pragma: no cover - older Python
        pass

    # Explicit operator-provided bundle wins.
    if settings.kb_ca_bundle:
        try:
            context.load_verify_locations(cafile=settings.kb_ca_bundle)
            logger.info("Loaded KB_CA_BUNDLE: %s", settings.kb_ca_bundle)
            return context
        except (FileNotFoundError, ssl.SSLError, OSError) as exc:
            logger.error("Failed to load KB_CA_BUNDLE (%s): %s", settings.kb_ca_bundle, exc)

    # Auto-recover the missing intermediate straight from the server.
    host, port = _host_port(settings.kb_base_url)
    try:
        pems = _capture_server_chain_pem(host, port)
        loaded = 0
        for pem in pems:
            try:
                context.load_verify_locations(cadata=pem)
                loaded += 1
            except ssl.SSLError:
                continue
        if loaded:
            logger.warning(
                "Auto-loaded %d certificate(s) from %s:%d to complete the "
                "server's incomplete chain (missing intermediate). TLS "
                "verification remains ENABLED.",
                loaded,
                host,
                port,
            )
        else:
            logger.error(
                "Could not capture certificates from %s:%d for chain "
                "completion. Provide KB_CA_BUNDLE or set KB_VERIFY_SSL=false.",
                host,
                port,
            )
    except Exception as exc:  # broad: sniffing must never crash ingestion
        logger.error(
            "TLS chain auto-recovery against %s:%d failed: %r — provide "
            "KB_CA_BUNDLE or set KB_VERIFY_SSL=false.",
            host,
            port,
            exc,
        )

    return context
