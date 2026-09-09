"""Smoke-test a deployed Amref Help Desk API.

Checks liveness, registration/login, and one real NVIDIA-backed chat request.
The script never prints credentials. Use ``--skip-auth`` for a liveness-only
check when PostgreSQL/auth is intentionally unavailable.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from dataclasses import dataclass

import httpx


@dataclass
class SmokeFailure(Exception):
    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


def _expect(response: httpx.Response, check: str, status: int) -> dict:
    if response.status_code != status:
        raise SmokeFailure(check, f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise SmokeFailure(check, "response was not JSON") from exc


def run(base_url: str, timeout: float, skip_auth: bool) -> None:
    root = base_url.rstrip("/")
    api = f"{root}/api/v1"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        started = time.monotonic()
        health = _expect(client.get(f"{root}/health"), "health", 200)
        if health.get("status") != "healthy":
            raise SmokeFailure("health", f"unexpected payload: {health}")
        print(f"PASS health ({time.monotonic() - started:.2f}s)")

        token = ""
        if not skip_auth:
            suffix = secrets.token_hex(6)
            email = f"smoke-{suffix}@example.invalid"
            password = secrets.token_urlsafe(18)
            register = client.post(
                f"{api}/auth/register",
                json={"email": email, "password": password, "full_name": "CI smoke test"},
            )
            if register.status_code == 200:
                token = _expect(register, "register", 200).get("access_token", "")
            elif register.status_code == 409:
                login = client.post(
                    f"{api}/auth/login", json={"email": email, "password": password}
                )
                token = _expect(login, "login", 200).get("access_token", "")
            else:
                raise SmokeFailure(
                    "register", f"HTTP {register.status_code}: {register.text[:300]}"
                )
            if not token:
                raise SmokeFailure("auth", "no access token returned")
            print("PASS auth")

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        started = time.monotonic()
        response = client.post(
            f"{api}/chat",
            headers=headers,
            json={"message": "How do I reset my student portal password?"},
        )
        payload = _expect(response, "chat", 200)
        answer = payload.get("answer", "")
        if len(answer.strip()) < 20:
            raise SmokeFailure("chat", "answer was empty or suspiciously short")
        if "could not generate an answer" in answer.lower():
            raise SmokeFailure("chat", answer[:300])
        if not payload.get("session_id") or not payload.get("message_id"):
            raise SmokeFailure("chat", "response omitted session_id or message_id")
        print(f"PASS chat ({time.monotonic() - started:.2f}s, {len(answer)} chars)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-auth", action="store_true")
    args = parser.parse_args()
    try:
        run(args.base_url, args.timeout, args.skip_auth)
    except (httpx.HTTPError, SmokeFailure) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
