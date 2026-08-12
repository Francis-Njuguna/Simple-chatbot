"""Custom exceptions."""

from fastapi import HTTPException, status


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class AuthenticationError(AppError):
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class IngestionError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class DatabaseUnavailableError(AppError):
    """Could not obtain a database connection (pool exhausted / DB unreachable).

    503 rather than 500: the request failed because the server is saturated, not
    because it was malformed, so a client may retry. The message is deliberately
    generic — pool sizes, hostnames and driver text are diagnostics for our logs,
    never for the caller.
    """

    def __init__(self, message: str = "Service is busy. Please retry shortly.") -> None:
        super().__init__(message, status_code=503)


def app_error_to_http(exc: AppError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)
