from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def service_unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


def bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def actionable_file_error(exc: FileNotFoundError) -> HTTPException:
    message = str(exc)
    if "places.faiss" in message or "places_metadata" in message or "vector index" in message.lower():
        return service_unavailable(
            "Vector index not found. Run `python -m geoinsight.cli build-index` first."
        )
    return not_found(message)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "GeoInsight service error. Check generated data files and local providers, then retry.",
            "error_type": exc.__class__.__name__,
        },
    )
