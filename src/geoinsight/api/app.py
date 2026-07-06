from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geoinsight.api.errors import unhandled_exception_handler
from geoinsight.api.routes import router
from geoinsight.config import load_config


def create_app() -> FastAPI:
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("geoinsight.api")
    app = FastAPI(
        title="GeoInsight Agent API",
        description=(
            "Production-style API for geospatial analytics, hybrid retrieval, "
            "LangGraph agent workflows, and evidence-grounded LLM reasoning."
        ),
        version="0.1.0",
    )

    @app.middleware("http")
    async def log_requests(request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            logger.exception(
                "api_request_failed path=%s method=%s provider=%s error=%s",
                request.url.path,
                request.method,
                config.llm_provider,
                exc.__class__.__name__,
            )
            raise
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "api_request path=%s method=%s status=%s latency_ms=%s provider=%s",
                request.url.path,
                request.method,
                status_code,
                latency_ms,
                config.llm_provider,
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    return app


app = create_app()
