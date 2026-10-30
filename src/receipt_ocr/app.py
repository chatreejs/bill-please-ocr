import asyncio
import re
import secrets
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from receipt_ocr.config import Config
from receipt_ocr.engine import OcrEngine
from receipt_ocr.image import InvalidImageError, decode_image
from receipt_ocr.lines import build_lines
from receipt_ocr.log import get_logger, request_id

access_log = get_logger("http.access")
log = get_logger("ocr.route")

REQUEST_ID_PATTERN = re.compile(r"^[\w-]{1,64}$")
# Kubernetes probes hit /healthz and /readyz every few seconds
PROBE_USER_AGENT = re.compile(r"^kube-probe/", re.IGNORECASE)
PROBE_PATHS = ("/healthz", "/readyz")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(config: Config, load_engine: Callable[[], OcrEngine]) -> FastAPI:
    """`load_engine` runs in a background thread at startup, so /healthz answers while
    the models load and /readyz turns 200 once a warm-up inference has finished."""

    state: dict[str, OcrEngine | None] = {"engine": None}

    def load() -> None:
        started = time.perf_counter()
        engine = load_engine()
        # The first inference allocates buffers and is several times slower than the next ones
        engine.recognize(np.full((64, 256, 3), 255, dtype=np.uint8))
        state["engine"] = engine
        elapsed = round((time.perf_counter() - started) * 1000)
        log.info("OCR ready in %d ms", elapsed, extra={"fields": {"ocr": {"loadMs": elapsed}}})

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(asyncio.to_thread(load))
        task.add_done_callback(
            lambda t: t.exception() and log.error("OCR engine failed to load", exc_info=t.exception())
        )
        yield

    app = FastAPI(title="bill-please-ocr", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if REQUEST_ID_PATTERN.match(incoming) else secrets.token_hex(4)
        token = request_id.set(rid)
        started = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("Unhandled error on %s %s", request.method, request.url.path)
                response = _error(500, "Internal server error")
            response.headers["x-request-id"] = rid
            duration = round((time.perf_counter() - started) * 1000)
            user_agent = request.headers.get("user-agent", "")
            status = response.status_code
            line = f"{request.method} {request.url.path} {status} {duration}ms"
            fields = {
                "http": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "durationMs": duration,
                },
                **({"clientIp": request.client.host} if request.client else {}),
                **({"userAgent": user_agent} if user_agent else {}),
            }
            extra = {"fields": fields}
            # /readyz answers 503 on purpose while the models load
            probe_ok = status < 400 or (request.url.path == "/readyz" and status == 503)
            if request.url.path in PROBE_PATHS and probe_ok:
                if PROBE_USER_AGENT.match(user_agent):
                    access_log.debug(line, extra=extra)
                else:
                    access_log.info(line, extra=extra)
            elif status >= 500:
                access_log.error(line, extra=extra)
            elif status >= 400:
                access_log.warning(line, extra=extra)
            else:
                access_log.info(line, extra=extra)
            return response
        finally:
            request_id.reset(token)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if state["engine"] is None:
            return _error(503, "OCR models are loading")
        return {"status": "ok"}

    @app.post("/v1/ocr")
    async def ocr(image: UploadFile):
        engine = state["engine"]
        if engine is None:
            return _error(503, "OCR models are loading")
        if image.content_type not in config.supported_types:
            return _error(415, f"Unsupported image type {image.content_type!r}")
        # Read one byte past the limit to detect oversize uploads without loading more
        data = await image.read(config.max_image_bytes + 1)
        if len(data) > config.max_image_bytes:
            return _error(413, f"Image is larger than {config.max_image_bytes} bytes")

        started = time.perf_counter()
        try:
            pixels = await run_in_threadpool(decode_image, data, config.max_image_side)
        except InvalidImageError:
            return _error(422, "Image could not be decoded")
        words = await run_in_threadpool(engine.recognize, pixels)
        lines = build_lines(words)
        duration = round((time.perf_counter() - started) * 1000)

        mean_confidence = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        # Never log the image or the recognized text, because receipts can hold names and card digits
        log.info(
            "OCR complete with %d lines in %d ms",
            len(lines),
            duration,
            extra={
                "fields": {
                    "ocr": {
                        "width": pixels.shape[1],
                        "height": pixels.shape[0],
                        "words": len(words),
                        "lines": len(lines),
                        "meanConfidence": round(mean_confidence, 3),
                        "durationMs": duration,
                    }
                }
            },
        )
        return {
            "lines": [
                {"text": line.text, "confidence": line.confidence, "box": list(line.box)} for line in lines
            ],
            "text": "\n".join(line.text for line in lines),
            "width": pixels.shape[1],
            "height": pixels.shape[0],
            "durationMs": duration,
        }

    return app
