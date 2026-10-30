"""JSON lines logging, with the same fields as bill-please-mcp src/logger.ts.

  {"time":"2026-10-30T14:03:12.345Z","level":"info","service":"bill-please-ocr",
   "logger":"ocr.engine","msg":"Loaded OCR models"}

Each record is one line; the example wraps only to fit here.

`requestId` is added inside a request. Pass structured values with `extra={"fields": {...}}`,
and an exception becomes the `error` field.
"""

import json
import logging
import sys
import traceback
import warnings
from contextvars import ContextVar
from datetime import UTC, datetime

APP_NAME = "bill-please-ocr"
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "OFF")

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# The level names bill-please-mcp writes
_LEVEL_NAMES = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}
_TO_PYTHON = {
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "OFF": logging.CRITICAL + 10,
}


def parse_log_level(value: str | None, fallback: str = "INFO") -> str:
    upper = (value or "").strip().upper()
    return upper if upper in LOG_LEVELS else fallback


def _serialize_error(error: BaseException) -> dict:
    serialized = {
        "type": type(error).__name__,
        "message": str(error),
        "stack": "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip(),
    }
    if error.__cause__ is not None:
        serialized["cause"] = _serialize_error(error.__cause__)
    return serialized


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": _LEVEL_NAMES.get(record.levelno, record.levelname.lower()),
            "service": APP_NAME,
            "logger": record.name,
        }
        rid = request_id.get()
        if rid:
            entry["requestId"] = rid
        entry["msg"] = record.getMessage()
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info and record.exc_info[1] is not None:
            entry["error"] = _serialize_error(record.exc_info[1])
        # default=str keeps the record when a field is not JSON, for example a numpy number
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_logging(level: str, stream=None) -> None:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(_TO_PYTHON[parse_log_level(level)])
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    # Our middleware writes the access log
    logging.getLogger("uvicorn.access").disabled = True
    # Paddle's C++ extension loader warns about a missing compiler cache. We compile nothing.
    warnings.filterwarnings("ignore", message="No ccache found")
    # Pillow logs every EXIF tag of every photo at DEBUG
    logging.getLogger("PIL").setLevel(max(logging.WARNING, root.level))
    adopt_library_loggers()


def adopt_library_loggers() -> None:
    """PaddleX and PaddleOCR install their own colored stderr handler when imported. Send their records
    through our JSON format instead, and only from WARN up, because they log every model load at INFO."""
    root_level = logging.getLogger().level
    for name in ("paddlex", "paddleocr"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(max(logging.WARNING, root_level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
