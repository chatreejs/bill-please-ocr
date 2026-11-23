"""JSON lines logging, with the same fields as bill-please-api src/logger.ts.

  {"time":"2026-10-30T14:03:12.345Z","level":"info","service":"bill-please-ocr",
   "logger":"ocr.engine","msg":"Loaded OCR models"}

Each record is one line. The example wraps only to fit here.

Records written inside an OpenTelemetry span carry `traceId` and `spanId`. bill-please-api sends
the trace in the `traceparent` header, so a scan has the same `traceId` in both services' logs.
Grafana uses `traceId` to link a log line in Loki to its trace in Tempo. In the cluster, the
OpenTelemetry Operator injects the SDK. Without the SDK, the span is never valid and the records
have no ids. Pass structured values with `extra={"fields": {...}}`, and an exception becomes the
`error` field.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime

from opentelemetry import trace

APP_NAME = "bill-please-ocr"
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "OFF")

# The level names bill-please-api writes
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
        span = trace.get_current_span().get_span_context()
        if span.is_valid:
            entry["traceId"] = format(span.trace_id, "032x")
            entry["spanId"] = format(span.span_id, "016x")
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
    # Pillow logs every EXIF tag of every photo at DEBUG
    logging.getLogger("PIL").setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
