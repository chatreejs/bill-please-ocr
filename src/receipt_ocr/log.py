"""Spring Boot style logging, matching bill-please-mcp src/logger.ts.

  2026-10-11T14:03:12.345+07:00  INFO 4242 --- [bill-please-ocr] [9f2c1a7e] ocr.engine       : Loaded models

The bracket after the app name holds the request id, or `main` outside a request.
"""

import logging
import os
import sys
import warnings
from contextvars import ContextVar
from datetime import datetime

APP_NAME = "bill-please-ocr"
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "OFF")

request_id: ContextVar[str] = ContextVar("request_id", default="main")

# Python names, as they appear in Spring Boot output
_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "ERROR",
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


class SpringFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).astimezone()
        level = _LEVEL_NAMES.get(record.levelno, record.levelname)
        line = (
            f"{timestamp.isoformat(timespec='milliseconds')} {level:>5} {os.getpid()} "
            f"--- [{APP_NAME}] [{request_id.get():>8}] {record.name:<16} : {record.getMessage()}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str, stream=None) -> None:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(SpringFormatter())
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
    adopt_library_loggers()


def adopt_library_loggers() -> None:
    """PaddleX and PaddleOCR install their own colored stderr handler when imported. Send their
    records through our format instead, and only from WARN up, because they log every model load at INFO."""
    root_level = logging.getLogger().level
    for name in ("paddlex", "paddleocr"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(max(logging.WARNING, root_level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
