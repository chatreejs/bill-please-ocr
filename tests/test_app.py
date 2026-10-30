import io
import json
import logging
import re
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from receipt_ocr.app import create_app
from receipt_ocr.config import Config
from receipt_ocr.lines import Word
from receipt_ocr.log import JsonFormatter, configure_logging, request_id

WORDS = [
    Word("Pad Thai", 0.98, (20, 100, 140, 130)),
    Word("2", 0.99, (300, 101, 312, 129)),
    Word("240.00", 1.0, (400, 102, 480, 128)),
    Word("Total", 0.97, (20, 160, 90, 190)),
    Word("240.00", 1.0, (400, 160, 480, 190)),
]


class FakeEngine:
    def __init__(self, words=WORDS):
        self.words = words
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return self.words


def jpeg(size=(600, 400)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, "JPEG")
    return buffer.getvalue()


def wait_ready(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/readyz").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("never became ready")


@pytest.fixture
def client():
    with TestClient(create_app(Config(max_image_bytes=50_000), FakeEngine)) as client:
        wait_ready(client)
        yield client


def scan(client: TestClient, data: bytes, content_type="image/jpeg", **kwargs):
    return client.post("/v1/ocr", files={"image": ("receipt.jpg", data, content_type)}, **kwargs)


def test_ocr_returns_lines(client):
    response = scan(client, jpeg())
    assert response.status_code == 200
    body = response.json()
    assert [" ".join(line["text"].split()) for line in body["lines"]] == ["Pad Thai 2 240.00", "Total 240.00"]
    assert body["lines"][0]["box"] == [20, 100, 480, 130]
    assert body["text"] == "\n".join(line["text"] for line in body["lines"])
    assert (body["width"], body["height"]) == (600, 400)
    assert isinstance(body["durationMs"], int)


def test_rejects_unsupported_type(client):
    assert scan(client, b"%PDF", content_type="application/pdf").status_code == 415


def test_rejects_large_image(client):
    assert scan(client, b"0" * 50_001).status_code == 413


def test_rejects_undecodable_image(client):
    assert scan(client, b"not a jpeg").status_code == 422


def test_missing_field_is_422(client):
    assert client.post("/v1/ocr").status_code == 422


def test_request_id_is_reused_or_generated(client):
    assert client.get("/healthz", headers={"x-request-id": "abc-123"}).headers["x-request-id"] == "abc-123"
    generated = client.get("/healthz", headers={"x-request-id": "bad id!"}).headers["x-request-id"]
    assert len(generated) == 8 and generated != "bad id!"


def test_not_ready_while_models_load():
    release = threading.Event()

    def slow_engine():
        release.wait(5)
        return FakeEngine()

    with TestClient(create_app(Config(), slow_engine)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert scan(client, jpeg()).status_code == 503
        release.set()
        wait_ready(client)
        assert scan(client, jpeg()).status_code == 200


def test_warm_up_runs_before_ready():
    engine = FakeEngine()
    with TestClient(create_app(Config(), lambda: engine)) as client:
        wait_ready(client)
        assert engine.calls == 1


def test_logs_never_contain_recognized_text(client, caplog):
    secret = Word("Card 4111 1111 1111 1111", 0.99, (20, 100, 400, 130))
    with TestClient(create_app(Config(), lambda: FakeEngine([secret]))) as other:
        wait_ready(other)
        with caplog.at_level(logging.DEBUG):
            assert scan(other, jpeg()).status_code == 200
    assert "4111" not in caplog.text
    record = next(r for r in caplog.records if r.name == "ocr.route")
    assert record.fields["ocr"]["words"] == 1
    assert record.fields["ocr"]["lines"] == 1


def test_json_log_format():
    record = logging.LogRecord("ocr.route", logging.WARNING, __file__, 1, "hello %s", ("world",), None)
    record.fields = {"ocr": {"lines": 3}}
    line = JsonFormatter().format(record)
    assert "\n" not in line
    entry = json.loads(line)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", entry.pop("time"))
    assert entry == {
        "level": "warn",
        "service": "bill-please-ocr",
        "logger": "ocr.route",
        "msg": "hello world",
        "ocr": {"lines": 3},
    }


def test_json_log_request_id_and_error():
    token = request_id.set("abcd1234")
    try:
        try:
            raise ValueError("outer") from KeyError("inner")
        except ValueError:
            record = logging.LogRecord("ocr.route", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
        entry = json.loads(JsonFormatter().format(record))
    finally:
        request_id.reset(token)
    assert entry["requestId"] == "abcd1234"
    assert entry["error"]["type"] == "ValueError"
    assert entry["error"]["message"] == "outer"
    assert entry["error"]["cause"]["type"] == "KeyError"
    assert "Traceback" in entry["error"]["stack"]


def test_pillow_stays_quiet_at_debug():
    configure_logging("DEBUG", io.StringIO())
    assert not logging.getLogger("PIL.TiffImagePlugin").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("PIL.TiffImagePlugin").isEnabledFor(logging.WARNING)
