# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. `README.md` covers the API, the configuration and line reconstruction.

## Commands

```bash
uv sync                          # creates .venv
uv run pytest                    # unit tests, no models needed
uv run pytest -m paddle          # real PaddleOCR models on tests/fixtures/receipts
uv run ruff check . && uv run ruff format --check .
PORT=8000 uv run python -m receipt_ocr
```

`.tool-versions` pins Python 3.13.18 and uv 0.13.6, installed with asdf. The user's global `~/.tool-versions` has `python system`, so `python3` outside this folder is the system Python.

## Purpose

This service is the OCR step of the planned hybrid scanner. The design is in `../bill-please-mcp/docs/hybrid-ocr-llm-pipeline.md`, which exists only locally and is not in git.

It runs FastAPI and PaddleOCR 3.7 on the CPU, with `PP-OCRv5_mobile_det` for detection and `th_PP-OCRv5_mobile_rec` for recognition. `POST /v1/ocr` takes a multipart `image` and returns receipt rows rebuilt from the OCR boxes. `/healthz` and `/readyz` are the probes.

The user decided to keep OCR in Python and to keep the Elysia server in `../bill-please-mcp` as the API, which will call this service over HTTP. The user also decided on separate repos. Keep the `/v1/ocr` response stable. If it has to change, change this service first and the mcp client second.

## Architecture

```
src/receipt_ocr/
  __main__.py   entry point: load_config, configure_logging, create_app, then uvicorn with one worker
  app.py        create_app(config, load_engine): request id and access log middleware, routes, engine loaded in a background thread
  config.py     environment variables into a frozen Config dataclass; SUPPORTED_IMAGE_TYPES mirrors ../bill-please-mcp/src/schemas/receipt.ts
  log.py        Spring Boot style format, the same as ../bill-please-mcp/src/logger.ts; request id in a ContextVar
  image.py      decode_image: Pillow and pi-heif, EXIF rotation, downscale to MAX_IMAGE_SIDE, RGB to BGR
  engine.py     OcrEngine protocol; PaddleOcrEngine imports paddleocr lazily and locks around predict
  lines.py      Word and Line, skew estimate, row grouping, column layout with right-aligned numbers
tests/          pytest; FakeEngine for the app tests; real receipts in fixtures/receipts for the paddle test
```

Key points:

- **Dependency injection.** `create_app(config, load_engine)` takes a factory, so tests pass a `FakeEngine` and never import Paddle. `/readyz` returns 200 only after a warm-up inference.
- **Blocking work.** `decode_image` and `engine.recognize` run in the threadpool, so the probes keep answering during an OCR request. Paddle's predictor is not thread-safe, so the engine holds a lock around it.
- **Logging.** Never log the image or the recognized text, because receipts can hold names and card digits. Log the image size, the word and line counts, the mean confidence and the duration. `adopt_library_loggers` sends PaddleX and PaddleOCR records through our format, from WARN up.
- **Models.** The Docker build bakes them into the image under `PADDLE_PDX_CACHE_HOME=/opt/paddlex`. Changing `OCR_DET_MODEL` or `OCR_REC_MODEL` in the image needs a rebuild, because they are build args.
- **Fixtures.** The user drops real receipts into `tests/fixtures/receipts` as `<name>.jpg` plus `<name>.json`, the answer of the bill (items with name, optional qty and amount, and total). `test_paddle.py` runs one case per image. The folder is gitignored, so keep real receipts out of git.

## Conventions

- Code work happens on `develop`. Commits follow Conventional Commits with a lowercase subject.
- CI is Jenkins only, with no GitHub Actions CI. The user never runs Docker locally.

## Deployment

`Dockerfile` builds `chatreejs/bill-please-ocr` on `python:3.13.18-slim` and runs it as uid 1000. The Jenkins pipeline in `ci/Jenkinsfile` tests, builds, scans with Trivy and pushes. Its last stage triggers the gitops job `chatreejs/gitops/bill-please-backend-manifest-{dev,prod}` with `OCR_IMAGE_TAG`. That manifest repo (`../../gitlab.com/chatreedev-manifest/bill-please-backend`) deploys this service next to bill-please-mcp, and a NetworkPolicy admits only the mcp pods. Nobody has run the Docker build or created the Jenkins job yet.
