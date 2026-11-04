# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. `README.md` covers the API, the configuration and line reconstruction.

## Commands

```bash
uv sync                          # creates .venv
uv run pytest                    # unit tests, no models needed
uv run pytest -m models          # real ONNX models (OCR_ONNX_MODEL_DIR) on tests/fixtures/receipts
uv run ruff check . && uv run ruff format --check .
PORT=8000 uv run python -m receipt_ocr
```

`.tool-versions` pins Python 3.13.18 and uv 0.13.6, installed with asdf. The user's global `~/.tool-versions` has `python system`, so `python3` outside this folder is the system Python.

## Purpose

This service is the OCR step of the planned hybrid scanner. The design is in `../bill-please-mcp/docs/hybrid-ocr-llm-pipeline.md`, which exists only locally and is not in git.

It runs FastAPI and ONNX Runtime on the GPU only (the user wants no CPU image), with the PaddleOCR models `PP-OCRv5_mobile_det` for detection and `th_PP-OCRv5_mobile_rec` for recognition. `POST /v1/ocr` takes a multipart `image` and returns receipt rows rebuilt from the OCR boxes. `/healthz` and `/readyz` are the probes.

The user decided to keep OCR in Python and to keep the Elysia server in `../bill-please-mcp` as the API, which will call this service over HTTP. The user also decided on separate repos. Keep the `/v1/ocr` response stable. If it has to change, change this service first and the mcp client second.

## Architecture

```
src/receipt_ocr/
  __main__.py   entry point: load_config, configure_logging, create_app, then uvicorn with one worker
  app.py        create_app(config, load_engine): request id and access log middleware, routes, engine loaded in a background thread
  config.py     environment variables into a frozen Config dataclass; SUPPORTED_IMAGE_TYPES mirrors ../bill-please-mcp/src/schemas/receipt.ts
  log.py        JSON lines with the same fields as ../bill-please-mcp/src/logger.ts; request id in a ContextVar
  image.py      decode_image: Pillow and pi-heif, EXIF rotation, downscale to MAX_IMAGE_SIDE, RGB to BGR
  engine.py     OcrEngine protocol; OnnxOcrEngine ports the PaddleX 3.7.2 det/rec pre- and post-processing onto ONNX Runtime and locks around the sessions
  lines.py      Word and Line, skew estimate, row grouping, column layout with right-aligned numbers
scripts/      convert_onnx.py, run by the Dockerfile's paddle2onnx stage
tests/          pytest; FakeEngine for the app tests; real receipts in fixtures/receipts for the models test, scored by receipts.py
```

Key points:

- **Dependency injection.** `create_app(config, load_engine)` takes a factory, so tests pass a `FakeEngine` and never load ONNX Runtime. `/readyz` returns 200 only after a warm-up inference.
- **Blocking work.** `decode_image` and `engine.recognize` run in the threadpool, so the probes keep answering during an OCR request. The engine holds a lock around the ONNX Runtime sessions, so one request at a time uses the GPU.
- **Logging.** Never log the image or the recognized text, because receipts can hold names and card digits. Log the image size, the word and line counts, the mean confidence and the duration.
- **Log format.** Every record is one JSON object: `time` (UTC), `level` (lowercase), `service`, `logger`, `requestId` (only inside a request) and `msg`. Pass numbers as fields with `extra={"fields": {...}}`. An exception becomes `error: { type, message, stack, cause }`. The access log has `http: { method, path, status, durationMs }`, `clientIp` and `userAgent`, like bill-please-mcp.
- **Models.** The Dockerfile's `python:3.12-slim` stage downloads them from Hugging Face at pinned revisions and converts them with paddle2onnx (Python 3.12 at most) into `/opt/onnx/<model>/inference.onnx` and `config.json`. Changing `OCR_DET_MODEL`, `OCR_REC_MODEL` or the revisions needs a rebuild, because they are build args. The build warms up on the CPU, because the build machine has no GPU.
- **GPU only.** `onnxruntime-gpu[cuda,cudnn]` is linux-only in `pyproject.toml`, so `uv sync` on the Mac skips it and the unit tests still run. Keep it at 1.26, because 1.27 needs CUDA 13 and driver 580, and the dev GPU node has driver 570. The CUDA provider uses `OCR_GPU_MEM_LIMIT_MB` (768) because llm-server shares the 4 GB dev GPU, and the engine fails to start if CUDA does not load. `pyproject.toml` pins `nvidia-cudnn-cu12` to 9.26.1.1, because with 9.27 the `HEURISTIC` algorithm search finds nothing for some detection convolutions on a Turing GPU. `DEFAULT` works around that but runs cuDNN's fallback algorithms, about 4 times slower. The sessions disable ONNX Runtime's silent CPU fallback and shrink the CUDA arena after each run, and the warm-up runs a blank image at `MAX_IMAGE_SIDE` square. The Jenkins test stage skips the CUDA wheels with `--no-install-package`. Keep `engine.py` in step with the PaddleX pipeline settings, or it stops reading the same text as PaddleOCR.
- **Fixtures.** The user drops real receipts into `tests/fixtures/receipts` as `<name>.jpg` plus `<name>.json`, the answer of the bill (items with name, optional qty and amount, and total). `test_models.py` runs one case per image and scores names by edit distance with `tests/receipts.py` (pass at 0.8). On the first three bills the default settings beat wider boxes (`text_det_unclip_ratio` 2.0 and 2.5), a 2400 px image and `PP-OCRv5_server_det`, so the user keeps the defaults. The folder is gitignored, so keep real receipts out of git.

## Conventions

- Code work happens on `develop`. Commits follow Conventional Commits with a lowercase subject.
- CI is Jenkins only, with no GitHub Actions workflows. The user never runs Docker locally.

## Deployment

`Dockerfile` builds `chatreejs/bill-please-ocr` on `python:3.13.18-slim` and runs it as uid 1000. The Jenkins pipeline in `ci/Jenkinsfile` runs ruff and the tests with coverage, scans with SonarQube, builds, scans the image with Trivy and pushes. Its last stage triggers the gitops job `chatreejs/gitops/bill-please-backend-manifest-{dev,prod}` with `OCR_IMAGE_TAG`. That manifest repo (`../../gitlab.com/chatreedev-manifest/bill-please-backend`) deploys this service next to bill-please-mcp, and a NetworkPolicy admits only the mcp pods.

- **Dev.** The Argo CD app `bill-please-backend-dev` auto-syncs the manifest repo, so a push to `develop` reaches dev once Jenkins has built the image. Dev runs at `LOG_LEVEL=DEBUG`, and Pillow stays at WARN so its EXIF records don't flood the logs. OCR runs on a GPU node, about 6.9 times faster than the cluster CPU (4 s for a small receipt and 11 s for a long one) with the same fixture scores. The manifest's `ocr.gpu.request` is 0, so the pod requests no `nvidia.com/gpu` and shares the card that llm-server holds. There is no prod Argo CD app yet.
- **Shutdown.** The pod sleeps 5 s in `preStop` so the Service stops routing to it, then uvicorn finishes in-flight OCR within `terminationGracePeriodSeconds: 30`. The mcp side stops waiting after 20 s.
- **Static analysis.** SonarQube, like bill-please-mcp: the Test stage writes `coverage.xml` with pytest-cov, and the Static Code Scan stage runs `sonar-scanner` with `sonar-project.properties` (project key `chatreejs_bill-please-ocr`) on the `SonarQube Server` Jenkins connection. There is no CodeQL.
