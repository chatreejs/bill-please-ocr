# bill-please-ocr

OCR service for [bill-please-mcp](https://github.com/chatreejs/bill-please-mcp). It takes a receipt photo and returns the text as receipt rows, ready for the LLM step. It uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 3.x with `PP-OCRv5_mobile_det` for text detection and `th_PP-OCRv5_mobile_rec` for recognition. The Thai recognition model also reads English letters and digits. It runs on the CPU.

## Develop

`.tool-versions` pins Python and uv. Install them with asdf.

```sh
asdf install                  # python 3.13.18, uv 0.13.6
uv sync                       # creates .venv
uv run pytest                 # unit tests, no models needed
uv run pytest -m paddle       # runs the real models on tests/fixtures/receipts
uv run ruff check . && uv run ruff format --check .
PORT=8000 uv run python -m receipt_ocr
```

The first run downloads the models from Hugging Face into `~/.paddlex/official_models` (about 13 MB). The Docker image already contains them.

```sh
curl -F image=@tests/fixtures/receipts/<name>.jpg localhost:8000/v1/ocr
```

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/ocr` | multipart field `image` (JPEG, PNG, WebP, HEIC or HEIF, at most `MAX_IMAGE_BYTES`). Returns the lines. |
| `GET /healthz` | Liveness. 200 once the process serves HTTP. |
| `GET /readyz` | Readiness. 503 while the models load, 200 after a warm-up inference. |

Response of `POST /v1/ocr`:

```json
{
  "lines": [
    { "text": "ผัดไทยกุ้งสด                  2    240.00", "confidence": 0.9955, "box": [37, 166, 561, 212] }
  ],
  "text": "...all lines joined with \n...",
  "width": 612,
  "height": 780,
  "durationMs": 3254
}
```

`box` is `[x0, y0, x1, y1]` in pixels of the decoded image. The service rotates the image by its EXIF orientation tag and scales it down to at most `MAX_IMAGE_SIDE`. `width` and `height` are the size after that.

Errors return `{"error": "..."}` with status 413 (too large), 415 (unsupported type), 422 (missing field or undecodable image) or 503 (models still loading). The service reuses an incoming `X-Request-Id` in its logs and sends it back in the response.

## Line reconstruction

PaddleOCR returns one box per text run, so "Pad Thai", "2" and "240.00" come back as three boxes. `lines.py` rebuilds the rows in three steps.

1. It estimates the photo's tilt from the top edges of the wide boxes and rotates the box centres back.
2. It groups boxes whose vertical centres are within half a text height of each other.
3. It places each word at the text column that matches its x position, using the median character width. It places numbers by their right edge, so a right-aligned amount column stays aligned. Thai vowel and tone marks take no column.

## Configuration

| Env | Default | Notes |
| --- | --- | --- |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `INFO` | `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR` or `OFF`. Logs are JSON lines with the same fields as bill-please-mcp. |
| `MAX_IMAGE_BYTES` | `5242880` | Same limit as bill-please-mcp |
| `MAX_IMAGE_SIDE` | `1600` | The service scales larger images down and never scales smaller ones up. |
| `OCR_DET_MODEL` | `PP-OCRv5_mobile_det` | `PP-OCRv5_server_det` (84 MB) took 5.1 s instead of 3.2 s on the synthetic receipt, with no better result. In the Docker image it is a build arg, because the build bakes the models in. |
| `OCR_REC_MODEL` | `th_PP-OCRv5_mobile_rec` | A build arg in the Docker image, like `OCR_DET_MODEL` |
| `OCR_MIN_CONFIDENCE` | `0.5` | Drops words recognized with a lower score, such as stray marks |
| `OCR_DEVICE` | `cpu` | |
| `OCR_CPU_THREADS` | `2` | Set it to the pod's CPU limit |
| `OCR_ENABLE_MKLDNN` | `false` | oneDNN on x86. Paddle 3.3 fails on the first predict with `ConvertPirAttribute2RuntimeAttribute not support`, so leave it off. |

The service never logs the image or the recognized text, because receipts can hold names and card digits. Each OCR request logs the image size, word and line counts, the mean confidence and the duration.

## Fixtures

Put real receipts in `tests/fixtures/receipts`: an image `<name>.jpg` (or .png, .webp, .heic, .heif) and the answer of the bill in `<name>.json`, as `{"items": [{"name": ..., "qty": ..., "amount": ...}], "total": ...}`. `.gitignore` keeps the folder out of git, because receipts hold names and card digits.

`uv run pytest -m paddle` scores each receipt with `tests/receipts.py`. An item's candidate lines are the ones that hold its amount, and its qty when given. Its score is 1 minus the edit distance from the name to the closest stretch of a candidate line, per character, so a dropped Thai tone mark costs a little. Items pass at 0.8, and some line must hold the total.

## Image

`Dockerfile` builds `chatreejs/bill-please-ocr` on `python:3.13.18-slim`. It runs as uid 1000, and the build downloads and loads the models once. Jenkins builds it with `ci/Jenkinsfile`.
