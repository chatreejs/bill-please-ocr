<!-- If your PR fixes an open issue, use `Closes #999` to link your PR with the issue. #999 stands for the issue number you are fixing -->

## Fixes Issue

<!-- Remove this section if not applicable -->

<!-- Example: Closes #31 -->

## Changes proposed

<!-- List all the proposed changes in your PR -->

## How to test

<!-- Steps or curl requests that show your changes work. Never attach real receipts, they hold names and card digits -->

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run pytest -m models` passes (if OCR or line reconstruction changed)
- [ ] The `/v1/ocr` response is unchanged, or bill-please-mcp is updated to match

## Note to reviewers

<!-- Add notes to reviewers if applicable -->
