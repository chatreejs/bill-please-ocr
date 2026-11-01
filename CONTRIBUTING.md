# Bill Please OCR Contributing Guidelines

Thank you for taking the time to contribute to our project. Please take a moment to read the following guidelines before contributing:

> **⚠️IMPORTANT**
>
> **Pull Requests _having no issue associated_ with them _will not be accepted_. Firstly get an issue assigned, whether it's already opened or raised by you, and then create a Pull Request.**

## Prerequisites ⚠️

- Open Source Etiquette: If you've never contributed to an open source project before, have a read of [Basic etiquette](https://developer.mozilla.org/en-US/docs/MDN/Community/Open_source_etiquette) for open source projects.

- Basic familiarity with Git and GitHub: If you are also new to these tools, visit [GitHub for complete beginners](https://developer.mozilla.org/en-US/docs/MDN/Contribute/GitHub_beginners) for a comprehensive introduction to them.

- Make sure you have Python and [uv](https://docs.astral.sh/uv/) installed. The versions are pinned in `.tool-versions`, so if you use [asdf](https://asdf-vm.com) just run `asdf install`.
- (Optional) [Docker](https://www.docker.com) if you want to test the container build. The build downloads the PaddleOCR models and bakes them into the image.

---

## How to Contribute 🤔

To get started, look at the existing [**Issues**](https://github.com/chatreejs/bill-please-ocr/issues) or [**create a new issue**](https://github.com/chatreejs/bill-please-ocr/issues/new/choose)!

This service only turns a receipt image into text rows. Issues about the receipt scan API or the MCP server belong in [bill-please-mcp](https://github.com/chatreejs/bill-please-mcp/issues), and issues about the web app (UI, bill calculation, translations) belong in [bill-please](https://github.com/chatreejs/bill-please/issues).

### Setup guidelines 🪜

Follow these steps to set up Bill Please OCR on your local machine

1. [Fork](https://github.com/chatreejs/bill-please-ocr/fork) the project
2. Clone the project to run on your local machine using the following command:

   ```sh
   git clone https://github.com/<your_github_username>/bill-please-ocr.git
   ```

3. Get into the root directory

   ```sh
   cd bill-please-ocr
   ```

4. Install all dependencies (this creates `.venv`)

   ```sh
   uv sync
   ```

5. Create your branch from `develop`

   ```sh
   git checkout develop
   git checkout -b <your_branch_name>
   ```

6. Run the server on localhost (http://localhost:8000). The first start downloads the PaddleOCR models, and `/readyz` returns 200 once the warm-up inference is done.

   ```sh
   PORT=8000 uv run python -m receipt_ocr
   ```

7. Make your changes, then make sure the tests and the linter pass

   ```sh
   uv run pytest
   uv run ruff check . && uv run ruff format --check .
   ```

   If you changed the OCR settings or the line reconstruction, also run the tests with the real models on your own receipts in `tests/fixtures/receipts` (see [Receipt fixtures](#receipt-fixtures-))

   ```sh
   uv run pytest -m paddle
   ```

8. Stage your changes

   ```sh
   git add <filename>
   ```

9. Commit your changes

   ```sh
   git commit -m "<your-commit-message>"
   ```

10. Push your changes to your branch

    ```sh
    git push origin "<your_branch_name>"
    ```

11. Create a [PULL REQUEST](https://github.com/chatreejs/bill-please-ocr/compare) against `develop` 💣

    > Click _compare across forks_ if you don't see your branch

---

## Project Conventions 🧭

- **Keep the `/v1/ocr` response stable.** [bill-please-mcp](https://github.com/chatreejs/bill-please-mcp) calls this service over HTTP. If the response has to change, change this service first and the mcp client second.
- **Keep the supported image types in sync.** `SUPPORTED_IMAGE_TYPES` in `src/receipt_ocr/config.py` mirrors `src/schemas/receipt.ts` in bill-please-mcp.
- **Tests never load Paddle.** `create_app(config, load_engine)` takes an engine factory, so the app tests pass a `FakeEngine`. Only tests marked `paddle` use the real models. Add tests for any new behavior in `tests/`.
- **Keep blocking work off the event loop.** Image decoding and OCR run in the threadpool so the probes keep answering, and the engine holds a lock around Paddle's predictor because it is not thread-safe.
- **Never log the image or the recognized text.** Receipts can hold names and card digits. Log sizes, counts, confidence and durations as fields with `extra={"fields": {...}}`.
- **New settings** are environment variables read into the `Config` dataclass in `src/receipt_ocr/config.py`. Document them in the configuration table of `README.md`.
- **Never commit secrets** (`.env`) or real receipts.

### Receipt fixtures 🧾

The `paddle` tests read real receipts from `tests/fixtures/receipts`, as `<name>.jpg` plus `<name>.json` with the answer of the bill (items with name, optional qty and amount, and total). Each image is one test case, and item names pass at an edit distance score of 0.8. The folder is gitignored: **never commit real receipts or attach them to issues or pull requests.**

---

## Issues 🎃

- Select an issue template from the [issues](https://github.com/chatreejs/bill-please-ocr/issues/new/choose) tab. Otherwise, choose **Other** if it doesn't match what you're looking for.
- When creating an issue, make sure you fill up all the fields properly.
- Make sure that you are NOT raising a **duplicate issue**.
- **Note:** If you aren't the owner of the issue, please comment that you're willing to work on the issue and wait for maintainers to assign you the issue. Also, don't work on the issue if you're NOT assigned.
- Work on only **ONE** issue at a time.
- **Security vulnerabilities must not be reported as public issues.** See [SECURITY.md](./SECURITY.md).

**Closing the issue** 📍

- If you decide to close the issue, please leave a brief comment describing why (e.g., I'm busy with other obligations.) before you do.
- **Note:** If the Pull Request associated with the issue gets merged and the issue still remains open, it's **your** responsibility to close the issue.

## Commits Message Guidelines 💬

We follow the [Conventional Commits](https://www.conventionalcommits.org) format. Each commit message should adhere to the following guidelines:

1. **Type**: The commit type must be one of the following:

   - `feat`: A new feature or enhancement.
   - `fix`: A bug fix.
   - `docs`: Documentation changes.
   - `style`: Code style changes (e.g., formatting).
   - `refactor`: Code refactorings with no feature changes or bug fixes.
   - `test`: Adding or improving tests.
   - `ci`: Changes to the Jenkins pipeline.
   - `chore`: General maintenance tasks, build changes, etc.

2. **Scope** (Optional): The scope provides context for the commit, indicating the specific part of the project being affected. Use a short description in lowercase (e.g., `engine`, `lines`, `log`, `docker`).

3. **Description**: A brief and meaningful description of the changes made, in lowercase and in the imperative mood (e.g., "add new feature" instead of "Added new feature").

4. **Issue reference** (Optional): Include the issue number associated with the commit (e.g., `#123`).

#### ✔️ Examples of Valid Commit Messages

- `feat(lines): merge words split across a column gap`
- `fix(image): rotate heic images by their exif orientation`
- `docs: update docker instructions`
- `test: score receipt names by edit distance`
- `chore(deps): update paddleocr`
- `fix: reject empty uploads (#456)`

#### ❌ Examples of Invalid Commit Messages

- `Added new stuff`
- `Fixed a bug`
- `Updated code`
- `engine update`
- `chore: fixed some stuff`

---

## Making Pull Requests 💥

1. Make sure `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` pass before you open a pull request. If you don't know how to resolve a failure, you can ask for help.

2. Each pull request should contain a single logical change or related set of changes that make sense to submit together. If a pull request becomes too large or contains too many unrelated changes, it becomes too difficult to review. In such cases, the reviewer has the right to close your pull request and ask that you submit a separate pull request for each logical set of changes that belong together.

3. Link the issue you have resolved in the pull request description (e.g. Closes/Fixes #99).
4. Use our [Commit messages Guidelines](#commits-message-guidelines-) for your changes.
5. Do not re-open a pull request that a reviewer has closed.
   - Make sure to tick the "Allow edits from maintainers" box. This allows us to directly make minor edits / refactors and saves a lot of time.
     > **Note**
     > If your pull request has merge conflicts with the `develop` branch (GitHub checks for this automatically and notifies you), you are responsible for resolving them. You can do this by merging the `develop` branch into your branch (`git pull upstream develop`), and then pushing the updated branch to your fork (`git push`). If you need more tips, check out [Resolving a merge conflict on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github).

---

## Remarks ✅

- If something is missing here, or you feel something is not well described, either create a PR, [raise an issue](https://github.com/chatreejs/bill-please-ocr/issues), or [do a code review of the person’s PR](https://www.freecodecamp.org/news/code-review-tips/) (ensure that your review conforms to the [Code of Conduct](https://github.com/chatreejs/bill-please-mcp/blob/main/CODE_OF_CONDUCT.md)).

- By contributing, you agree that your contributions will be licensed under the [AGPL-3.0 License](./LICENSE).
