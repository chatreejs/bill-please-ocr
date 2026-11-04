"""Download PaddleOCR models from Hugging Face and convert them to ONNX for the image.

    python scripts/convert_onnx.py OUT_DIR MODEL=REVISION [MODEL=REVISION ...]

For each model this writes OUT_DIR/MODEL/inference.onnx and OUT_DIR/MODEL/config.json, which is the
model's inference.yml as JSON (OnnxOcrEngine reads the recognition dictionary from it).

paddle2onnx supports Python 3.8 to 3.12 only, so the Dockerfile runs this in a python:3.12 stage
with paddlepaddle, paddle2onnx and PyYAML. The service never imports it.
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

FILES = ("inference.json", "inference.pdiparams", "inference.yml")
OPSET = os.environ.get("ONNX_OPSET", "14")


def download(model: str, revision: str, folder: Path) -> None:
    for name in FILES:
        url = f"https://huggingface.co/PaddlePaddle/{model}/resolve/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response, open(folder / name, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)


def convert(model: str, revision: str, out_dir: Path) -> None:
    target = out_dir / model
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        download(model, revision, source)
        subprocess.run(
            [
                "paddle2onnx",
                "--model_dir", str(source),
                "--model_filename", "inference.json",
                "--params_filename", "inference.pdiparams",
                "--save_file", str(target / "inference.onnx"),
                "--opset_version", OPSET,
            ],
            check=True,
        )  # fmt: skip
        config = yaml.safe_load((source / "inference.yml").read_text(encoding="utf-8"))
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    print(f"Converted {model}@{revision[:12]} to {target / 'inference.onnx'}")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit(__doc__)
    out_dir = Path(argv[0])
    for spec in argv[1:]:
        model, _, revision = spec.partition("=")
        convert(model, revision or "main", out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
