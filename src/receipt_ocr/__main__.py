import uvicorn

from receipt_ocr.app import create_app
from receipt_ocr.config import load_config
from receipt_ocr.log import configure_logging, get_logger


def main() -> None:
    config = load_config()
    configure_logging(config.log_level)
    log = get_logger("main")

    def load_engine():
        from receipt_ocr.engine import OnnxOcrEngine

        return OnnxOcrEngine(config)

    app = create_app(config, load_engine)
    log.info("Starting bill-please-ocr on port %d", config.port)
    # One worker, because each worker holds its own copy of the models in VRAM.
    # uvicorn finishes in-flight requests on SIGTERM.
    uvicorn.run(app, host="0.0.0.0", port=config.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
