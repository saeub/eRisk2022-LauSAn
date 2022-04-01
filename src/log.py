import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    log_handler = logging.StreamHandler(sys.stderr)
    log_handler.setFormatter(
        logging.Formatter("[{asctime}] {levelname}: {message}", style="{")
    )
    logger.addHandler(log_handler)
