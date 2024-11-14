import logging
from logging import Logger
from pathlib import Path


def setup_logger(output_path: Path) -> Logger:
    """Setup python logger."""
    # Create the log file path
    log_file = output_path / "compute_metric.log"

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers (important for pytest re-runs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Optional: Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger
