from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

from config.config import LoggingConfig


def _build_formatter(config: LoggingConfig) -> logging.Formatter:
    return logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)


def setup_logging(log_dir: Path, config: LoggingConfig) -> None:
    """Configure root logging once for the full project lifecycle."""

    log_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = _build_formatter(config)

    if config.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if config.enable_file:
        file_handler = logging.FileHandler(
            filename=log_dir / config.log_filename,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    """Return a named logger for module-level use."""

    return logging.getLogger(name)
