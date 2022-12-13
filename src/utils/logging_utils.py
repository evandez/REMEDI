"""Utilities to make using python logging module easier."""
import logging
import sys
from typing import Any

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def configure(**kwargs: Any) -> None:
    """Configure logging globally with reasonable defaults."""
    kwargs.setdefault("stream", sys.stdout)
    kwargs.setdefault("format", DEFAULT_FORMAT)
    kwargs.setdefault("datefmt", DEFAULT_DATEFMT)
    kwargs.setdefault("level", DEFAULT_LEVEL)
    logging.basicConfig(**kwargs)
