"""
Logging configuration
"""

import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration with Rich formatting."""
    console = Console(stderr=True)

    # Determine log level
    level = logging.DEBUG if verbose else logging.INFO

    # Create rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )

    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name or "arc_competition")
