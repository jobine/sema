# A logger utility support standard output with color and file logging
# 
# Usage:
#   1. Call setup_logging() once at application startup (e.g., in main.py)
#   2. In each module, use: logger = logging.getLogger(__name__)
#
# Example:
#   # main.py
#   from utils.logging import setup_logging
#   setup_logging()
#
#   # benchmarks/utils.py
#   import logging
#   logger = logging.getLogger(__name__)
#   logger.info("This will be logged with module name 'benchmarks.utils'")

import os
import sys
import logging
from contextlib import contextmanager
from logging import Logger

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""

    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f'{log_color}{message}{self.RESET}'


def setup_logging(log_file: str = None, level: int = logging.DEBUG) -> None:
    """
    Setup root logger with colored console output and optional file logging.
    Call this once at application startup.
    
    Note: For proper UTF-8 support on Windows, set environment variable:
        PYTHONIOENCODING=utf-8
    Or run Python with: python -X utf8
    """
    root_logger = logging.getLogger()
    
    # Avoid adding multiple handlers
    if root_logger.handlers:
        return
    
    root_logger.setLevel(level)

    # Console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with standard formatter
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    """
    Get a logger with the specified name.
    Make sure setup_logging() is called before using this.
    """
    return logging.getLogger(name)


@contextmanager
def suppress_logging(level: int = logging.ERROR, logger: Logger | None = None):
    """
    Temporarily raise the log threshold to the specified level within a context.

    Args:
        level: Temporary log level to apply. Messages below this level are suppressed.
        logger: Target logger. Defaults to the root logger when not provided.
    """
    target_logger = logger or logging.getLogger()
    original_level = target_logger.level
    target_logger.setLevel(level)
    try:
        yield
    finally:
        target_logger.setLevel(original_level)


# Default log path
_default_log_path = os.path.normpath(os.path.join(os.path.expanduser('~'), '.sema', 'logs', 'sema.log'))

# For convenience, setup logging when this module is imported
# You can also call setup_logging() explicitly in your main.py for more control
setup_logging(_default_log_path)


# Example usage:
if __name__ == "__main__":
    # Backward compatible: export a default logger instance
    logger = get_logger('SEMA')

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')