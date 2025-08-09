import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to the entire log message with a pretty scheme"""

    # ANSI color codes for different elements
    COLORS = {
        "timestamp": "\033[36m",  # Cyan
        "name": "\033[38;5;147m",  # Light purple
        "debug": "\033[38;5;105m",  # Medium purple
        "info": "\033[38;5;78m",  # Seafoam green
        "warning": "\033[38;5;214m",  # Orange
        "error": "\033[38;5;203m",  # Coral red
        "critical": "\033[38;5;196m",  # Bright red
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def format(self, record):
        # Save the original format
        original_format = self._style._fmt

        # Color scheme for different parts of the message
        colored_format = (
            f"{self.DIM}{self.COLORS['timestamp']}%(asctime)s{self.RESET} "  # Dimmed cyan timestamp
            f"{self.COLORS['name']}%(name)s{self.RESET} "  # Light purple logger name
            f"{self.DIM}[%(filename)s:%(lineno)d]{self.RESET} "  # Add filename and line number
            f"{self.BOLD}%(levelname)s{self.RESET} "  # Bold level name (will be colored below)
            f"%(message)s"  # Message (might include colors from code)
        )

        # Update format with colors
        self._style._fmt = colored_format

        # Color the level name based on level
        level_color = {
            logging.DEBUG: self.COLORS["debug"],
            logging.INFO: self.COLORS["info"],
            logging.WARNING: self.COLORS["warning"],
            logging.ERROR: self.COLORS["error"],
            logging.CRITICAL: self.COLORS["critical"],
        }.get(record.levelno, self.RESET)

        record.levelname = f"{level_color}{record.levelname}"

        # Format the record
        result = super().format(record)

        # Restore the original format
        self._style._fmt = original_format

        return result


def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """Get a formatted logger with colored console output.

    Parameters
    ----------
        name: Logger name, by default None (uses __name__)
        level: Logging level, by default logging.INFO

    Returns:
    -------
        logging.Logger: Configured logger instance with colored output
    """
    logger = logging.getLogger(name or __name__)

    # Important: Set propagate to False to prevent duplicate logs
    logger.propagate = False

    # Set the level on the logger itself
    logger.setLevel(level)

    # Only add handler if none exists to avoid duplicates
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        # Important: Set the same level on the handler
        console_handler.setLevel(level)

        # Use our custom colored formatter with date format
        formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        # If handlers exist, update their levels to match the logger
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger
