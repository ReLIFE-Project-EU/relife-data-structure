import logging
import os
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except Exception:
    Console = None
    RichHandler = None
    _RICH_AVAILABLE = False


def _resolve_log_level(verbose: bool, env_var: str) -> int:
    """Resolve the log level from the environment variable or use the verbose flag."""

    level_name = os.getenv(env_var)

    if level_name:
        candidate = getattr(logging, str(level_name).upper(), None)

        if isinstance(candidate, int):
            return candidate

    return logging.DEBUG if verbose else logging.INFO


def configure_logging(
    verbose: bool = False,
    *,
    env_var: str = "LOG_LEVEL",
    console: Optional[Any] = None,
) -> None:
    """Configure application logging.

    Prefers Rich logging when available. Honors an environment variable for
    log level, falling back to DEBUG/INFO depending on ``verbose``.

    Args:
        verbose: When True, sets log level to DEBUG unless overridden by env.
        env_var: Environment variable name to read the desired log level from.
        console: Optional ``rich.console.Console`` instance to use for RichHandler.
    """

    level = _resolve_log_level(verbose, env_var)

    if _RICH_AVAILABLE:
        use_console = console

        if use_console is None and Console is not None:
            try:
                use_console = Console()
            except Exception:
                use_console = None

        if use_console is not None and RichHandler is not None:
            logging.basicConfig(
                level=level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[
                    RichHandler(
                        rich_tracebacks=True,
                        markup=True,
                        console=use_console,
                    )
                ],
            )

            return

    # Fallback basic configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
