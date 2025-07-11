import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("GEB")


def _build_method(
    func: Callable[..., Any], logger: logging.Logger
) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Running {func.__name__}")
        for key, value in kwargs.items():
            logger.debug(f"{func.__name__}.{key}: {value}")
        value = func(*args, **kwargs)
        logger.info(f"Completed {func.__name__}")
        return value

    return wrapper


build_method = functools.partial(_build_method, logger=logger)
