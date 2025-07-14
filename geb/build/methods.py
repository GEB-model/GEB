import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("GEB")

__all__ = ["build_method"]


class _build_method:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.tree = {}

    def __call__(
        self, func: Callable[..., Any] | None = None, depends_on: None = None
    ) -> Callable[..., Any]:
        def partial_decorator(func):
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.logger.info(f"Running {func.__name__}")
                for key, value in kwargs.items():
                    self.logger.debug(f"{func.__name__}.{key}: {value}")
                value = func(*args, **kwargs)
                self.logger.info(f"Completed {func.__name__}")
                return value

            if depends_on is not None:
                if isinstance(depends_on, str):
                    self.tree[func.__name__] = [depends_on]
                elif isinstance(depends_on, list):
                    self.tree[func.__name__] = depends_on
                else:
                    raise ValueError("depends_on must be a string or a list of strings")

            wrapper.__is_build_method__ = True
            return wrapper

        if func is None:
            return partial_decorator
        else:
            f = partial_decorator(func)
            return f


build_method = _build_method(logger=logger)
