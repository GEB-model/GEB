"""Contains classes and methods for building the dependency tree of build methods, verification etc."""

import functools
import inspect
import logging
from logging import Logger
from pathlib import Path
from time import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx

logger: Logger = logging.getLogger("GEB")

__all__: list[str] = ["build_method"]


class _build_method:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.tree = nx.DiGraph()
        self.time_taken: dict[str, float] = {}

    def __call__(
        self,
        func: Callable[..., Any] | None = None,
        depends_on: str | list[str] | None = None,
    ) -> Callable[..., Any]:
        def partial_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.logger.info(f"Running {func.__name__}")
                for key, value in kwargs.items():
                    self.logger.debug(f"{func.__name__}.{key}: {value}")

                start_time: float = time()
                value: Any = func(*args, **kwargs)
                end_time: float = time()

                elapsed_time: float = end_time - start_time

                self.time_taken[func.__name__] = elapsed_time
                self.logger.info(
                    f"Completed {func.__name__} in {elapsed_time:.2f} seconds"
                )
                return value

            self.add_tree_node(func)

            if depends_on is not None:
                if isinstance(depends_on, str):
                    self.add_tree_edge(func, depends_on)
                elif isinstance(depends_on, list):
                    for item in depends_on:
                        self.add_tree_edge(func, item)
                else:
                    raise ValueError("depends_on must be a string or a list of strings")

            wrapper.__is_build_method__ = True
            return wrapper

        if func is None:
            return partial_decorator
        else:
            f: Callable[..., Any] = partial_decorator(func)
            return f

    def add_tree_node(self, func: Callable[..., Any]) -> None:
        """Add a node to the dependency tree."""
        parameters = inspect.signature(func).parameters
        required_parameters = [
            param.name
            for param in parameters.values()
            if param.default is inspect.Parameter.empty and param.name != "self"
        ]
        optional_parameters = [
            param.name
            for param in parameters.values()
            if param.default is not inspect.Parameter.empty
        ]
        self.tree.add_node(
            func.__name__,
            attr={
                "_function_exists": True,
                "_required_parameters": required_parameters,
                "_optional_parameters": optional_parameters,
            },
        )

    def add_tree_edge(self, func: Callable[..., Any], depends_on: str) -> None:
        """Add an edge to the dependency tree.

        Raises:
            ValueError: if a method depends on "setup_region" since everything depends on it.
                "setup_region" should therefore not be included in the dependency tree.

        """
        if depends_on == "setup_region":
            raise ValueError(
                "Everything depends on setup_region so we don't include it."
            )
        self.tree.add_edge(depends_on, func.__name__)

    def validate_tree(self) -> None:
        """Validate the dependency tree.

        Checks if all the node dependencies are present in the tree.

        Raises:
            ValueError: if a method depends on another method that is not a build function.
        """
        assert nx.is_directed_acyclic_graph(self.tree)
        for method in self.tree.nodes:
            depencencies = list(self.tree.predecessors(method))
            for dependency in depencencies:
                if (
                    not self.tree.nodes[dependency]
                    .get("attr", {})
                    .get("_function_exists", False)
                ):
                    raise ValueError(
                        f"Method {method} depends on {dependency}, "
                        "which is not a build function."
                    )
        self.logger.debug("Builder dependency tree validation passed.")

    def validate_methods(
        self, methods: dict[str, Any], validate_order: bool = True
    ) -> None:
        """Validate the methods in the dependency tree.

        Currently check the order of methods and whether they have the correct parameters.

        Args:
            methods: A dictionary of methods to validate.
            validate_order: If True, checks if methods depend on other methods that may come after them in the build file.

        Raises:
            ValueError: If a method depends on another method that is not a build function
            ValueError: If a method depends on another method that may come after it in the build file or not be present at all.
            ValueError: If a method has parameters that do not match the expected parameters.
        """
        for method in methods:
            if not self.tree.has_node(method):
                raise ValueError(
                    f"Method {method} is not a build function.  If you are sure this should be build method, please decorate it with @build_method.'"
                )

            node = self.tree.nodes[method]
            if not node.get("attr", {}).get("_function_exists", False):
                raise ValueError(f"Method {method} is not a build function.")

        if validate_order:
            processed_methods = set()
            for method in methods:
                direct_dependencies = self.get_dependencies(
                    method, depth_limit=2
                )  # 1 is the method itself
                for direct_dependency in direct_dependencies:
                    if direct_dependency == method:
                        continue
                    if direct_dependency not in processed_methods:
                        if direct_dependency not in self.methods:
                            raise ValueError(
                                f"Method {method} depends on {direct_dependency}, "
                                "which is not a build function."
                            )
                        else:
                            raise ValueError(
                                f"Method {method} depends on {direct_dependency}, "
                                "which may come after this method in the build file or not be present at all."
                            )
                processed_methods.add(method)

        for method, args in methods.items():
            args_set = set(args) if args is not None else set()

            required_parameters = self.tree.nodes[method]["attr"][
                "_required_parameters"
            ]
            if not set(required_parameters).issubset(args_set):
                raise ValueError(
                    f"Method {method} has parameters {list(args.keys())}, "
                    f"but expected parameters are {required_parameters}."
                )

            optional_parameters = self.tree.nodes[method]["attr"][
                "_optional_parameters"
            ]
            if not set(args_set).issubset(
                set(required_parameters + optional_parameters)
            ):
                raise ValueError(
                    f"Method {method} has parameters {list(args.keys())}, "
                    f"but required parameters are {required_parameters} and "
                    f"optional parameters are {optional_parameters}."
                )

    def export_tree(self) -> None:
        pos = nx.spring_layout(self.tree)
        nx.draw(self.tree, pos, with_labels=True, arrows=True)
        plt.savefig("dependency_graph.png")

    def get_dependencies(
        self, method: str, depth_limit: None | int = None
    ) -> list[str]:
        """Get all dependencies for a given method.

        Args:
            method: The method for which to find dependencies.
            depth_limit: Optional depth limit for the search.

        Returns:
            A list of methods that the specified method depends on.
        """
        # [:-1] is used to skip the method itself in the result
        return list(
            nx.dfs_postorder_nodes(self.tree.reverse(), method, depth_limit=depth_limit)
        )[:-1]

    def get_dependents(self, method: str, depth_limit: None | int = None) -> list[str]:
        """Get all methods that depend on a given method.

        Args:
            method: The method for which to find dependents.
            depth_limit: Optional depth limit for the search.

        Returns:
            A list of methods that depend on the specified method.
        """
        # [1:] is used to skip the method itself in the result
        return list(nx.dfs_preorder_nodes(self.tree, method, depth_limit=depth_limit))[
            1:
        ]

    def record_progress(self, progress_path: Path, method: str) -> None:
        """Record progress to txt progress file.

        Args:
            progress_path: Path to the progress file.
            method: Method that has been completed.
        """
        with open(progress_path, "a") as f:
            f.write(f"{method}\n")

    def read_progress(self, progress_path: Path) -> list[str]:
        """Get the list of methods that have been completed from the progress file.

        Args:
            progress_path: Path to the progress file.

        Returns:
            A list of methods that have been completed.
        """
        if not progress_path.exists():
            return []
        with open(progress_path, "r") as f:
            completed_methods = f.read().splitlines()
        return completed_methods

    def log_time_taken(self) -> None:
        """Log the time taken for each method in the dependency tree."""
        for method, time_taken in self.time_taken.items():
            self.logger.info(f"Method {method} took {time_taken:.2f} seconds.")

    @property
    def methods(self) -> list[str]:
        """Return the methods in the dependency tree, sorted alphabetically.

        Returns:
            A alphabetically sorted list of method names in the dependency tree.
        """
        return sorted(list(self.tree.nodes))


build_method = _build_method(logger=logger)
