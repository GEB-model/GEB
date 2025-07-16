import functools
import logging
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger("GEB")

__all__ = ["build_method"]


class _build_method:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.tree = nx.DiGraph()

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

            self.add_tree_node(func.__name__)

            if depends_on is not None:
                if isinstance(depends_on, str):
                    self.add_tree_edge(func.__name__, depends_on)
                elif isinstance(depends_on, list):
                    for item in depends_on:
                        self.add_tree_edge(func.__name__, item)
                else:
                    raise ValueError("depends_on must be a string or a list of strings")

            wrapper.__is_build_method__ = True
            return wrapper

        if func is None:
            return partial_decorator
        else:
            f = partial_decorator(func)
            return f

    def add_tree_node(self, func: str):
        """Add a node to the dependency tree."""
        self.tree.add_node(func, attr={"function_exists": True})

    def add_tree_edge(self, func: str, depends_on: str):
        """Add an edge to the dependency tree."""
        if depends_on == "setup_region":
            raise ValueError(
                "Everything depends on setup_region so we don't include it."
            )
        self.tree.add_edge(depends_on, func)

    def validate_tree(self):
        """Validate the dependency tree.

        Checks if all the node dependencies are present in the tree.
        """
        assert nx.is_directed_acyclic_graph(self.tree)
        for node in self.tree.nodes:
            depencencies = list(self.tree.predecessors(node))
            for dependency in depencencies:
                if (
                    not self.tree.nodes[dependency]
                    .get("attr", {})
                    .get("function_exists", False)
                ):
                    raise ValueError(
                        f"Node {node} depends on {dependency}, "
                        "which is not a build function."
                    )
        self.logger.debug("Builder dependency tree validation passed.")

    def export_tree(self):
        pos = nx.spring_layout(self.tree)
        nx.draw(self.tree, pos, with_labels=True, arrows=True)
        plt.savefig("dependency_graph.png")

    def get_dependencies(self, method: str) -> list[str]:
        """Get all dependencies for a given method."""
        return list(nx.dfs_postorder_nodes(self.tree.reverse(), method))

    def get_dependents(self, method: str) -> list[str]:
        """Get all methods that depend on a given method."""
        dependents = []
        for _, successors in nx.bfs_successors(self.tree, method):
            dependents.extend(successors)
        return dependents

    @property
    def methods(self):
        """Return the methods in the dependency tree."""
        return sorted(list(self.tree.nodes))


build_method = _build_method(logger=logger)
