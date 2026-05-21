"""Tests for build method validation logic."""

import logging
from typing import Any

import networkx as nx
import pytest

from geb.build.methods import validate_build_methods


class MockLogger(logging.Logger):
    """Mock logger for capturing warnings and info messages."""

    def __init__(self, name: str = "test") -> None:
        """Initialize MockLogger.

        Args:
            name: Logger name.
        """
        super().__init__(name)
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Capture warning message.

        Args:
            msg: The message.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        self.warnings.append(msg)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Capture info message.

        Args:
            msg: The message.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        self.infos.append(msg)


@pytest.fixture
def logger() -> MockLogger:
    """Fixture for MockLogger.

    Returns:
        The mock logger instance.
    """
    return MockLogger()


def create_tree(
    nodes_with_params: dict[str, tuple[list[str], list[str]]],
    edges: list[tuple[str, str]] | None = None,
) -> nx.DiGraph:
    """Helper to create a dependency tree for testing.

    Args:
        nodes_with_params: Dict of node names to (required, optional) params.
        edges: Optional list of edges.

    Returns:
        The created DiGraph.
    """
    tree = nx.DiGraph()
    for node, (req_params, opt_params) in nodes_with_params.items():
        tree.add_node(
            node,
            attr={
                "_function_exists": True,
                "_required_parameters": req_params,
                "_optional_parameters": opt_params,
            },
        )
    if edges:
        for u, v in edges:
            tree.add_edge(u, v)
    return tree


def test_validate_methods_circular_dependency(logger: MockLogger) -> None:
    """Test that circular dependencies are detected."""
    tree = create_tree({"a": ([], []), "b": ([], [])}, edges=[("a", "b"), ("b", "a")])
    methods = {"a": {}, "b": {}}

    with pytest.raises(
        ValueError,
        match="Method a depends on b, which may come after this method in the build file or not be present at all.",
    ):
        validate_build_methods(tree, methods, logger=logger)


def test_validate_methods_required_parameters_check(logger: MockLogger) -> None:
    """Test that parameter validation still occurs after fixing order."""
    tree = create_tree({"a": (["param1"], [])})
    methods = {"a": {}}  # Missing param1

    with pytest.raises(ValueError, match=r"expected parameters are \['param1'\]"):
        validate_build_methods(tree, methods, logger=logger)


def test_validate_methods_missing_dependency(logger: MockLogger) -> None:
    """Test that missing dependencies are detected even when fixing order."""
    # b depends on a, but a is missing from methods dict
    tree = create_tree({"a": ([], []), "b": ([], [])}, edges=[("a", "b")])
    methods = {"b": {}}

    with pytest.raises(ValueError, match="is missing from the requested methods"):
        validate_build_methods(tree, methods, logger=logger)
