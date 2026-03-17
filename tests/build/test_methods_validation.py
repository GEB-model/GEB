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


def test_validate_methods_no_fix_needed(logger: MockLogger) -> None:
    """Test that no fix is applied when order is correct."""
    tree = create_tree({"a": ([], []), "b": ([], [])}, edges=[("a", "b")])
    methods = {"a": {}, "b": {}}

    result = validate_build_methods(
        tree, methods, fix_order_if_broken=True, logger=logger
    )

    assert list(result.keys()) == ["a", "b"]
    assert len(logger.warnings) == 0


def test_validate_methods_fix_minimal_move(logger: MockLogger) -> None:
    """Test that minimal moves are applied to fix broken order."""
    # Dependencies: a -> b -> c (c depends on b, b depends on a)
    tree = create_tree(
        {"a": ([], []), "b": ([], []), "c": ([], [])}, edges=[("a", "b"), ("b", "c")]
    )

    # Broken order: c depends on b, but c is first. b depends on a, but a is second.
    methods = {"c": {}, "a": {}, "b": {}}

    result = validate_build_methods(
        tree, methods, fix_order_if_broken=True, logger=logger
    )

    # Minimal move logic: [a, b, c]
    assert list(result.keys()) == ["a", "b", "c"]
    assert any("has been auto-fixed" in w for w in logger.warnings)
    assert any("New method order: ['a', 'b', 'c']" in i for i in logger.infos)


def test_validate_methods_deterministic(logger: MockLogger) -> None:
    """Test that the fix is deterministic for independent chains."""
    # Independent pairs: {a, b} and {c, d}
    tree = create_tree(
        {"a": ([], []), "b": ([], []), "c": ([], []), "d": ([], [])},
        edges=[("a", "b"), ("c", "d")],
    )

    methods = {"b": {}, "a": {}, "d": {}, "c": {}}
    result = validate_build_methods(
        tree, methods, fix_order_if_broken=True, logger=logger
    )

    assert list(result.keys()) == ["a", "b", "c", "d"]


def test_validate_methods_circular_dependency(logger: MockLogger) -> None:
    """Test that circular dependencies are detected."""
    tree = create_tree({"a": ([], []), "b": ([], [])}, edges=[("a", "b"), ("b", "a")])
    methods = {"a": {}, "b": {}}

    with pytest.raises(ValueError, match="cycle detected"):
        validate_build_methods(tree, methods, fix_order_if_broken=True, logger=logger)


def test_validate_methods_required_parameters_check(logger: MockLogger) -> None:
    """Test that parameter validation still occurs after fixing order."""
    tree = create_tree({"a": (["param1"], [])})
    methods = {"a": {}}  # Missing param1

    with pytest.raises(ValueError, match=r"expected parameters are \['param1'\]"):
        validate_build_methods(tree, methods, fix_order_if_broken=True, logger=logger)
