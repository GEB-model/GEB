import logging

import networkx as nx
import pytest

from geb.build.methods import validate_build_methods


class MockLogger(logging.Logger):
    def __init__(self, name: str = "test"):
        super().__init__(name)
        self.warnings = []
        self.infos = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)


@pytest.fixture
def logger():
    return MockLogger()


def create_tree(nodes_with_params, edges=None):
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


def test_validate_methods_no_fix_needed(logger):
    tree = create_tree({"a": ([], []), "b": ([], [])}, edges=[("a", "b")])
    methods = {"a": {}, "b": {}}

    result = validate_build_methods(
        tree, methods, fix_order_if_broken=True, logger=logger
    )

    assert list(result.keys()) == ["a", "b"]
    assert len(logger.warnings) == 0


def test_validate_methods_fix_minimal_move(logger):
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


def test_validate_methods_deterministic(logger):
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


def test_validate_methods_circular_dependency(logger):
    tree = create_tree({"a": ([], []), "b": ([], [])}, edges=[("a", "b"), ("b", "a")])
    methods = {"a": {}, "b": {}}

    with pytest.raises(ValueError, match="cycle detected"):
        validate_build_methods(tree, methods, fix_order_if_broken=True, logger=logger)


def test_validate_methods_required_parameters_check(logger):
    tree = create_tree({"a": (["param1"], [])})
    methods = {"a": {}}  # Missing param1

    with pytest.raises(ValueError, match="expected parameters are \['param1'\]"):
        validate_build_methods(tree, methods, fix_order_if_broken=True, logger=logger)
