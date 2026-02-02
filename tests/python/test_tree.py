"""Tests for tree conversion and manipulation."""

import pytest
import numpy as np
import pylaml


class TestMakeTree:
    """Tests for pylaml.make_tree helper function."""

    def test_basic_tree_creation(self):
        """Test creating a simple tree."""
        tree = pylaml.make_tree(
            edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
            branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
            num_leaves=3
        )

        assert tree["num_leaves"] == 3
        assert tree["num_nodes"] == 5
        assert tree["root"] == 4
        assert len(tree["edges"]) == 4
        assert len(tree["branch_lengths"]) == 5
        assert len(tree["node_names"]) == 5

    def test_root_inference(self):
        """Test that root is correctly inferred when not provided."""
        tree = pylaml.make_tree(
            edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
            branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
            num_leaves=3
        )
        # Node 4 is the only node with no incoming edge
        assert tree["root"] == 4

    def test_explicit_root(self):
        """Test providing explicit root."""
        tree = pylaml.make_tree(
            edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
            branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
            num_leaves=3,
            root=4
        )
        assert tree["root"] == 4

    def test_custom_node_names(self):
        """Test providing custom node names."""
        names = ["A", "B", "C", "D", "E"]
        tree = pylaml.make_tree(
            edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
            branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
            num_leaves=3,
            node_names=names
        )
        assert tree["node_names"] == names


class TestTreeRoundTrip:
    """Test that trees survive round-trip through optimization."""

    def test_tree_structure_preserved(self, simple_tree, simple_character_matrix):
        """Test that tree structure is preserved after optimization."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            max_iterations=1
        )

        opt_tree = result.optimized_tree

        # Structure should be preserved
        assert opt_tree["num_leaves"] == simple_tree["num_leaves"]
        assert opt_tree["num_nodes"] == simple_tree["num_nodes"]
        assert opt_tree["root"] == simple_tree["root"]
        assert set(tuple(e) for e in opt_tree["edges"]) == set(tuple(e) for e in simple_tree["edges"])

    def test_node_names_preserved(self, simple_tree, simple_character_matrix):
        """Test that node names are preserved after optimization."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            max_iterations=1
        )

        assert result.optimized_tree["node_names"] == simple_tree["node_names"]


class TestProjectUltrametric:
    """Tests for ultrametric projection."""

    def test_basic_projection(self, simple_tree):
        """Test that projection produces valid output."""
        projected = pylaml.project_ultrametric(simple_tree)

        assert projected["num_leaves"] == simple_tree["num_leaves"]
        assert projected["num_nodes"] == simple_tree["num_nodes"]
        assert projected["root"] == simple_tree["root"]
        assert len(projected["branch_lengths"]) == simple_tree["num_nodes"]

    def test_branch_lengths_non_negative(self, simple_tree):
        """Test that projected branch lengths are non-negative."""
        projected = pylaml.project_ultrametric(simple_tree)

        for bl in projected["branch_lengths"]:
            assert bl >= 0

    def test_ultrametric_property(self, simple_tree):
        """Test that projected tree satisfies ultrametric property."""
        projected = pylaml.project_ultrametric(simple_tree)

        # Compute root-to-leaf distances
        def compute_distance(node, root, edges, branch_lengths):
            # Simple BFS to find path from root to node
            parent = {root: None}
            for p, c in edges:
                parent[c] = p

            dist = 0.0
            current = node
            while current != root:
                dist += branch_lengths[current]
                current = parent[current]
            dist += branch_lengths[root]
            return dist

        distances = []
        for leaf in range(projected["num_leaves"]):
            d = compute_distance(
                leaf,
                projected["root"],
                projected["edges"],
                projected["branch_lengths"]
            )
            distances.append(d)

        # All distances should be approximately equal
        assert len(distances) > 0
        for d in distances:
            assert abs(d - distances[0]) < 1e-4, f"Distances not equal: {distances}"
