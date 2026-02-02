"""Pytest configuration and shared fixtures for pylaml tests."""

import pytest
import numpy as np


@pytest.fixture
def simple_tree():
    """A simple tree with 3 leaves and 5 nodes.

    Tree structure:
           4 (root)
          / \\
         3   2
        / \\
       0   1
    """
    return {
        "num_leaves": 3,
        "num_nodes": 5,
        "root": 4,
        "edges": [(4, 3), (3, 0), (3, 1), (4, 2)],
        "branch_lengths": [0.1, 0.1, 0.2, 0.1, 0.0],
        "node_names": ["leaf0", "leaf1", "leaf2", "internal", "root"]
    }


@pytest.fixture
def simple_character_matrix():
    """A simple character matrix for 3 leaves and 2 characters."""
    return np.array([
        [0, 1],  # leaf 0
        [1, 0],  # leaf 1
        [0, 1],  # leaf 2
    ], dtype=np.int32)


@pytest.fixture
def character_matrix_with_missing():
    """Character matrix with missing data (-1)."""
    return np.array([
        [0, -1],  # leaf 0: character 1 missing
        [1, 0],   # leaf 1
        [-1, 1],  # leaf 2: character 0 missing
    ], dtype=np.int32)


@pytest.fixture
def larger_tree():
    """A larger tree with 5 leaves and 9 nodes.

    Tree structure:
              8 (root)
             / \\
            7   4
           / \\
          6   3
         / \\
        5   2
       / \\
      0   1
    """
    return {
        "num_leaves": 5,
        "num_nodes": 9,
        "root": 8,
        "edges": [
            (8, 7), (8, 4),
            (7, 6), (7, 3),
            (6, 5), (6, 2),
            (5, 0), (5, 1)
        ],
        "branch_lengths": [0.05, 0.05, 0.1, 0.15, 0.3, 0.1, 0.1, 0.2, 0.0],
        "node_names": ["l0", "l1", "l2", "l3", "l4", "i1", "i2", "i3", "root"]
    }


@pytest.fixture
def larger_character_matrix():
    """Character matrix for 5 leaves and 3 characters."""
    return np.array([
        [0, 1, 0],  # leaf 0
        [0, 1, 1],  # leaf 1
        [1, 0, 0],  # leaf 2
        [1, 0, 1],  # leaf 3
        [0, 0, 1],  # leaf 4
    ], dtype=np.int32)
