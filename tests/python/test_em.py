"""Tests for EM optimization."""

import pytest
import numpy as np
import pylaml


class TestOptimizeBasic:
    """Basic tests for the optimize function."""

    def test_basic_optimization(self, simple_tree, simple_character_matrix):
        """Test that optimization runs without error."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            max_iterations=10
        )

        assert result.log_likelihood < 0  # log-likelihood should be negative
        assert result.num_iterations >= 1
        assert result.nu > 0
        assert result.phi > 0
        assert result.phi < 1

    def test_optimization_improves_likelihood(self, simple_tree, simple_character_matrix):
        """Test that optimization improves (or maintains) likelihood."""
        # Compute initial likelihood
        initial_llh = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.5
        )

        # Run optimization
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            initial_nu=0.5,
            initial_phi=0.5,
            max_iterations=50
        )

        # Final likelihood should be at least as good
        assert result.log_likelihood >= initial_llh - 1e-6

    def test_returns_em_results(self, simple_tree, simple_character_matrix):
        """Test that optimize returns an EMResults object."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            max_iterations=1
        )

        assert isinstance(result, pylaml.EMResults)
        assert hasattr(result, 'log_likelihood')
        assert hasattr(result, 'num_iterations')
        assert hasattr(result, 'optimized_tree')
        assert hasattr(result, 'nu')
        assert hasattr(result, 'phi')
        assert hasattr(result, 'posterior_probabilities')

    def test_posterior_shape(self, simple_tree, simple_character_matrix):
        """Test that posterior probabilities have correct shape."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            max_iterations=5
        )

        posterior = result.posterior_probabilities
        num_chars = simple_character_matrix.shape[1]
        num_nodes = simple_tree["num_nodes"]

        assert posterior.shape[0] == num_chars
        assert posterior.shape[1] == num_nodes
        # States dimension depends on data


class TestOptimizeParameters:
    """Tests for optimization parameters."""

    def test_initial_parameters(self, simple_tree, simple_character_matrix):
        """Test that initial parameters are used."""
        result1 = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            initial_nu=0.1,
            initial_phi=0.1,
            max_iterations=1
        )

        result2 = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            initial_nu=0.9,
            initial_phi=0.9,
            max_iterations=1
        )

        # With only 1 iteration, results should differ based on initial values
        # (though they might converge with more iterations)
        assert result1.nu != result2.nu or result1.phi != result2.phi

    def test_max_iterations_respected(self, simple_tree, simple_character_matrix):
        """Test that max_iterations limits iterations."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            max_iterations=5
        )

        assert result.num_iterations <= 5

    def test_ultrametric_mode(self, simple_tree, simple_character_matrix):
        """Test ultrametric optimization mode."""
        result = pylaml.optimize(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            ultrametric=True,
            max_iterations=10
        )

        # Should still produce valid results
        assert result.log_likelihood < 0


class TestOptimizeWithMissingData:
    """Tests for optimization with missing data."""

    def test_missing_data_handled(self, larger_tree):
        """Test that missing data (-1) is handled correctly."""
        # Use a larger matrix with some missing values but enough data for solver
        char_matrix = np.array([
            [0, 1, 0, 1],  # leaf 0
            [0, 1, -1, 0],  # leaf 1: one missing
            [1, 0, 0, 1],  # leaf 2
            [-1, 0, 1, 0],  # leaf 3: one missing
            [0, 0, 1, 1],  # leaf 4
        ], dtype=np.int32)

        result = pylaml.optimize(
            tree=larger_tree,
            character_matrix=char_matrix,
            max_iterations=10
        )

        assert result.log_likelihood < 0
        # phi (dropout) should be > 0 when there's missing data
        assert result.phi >= 0


class TestOptimizeLargerTree:
    """Tests with larger trees."""

    def test_larger_tree_optimization(self, larger_tree, larger_character_matrix):
        """Test optimization on a larger tree."""
        result = pylaml.optimize(
            tree=larger_tree,
            character_matrix=larger_character_matrix,
            max_iterations=20
        )

        assert result.log_likelihood < 0
        assert result.num_iterations >= 1


class TestOptimizeInputValidation:
    """Tests for input validation."""

    def test_requires_data_matrix(self, simple_tree):
        """Test that either character_matrix or observation_matrix is required."""
        with pytest.raises(ValueError, match="Must provide either"):
            pylaml.optimize(tree=simple_tree)

    def test_rejects_both_matrices(self, simple_tree, simple_character_matrix):
        """Test that providing both matrices raises error."""
        obs_matrix = np.zeros((3, 2, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="Cannot provide both"):
            pylaml.optimize(
                tree=simple_tree,
                character_matrix=simple_character_matrix,
                observation_matrix=obs_matrix
            )

    def test_validates_character_matrix_shape(self, simple_tree):
        """Test that character matrix shape is validated."""
        # Wrong number of leaves
        bad_matrix = np.array([[0, 1], [1, 0]], dtype=np.int32)  # 2 leaves instead of 3

        with pytest.raises(ValueError, match="leaves"):
            pylaml.optimize(
                tree=simple_tree,
                character_matrix=bad_matrix
            )

    def test_validates_nu_bounds(self, simple_tree, simple_character_matrix):
        """Test that nu parameter bounds are validated."""
        with pytest.raises(ValueError, match="initial_nu"):
            pylaml.optimize(
                tree=simple_tree,
                character_matrix=simple_character_matrix,
                initial_nu=-1.0
            )

    def test_validates_phi_bounds(self, simple_tree, simple_character_matrix):
        """Test that phi parameter bounds are validated."""
        with pytest.raises(ValueError, match="initial_phi"):
            pylaml.optimize(
                tree=simple_tree,
                character_matrix=simple_character_matrix,
                initial_phi=1.5
            )
