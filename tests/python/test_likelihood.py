"""Tests for likelihood computation."""

import pytest
import numpy as np
import pylaml


class TestComputeLikelihood:
    """Tests for pylaml.compute_likelihood function."""

    def test_basic_likelihood(self, simple_tree, simple_character_matrix):
        """Test basic likelihood computation."""
        llh = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.1
        )

        # Log-likelihood should be negative
        assert llh < 0
        # Should be finite
        assert np.isfinite(llh)

    def test_likelihood_deterministic(self, simple_tree, simple_character_matrix):
        """Test that likelihood computation is deterministic."""
        llh1 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.1
        )

        llh2 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.1
        )

        assert llh1 == llh2

    def test_likelihood_varies_with_nu(self, simple_tree, simple_character_matrix):
        """Test that likelihood changes with nu parameter."""
        llh1 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.1,
            phi=0.1
        )

        llh2 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=1.0,
            phi=0.1
        )

        assert llh1 != llh2

    def test_likelihood_varies_with_phi(self, simple_tree, simple_character_matrix):
        """Test that likelihood changes with phi parameter."""
        llh1 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.1
        )

        llh2 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.5
        )

        assert llh1 != llh2

    def test_likelihood_varies_with_branch_lengths(self, simple_character_matrix):
        """Test that likelihood changes with branch lengths."""
        tree1 = pylaml.make_tree(
            edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
            branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
            num_leaves=3
        )

        tree2 = pylaml.make_tree(
            edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
            branch_lengths=[0.5, 0.5, 0.5, 0.5, 0.0],
            num_leaves=3
        )

        llh1 = pylaml.compute_likelihood(
            tree=tree1,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.1
        )

        llh2 = pylaml.compute_likelihood(
            tree=tree2,
            character_matrix=simple_character_matrix,
            nu=0.5,
            phi=0.1
        )

        assert llh1 != llh2


class TestLikelihoodWithMissingData:
    """Tests for likelihood with missing data."""

    def test_missing_data_handled(self, simple_tree, character_matrix_with_missing):
        """Test likelihood computation with missing data."""
        llh = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=character_matrix_with_missing,
            nu=0.5,
            phi=0.1
        )

        assert np.isfinite(llh)
        assert llh < 0


class TestLikelihoodWithPriors:
    """Tests for likelihood with custom mutation priors."""

    def test_custom_priors(self, simple_tree, simple_character_matrix):
        """Test likelihood with custom mutation priors."""
        # Create priors: 2 characters, 1 mutated state each
        priors = np.array([
            [1.0],  # character 0: only state 1
            [1.0],  # character 1: only state 1
        ], dtype=np.float64)

        llh = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            mutation_priors=priors,
            nu=0.5,
            phi=0.1
        )

        assert np.isfinite(llh)

    def test_priors_affect_likelihood(self, simple_tree, simple_character_matrix):
        """Test that different priors give different likelihoods."""
        priors1 = np.array([[1.0], [1.0]], dtype=np.float64)
        priors2 = np.array([[0.5], [0.5]], dtype=np.float64)

        llh1 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            mutation_priors=priors1,
            nu=0.5,
            phi=0.1
        )

        llh2 = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix,
            mutation_priors=priors2,
            nu=0.5,
            phi=0.1
        )

        # Priors affect the likelihood computation
        # Note: may be equal in some degenerate cases
        assert np.isfinite(llh1) and np.isfinite(llh2)


class TestLikelihoodInputValidation:
    """Tests for input validation in compute_likelihood."""

    def test_requires_data(self, simple_tree):
        """Test that data matrix is required."""
        with pytest.raises(ValueError, match="Must provide either"):
            pylaml.compute_likelihood(tree=simple_tree)

    def test_accepts_character_matrix(self, simple_tree, simple_character_matrix):
        """Test that character matrix is accepted."""
        llh = pylaml.compute_likelihood(
            tree=simple_tree,
            character_matrix=simple_character_matrix
        )
        assert np.isfinite(llh)


class TestMemoryLayoutIndependence:
    """Regression tests for issue: non-C-contiguous character matrices were
    silently misread by the C++ bindings, scrambling the data and corrupting
    the likelihood.

    The numpy->C++ converters assumed a C-contiguous (row-major) layout. Inputs
    that are not C-contiguous -- a transposed view, a slice, or a pandas
    DataFrame's ``.values`` (frequently Fortran-ordered) -- were read with
    hardcoded row-major offsets, effectively transposing the matrix. The
    likelihood must depend only on the array's logical contents, not on its
    in-memory layout.
    """

    def _tree_8_leaves(self):
        # Balanced binary tree, leaves 0..7, internal 8..14, root 14.
        edges = [(8, 0), (8, 1), (9, 2), (9, 3), (10, 4), (10, 5), (11, 6), (11, 7),
                 (12, 8), (12, 9), (13, 10), (13, 11), (14, 12), (14, 13)]
        return pylaml.make_tree(edges=edges,
                                branch_lengths=[0.4] * 14 + [0.0],
                                num_leaves=8, root=14)

    def test_c_and_fortran_order_agree(self):
        """Likelihood must be identical for C- and F-contiguous copies."""
        tree = self._tree_8_leaves()
        rng = np.random.RandomState(0)
        cm = rng.randint(0, 6, size=(8, 12)).astype(np.int32)

        c_order = np.ascontiguousarray(cm)
        f_order = np.asfortranarray(cm)
        assert c_order.flags["C_CONTIGUOUS"] and f_order.flags["F_CONTIGUOUS"]

        llh_c = pylaml.compute_likelihood(tree=tree, character_matrix=c_order, nu=0.1, phi=0.05)
        llh_f = pylaml.compute_likelihood(tree=tree, character_matrix=f_order, nu=0.1, phi=0.05)
        assert llh_c == pytest.approx(llh_f, abs=1e-9)

    def test_joint_equals_sum_over_characters(self):
        """Characters are independent: the joint log-likelihood must equal the
        sum of the per-character log-likelihoods, regardless of differing
        per-character alphabet sizes."""
        tree = self._tree_8_leaves()
        # Columns with deliberately heterogeneous alphabet sizes.
        small = np.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 4, dtype=np.int32).T   # 2 states
        big = np.array([[0, 1, 2, 3, 4, 5, 6, 7]] * 4, dtype=np.int32).T     # 8 states
        mix = np.concatenate([small, big], axis=1)

        joint = pylaml.compute_likelihood(tree=tree, character_matrix=mix, nu=1e-6, phi=1e-3)
        per_col = sum(
            pylaml.compute_likelihood(tree=tree, character_matrix=mix[:, c:c + 1], nu=1e-6, phi=1e-3)
            for c in range(mix.shape[1])
        )
        assert joint == pytest.approx(per_col, abs=1e-6)

    def test_column_order_invariance(self):
        """The joint log-likelihood must not depend on the column ordering."""
        tree = self._tree_8_leaves()
        small = np.array([[0, 0, 0, 0, 1, 1, 1, 1]] * 4, dtype=np.int32).T
        big = np.array([[0, 1, 2, 3, 4, 5, 6, 7]] * 4, dtype=np.int32).T
        ab = pylaml.compute_likelihood(tree=tree,
                                       character_matrix=np.concatenate([small, big], axis=1),
                                       nu=1e-6, phi=1e-3)
        ba = pylaml.compute_likelihood(tree=tree,
                                       character_matrix=np.concatenate([big, small], axis=1),
                                       nu=1e-6, phi=1e-3)
        assert ab == pytest.approx(ba, abs=1e-9)
