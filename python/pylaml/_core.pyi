"""Type stubs for pylaml._core C++ extension module."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt

__version__: str

BRANCH_LENGTH_LB: float
BRANCH_LENGTH_UB: float
NU_LB: float
NU_UB: float
PHI_LB: float
PHI_UB: float


class EMResults:
    """Results from EM optimization."""

    @property
    def log_likelihood(self) -> float:
        """Final log-likelihood value."""
        ...

    @property
    def num_iterations(self) -> int:
        """Number of EM iterations performed."""
        ...

    @property
    def optimized_tree(self) -> Dict:
        """Tree dict with optimized branch lengths."""
        ...

    @property
    def nu(self) -> float:
        """Optimized nu parameter (mutation rate)."""
        ...

    @property
    def phi(self) -> float:
        """Optimized phi parameter (dropout rate)."""
        ...

    @property
    def posterior_probabilities(self) -> npt.NDArray[np.float64]:
        """Posterior probabilities array (characters, nodes, states)."""
        ...


def optimize(
    tree: Dict,
    character_matrix: Optional[npt.NDArray[np.int32]] = None,
    observation_matrix: Optional[npt.NDArray[np.float64]] = None,
    mutation_priors: Optional[npt.NDArray[np.float64]] = None,
    initial_nu: float = 0.5,
    initial_phi: float = 0.5,
    ultrametric: bool = False,
    max_iterations: int = 100,
    min_branch_length: float = 0.01,
    verbose: bool = False
) -> EMResults:
    """Run EM optimization on a phylogenetic tree.

    Parameters
    ----------
    tree : dict
        Tree structure with keys: num_leaves, num_nodes, root, edges, branch_lengths
    character_matrix : numpy.ndarray, optional
        Character state matrix with shape (leaves, characters), dtype int32.
    observation_matrix : numpy.ndarray, optional
        Observation probability matrix with shape (leaves, characters, states).
    mutation_priors : numpy.ndarray, optional
        Prior probabilities for mutations.
    initial_nu : float
        Initial value for nu parameter.
    initial_phi : float
        Initial value for phi parameter.
    ultrametric : bool
        Whether to constrain tree to be ultrametric.
    max_iterations : int
        Maximum number of EM iterations.
    min_branch_length : float
        Minimum branch length fraction.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    EMResults
        Object containing optimization results.
    """
    ...


def compute_likelihood(
    tree: Dict,
    character_matrix: Optional[npt.NDArray[np.int32]] = None,
    observation_matrix: Optional[npt.NDArray[np.float64]] = None,
    mutation_priors: Optional[npt.NDArray[np.float64]] = None,
    nu: float = 0.5,
    phi: float = 0.5
) -> float:
    """Compute log-likelihood of a tree given character data.

    Parameters
    ----------
    tree : dict
        Tree structure.
    character_matrix : numpy.ndarray, optional
        Character state matrix.
    observation_matrix : numpy.ndarray, optional
        Observation probability matrix.
    mutation_priors : numpy.ndarray, optional
        Prior probabilities for mutations.
    nu : float
        Nu parameter.
    phi : float
        Phi parameter.

    Returns
    -------
    float
        Log-likelihood value.
    """
    ...


def project_ultrametric(tree: Dict) -> Dict:
    """Project tree branch lengths to satisfy ultrametric constraint.

    Parameters
    ----------
    tree : dict
        Tree structure.

    Returns
    -------
    dict
        Tree with projected branch lengths.
    """
    ...
