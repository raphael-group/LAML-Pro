"""
pylaml - Python bindings for fastLAML

Fast Lineage-Aware Maximum Likelihood for phylogenetic tree optimization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from . import _core

__version__ = _core.__version__

# Re-export constants
BRANCH_LENGTH_LB = _core.BRANCH_LENGTH_LB
BRANCH_LENGTH_UB = _core.BRANCH_LENGTH_UB
NU_LB = _core.NU_LB
NU_UB = _core.NU_UB
PHI_LB = _core.PHI_LB
PHI_UB = _core.PHI_UB


class EMResults:
    """Results from EM optimization.

    Attributes
    ----------
    log_likelihood : float
        Final log-likelihood value.
    num_iterations : int
        Number of EM iterations performed.
    optimized_tree : dict
        Tree dictionary with optimized branch lengths.
    nu : float
        Optimized nu parameter (mutation rate).
    phi : float
        Optimized phi parameter (dropout rate).
    posterior_probabilities : numpy.ndarray
        Posterior probabilities with shape (characters, nodes, states).
    """

    def __init__(self, core_result: _core.EMResults):
        self._result = core_result

    @property
    def log_likelihood(self) -> float:
        return self._result.log_likelihood

    @property
    def num_iterations(self) -> int:
        return self._result.num_iterations

    @property
    def optimized_tree(self) -> Dict:
        return self._result.optimized_tree

    @property
    def nu(self) -> float:
        return self._result.nu

    @property
    def phi(self) -> float:
        return self._result.phi

    @property
    def posterior_probabilities(self) -> np.ndarray:
        return self._result.posterior_probabilities

    def __repr__(self) -> str:
        return (
            f"EMResults(log_likelihood={self.log_likelihood:.4f}, "
            f"num_iterations={self.num_iterations}, "
            f"nu={self.nu:.4f}, phi={self.phi:.4f})"
        )


def recode_character_matrix(char_matrix: np.ndarray) -> np.ndarray:
    """Recode character states to be contiguous (0, 1, 2, ...).

    The C++ LAML implementation expects character states to be contiguous
    integers starting at 0. This function remaps non-contiguous states
    (e.g., {0, 3, 15, 27}) to contiguous ones ({0, 1, 2, 3}).

    State 0 (unmutated) is always mapped to 0. Missing data (-1) is preserved.

    Parameters
    ----------
    char_matrix : numpy.ndarray
        Character matrix with shape (leaves, characters), dtype int32.
        Values are character states, with -1 indicating missing data.

    Returns
    -------
    numpy.ndarray
        Recoded character matrix with contiguous states per character.

    Examples
    --------
    >>> matrix = np.array([[0, 3], [0, 15], [1, 0]], dtype=np.int32)
    >>> recoded = recode_character_matrix(matrix)
    >>> # Column 0: {0, 1} -> {0, 1}
    >>> # Column 1: {0, 3, 15} -> {0, 1, 2}
    """
    n_leaves, n_chars = char_matrix.shape
    recoded = np.zeros_like(char_matrix)

    for c in range(n_chars):
        col = char_matrix[:, c]
        # Get unique states excluding missing (-1), sorted
        unique_states = sorted(set(col) - {-1})

        # Build mapping: 0 always maps to 0, then others in sorted order
        state_map = {0: 0}
        next_idx = 1
        for s in unique_states:
            if s != 0 and s not in state_map:
                state_map[s] = next_idx
                next_idx += 1

        # Apply mapping
        for i in range(n_leaves):
            if col[i] == -1:
                recoded[i, c] = -1
            elif col[i] in state_map:
                recoded[i, c] = state_map[col[i]]
            else:
                # Unexpected state - treat as missing
                recoded[i, c] = -1

    return recoded


def make_tree(
    edges: List[Tuple[int, int]],
    branch_lengths: List[float],
    num_leaves: int,
    node_names: Optional[List[str]] = None,
    root: Optional[int] = None
) -> Dict:
    """Create a tree dictionary from edges and branch lengths.

    Parameters
    ----------
    edges : list of (int, int)
        List of (parent, child) edge tuples.
    branch_lengths : list of float
        Branch lengths indexed by node.
    num_leaves : int
        Number of leaf nodes. Leaves are assumed to be nodes 0..num_leaves-1.
    node_names : list of str, optional
        Names for each node, indexed by node id.
    root : int, optional
        Root node id. If not provided, inferred as the node with no incoming edges.

    Returns
    -------
    dict
        Tree dictionary ready for use with optimize() or compute_likelihood().

    Examples
    --------
    >>> tree = make_tree(
    ...     edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
    ...     branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
    ...     num_leaves=3
    ... )
    """
    num_nodes = len(branch_lengths)

    # Infer root if not provided
    if root is None:
        children = set(child for _, child in edges)
        parents = set(parent for parent, _ in edges)
        roots = parents - children
        if len(roots) != 1:
            raise ValueError(f"Could not infer unique root. Found {len(roots)} candidates.")
        root = roots.pop()

    # Default node names
    if node_names is None:
        node_names = [f"node_{i}" if i >= num_leaves else f"leaf_{i}"
                      for i in range(num_nodes)]

    return {
        "num_leaves": num_leaves,
        "num_nodes": num_nodes,
        "root": root,
        "edges": list(edges),
        "branch_lengths": list(branch_lengths),
        "node_names": list(node_names)
    }


def optimize(
    tree: Dict,
    character_matrix: Optional[np.ndarray] = None,
    observation_matrix: Optional[np.ndarray] = None,
    mutation_priors: Optional[np.ndarray] = None,
    initial_nu: float = 0.5,
    initial_phi: float = 0.5,
    ultrametric: bool = True,
    max_iterations: int = 100,
    min_branch_length: float = 0.01,
    verbose: bool = False
) -> EMResults:
    """Run EM optimization on a phylogenetic tree.

    Parameters
    ----------
    tree : dict
        Tree structure with keys:
        - num_leaves: int
        - num_nodes: int
        - root: int
        - edges: list of (parent, child) tuples
        - branch_lengths: list of float, indexed by node
        - node_names: list of str (optional)
    character_matrix : numpy.ndarray, optional
        Character state matrix with shape (leaves, characters), dtype int32.
        Use -1 for missing data. Either this or observation_matrix must be provided.
    observation_matrix : numpy.ndarray, optional
        Observation probability matrix with shape (leaves, characters, states).
        Values should be log-probabilities. Either this or character_matrix must
        be provided.
    mutation_priors : numpy.ndarray, optional
        Prior probabilities for each mutation state with shape (characters, states).
        If not provided, uniform priors are used.
    initial_nu : float
        Initial value for nu parameter (mutation rate). Default 0.5.
    initial_phi : float
        Initial value for phi parameter (dropout rate). Default 0.5.
    ultrametric : bool
        Whether to constrain tree to be ultrametric. Default False.
    max_iterations : int
        Maximum number of EM iterations. Default 100.
    min_branch_length : float
        Minimum branch length as fraction of tree height (for ultrametric). Default 0.01.
    verbose : bool
        Whether to print progress information. Default False.

    Returns
    -------
    EMResults
        Object containing:
        - log_likelihood: final log-likelihood
        - num_iterations: number of iterations performed
        - optimized_tree: tree dict with optimized branch lengths
        - nu: optimized nu parameter
        - phi: optimized phi parameter
        - posterior_probabilities: array (characters, nodes, states)

    Raises
    ------
    ValueError
        If neither character_matrix nor observation_matrix is provided.
        If input arrays have invalid shapes or dtypes.

    Examples
    --------
    >>> import pylaml
    >>> import numpy as np
    >>>
    >>> tree = pylaml.make_tree(
    ...     edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
    ...     branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
    ...     num_leaves=3
    ... )
    >>> char_matrix = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32)
    >>>
    >>> result = pylaml.optimize(tree=tree, character_matrix=char_matrix)
    >>> print(f"Log-likelihood: {result.log_likelihood:.2f}")
    """
    # Input validation
    if character_matrix is None and observation_matrix is None:
        raise ValueError("Must provide either character_matrix or observation_matrix")

    if character_matrix is not None and observation_matrix is not None:
        raise ValueError("Cannot provide both character_matrix and observation_matrix")

    # Convert and validate character matrix
    if character_matrix is not None:
        character_matrix = np.asarray(character_matrix, dtype=np.int32)
        if character_matrix.ndim != 2:
            raise ValueError("character_matrix must be 2-dimensional (leaves, characters)")
        if character_matrix.shape[0] != tree["num_leaves"]:
            raise ValueError(
                f"character_matrix has {character_matrix.shape[0]} rows but tree has "
                f"{tree['num_leaves']} leaves"
            )
        # Recode character states to be contiguous (required by C++ implementation)
        character_matrix = recode_character_matrix(character_matrix)

    # Convert and validate observation matrix
    if observation_matrix is not None:
        observation_matrix = np.asarray(observation_matrix, dtype=np.float64)
        if observation_matrix.ndim != 3:
            raise ValueError(
                "observation_matrix must be 3-dimensional (leaves, characters, states)"
            )
        if observation_matrix.shape[0] != tree["num_leaves"]:
            raise ValueError(
                f"observation_matrix has {observation_matrix.shape[0]} rows but tree has "
                f"{tree['num_leaves']} leaves"
            )

    # Convert and validate mutation priors
    if mutation_priors is not None:
        mutation_priors = np.asarray(mutation_priors, dtype=np.float64)
        if mutation_priors.ndim != 2:
            raise ValueError("mutation_priors must be 2-dimensional (characters, states)")

    # Validate parameter bounds
    if not (NU_LB <= initial_nu <= NU_UB):
        raise ValueError(f"initial_nu must be between {NU_LB} and {NU_UB}")
    if not (PHI_LB <= initial_phi <= PHI_UB):
        raise ValueError(f"initial_phi must be between {PHI_LB} and {PHI_UB}")

    # Call the C++ implementation
    result = _core.optimize(
        tree=tree,
        character_matrix=character_matrix,
        observation_matrix=observation_matrix,
        mutation_priors=mutation_priors,
        initial_nu=initial_nu,
        initial_phi=initial_phi,
        ultrametric=ultrametric,
        max_iterations=max_iterations,
        min_branch_length=min_branch_length,
        verbose=verbose
    )

    return EMResults(result)


def compute_likelihood(
    tree: Dict,
    character_matrix: Optional[np.ndarray] = None,
    observation_matrix: Optional[np.ndarray] = None,
    mutation_priors: Optional[np.ndarray] = None,
    nu: float = 0.5,
    phi: float = 0.5
) -> float:
    """Compute log-likelihood of a tree given character data.

    This function computes the likelihood without running optimization,
    useful for comparing trees or evaluating parameter choices.

    Parameters
    ----------
    tree : dict
        Tree structure (same format as optimize()).
    character_matrix : numpy.ndarray, optional
        Character state matrix with shape (leaves, characters).
    observation_matrix : numpy.ndarray, optional
        Observation probability matrix with shape (leaves, characters, states).
    mutation_priors : numpy.ndarray, optional
        Prior probabilities for mutations.
    nu : float
        Nu parameter (mutation rate). Default 0.5.
    phi : float
        Phi parameter (dropout rate). Default 0.5.

    Returns
    -------
    float
        Log-likelihood value.

    Examples
    --------
    >>> llh = pylaml.compute_likelihood(tree, character_matrix=char_matrix, nu=0.3, phi=0.1)
    """
    # Input validation
    if character_matrix is None and observation_matrix is None:
        raise ValueError("Must provide either character_matrix or observation_matrix")

    if character_matrix is not None:
        character_matrix = np.asarray(character_matrix, dtype=np.int32)
        # Recode character states to be contiguous (required by C++ implementation)
        character_matrix = recode_character_matrix(character_matrix)

    if observation_matrix is not None:
        observation_matrix = np.asarray(observation_matrix, dtype=np.float64)

    if mutation_priors is not None:
        mutation_priors = np.asarray(mutation_priors, dtype=np.float64)

    return _core.compute_likelihood(
        tree=tree,
        character_matrix=character_matrix,
        observation_matrix=observation_matrix,
        mutation_priors=mutation_priors,
        nu=nu,
        phi=phi
    )


def project_ultrametric(tree: Dict) -> Dict:
    """Project tree branch lengths to satisfy ultrametric constraint.

    Finds branch lengths that minimize the squared distance from the original
    while ensuring all root-to-leaf distances are equal.

    Parameters
    ----------
    tree : dict
        Tree structure.

    Returns
    -------
    dict
        Tree with projected branch lengths.

    Examples
    --------
    >>> ultrametric_tree = pylaml.project_ultrametric(tree)
    """
    return _core.project_ultrametric(tree)


class SearchResults:
    """Results from topology search.

    Attributes
    ----------
    log_likelihood : float
        Final log-likelihood after search and final EM.
    num_iterations : int
        Number of simulated annealing iterations.
    optimized_tree : dict
        Tree dictionary with optimized topology and branch lengths.
    nu : float
        Optimized nu parameter (mutation rate).
    phi : float
        Optimized phi parameter (dropout rate).
    posterior_probabilities : numpy.ndarray
        Posterior probabilities with shape (characters, nodes, states).
    log_likelihoods : list of float
        Log-likelihood trajectory during search.
    """

    def __init__(self, core_result: _core.SearchResults):
        self._result = core_result

    @property
    def log_likelihood(self) -> float:
        return self._result.log_likelihood

    @property
    def num_iterations(self) -> int:
        return self._result.num_iterations

    @property
    def optimized_tree(self) -> Dict:
        return self._result.optimized_tree

    @property
    def nu(self) -> float:
        return self._result.nu

    @property
    def phi(self) -> float:
        return self._result.phi

    @property
    def posterior_probabilities(self) -> np.ndarray:
        return self._result.posterior_probabilities

    @property
    def log_likelihoods(self) -> List[float]:
        return self._result.log_likelihoods

    def __repr__(self) -> str:
        return (
            f"SearchResults(log_likelihood={self.log_likelihood:.4f}, "
            f"num_iterations={self.num_iterations}, "
            f"nu={self.nu:.4f}, phi={self.phi:.4f})"
        )


def topology_search(
    tree: Dict,
    character_matrix: Optional[np.ndarray] = None,
    observation_matrix: Optional[np.ndarray] = None,
    mutation_priors: Optional[np.ndarray] = None,
    initial_nu: float = 0.5,
    initial_phi: float = 0.5,
    ultrametric: bool = True,
    strategy: str = "sim_annealing",
    max_iterations: int = 20000,
    temperature: float = 0.1,
    min_branch_length: float = 0.01,
    num_threads: int = 1,
    verbose: bool = False
) -> SearchResults:
    """Search for optimal tree topology using NNI moves.

    Parameters
    ----------
    tree : dict
        Initial tree structure (same format as optimize()).
    character_matrix : numpy.ndarray, optional
        Character state matrix with shape (leaves, characters), dtype int32.
        Use -1 for missing data.
    observation_matrix : numpy.ndarray, optional
        Observation probability matrix with shape (leaves, characters, states).
    mutation_priors : numpy.ndarray, optional
        Prior probabilities for mutations with shape (characters, states).
    initial_nu : float
        Initial nu parameter (mutation rate). Default 0.5.
    initial_phi : float
        Initial phi parameter (dropout rate). Default 0.5.
    ultrametric : bool
        Whether to constrain tree to be ultrametric. Default False.
    strategy : str
        Search strategy. Currently only "sim_annealing" is supported.
    max_iterations : int
        Maximum number of NNI iterations. Default 20000.
    temperature : float
        Starting temperature for simulated annealing. Default 0.1.
    min_branch_length : float
        Minimum branch length fraction for ultrametric trees. Default 0.01.
    num_threads : int
        Number of threads. Default 1.
    verbose : bool
        Whether to print progress. Default False.

    Returns
    -------
    SearchResults
        Object containing optimized tree, parameters, and search trajectory.

    Raises
    ------
    ValueError
        If strategy is not supported, or inputs are invalid.
    """
    if strategy != "sim_annealing":
        raise ValueError(f"Unknown strategy '{strategy}'. Supported: 'sim_annealing'")

    # Input validation
    if character_matrix is None and observation_matrix is None:
        raise ValueError("Must provide either character_matrix or observation_matrix")

    if character_matrix is not None and observation_matrix is not None:
        raise ValueError("Cannot provide both character_matrix and observation_matrix")

    if character_matrix is not None:
        character_matrix = np.asarray(character_matrix, dtype=np.int32)
        if character_matrix.ndim != 2:
            raise ValueError("character_matrix must be 2-dimensional (leaves, characters)")
        if character_matrix.shape[0] != tree["num_leaves"]:
            raise ValueError(
                f"character_matrix has {character_matrix.shape[0]} rows but tree has "
                f"{tree['num_leaves']} leaves"
            )
        character_matrix = recode_character_matrix(character_matrix)

    if observation_matrix is not None:
        observation_matrix = np.asarray(observation_matrix, dtype=np.float64)
        if observation_matrix.ndim != 3:
            raise ValueError(
                "observation_matrix must be 3-dimensional (leaves, characters, states)"
            )
        if observation_matrix.shape[0] != tree["num_leaves"]:
            raise ValueError(
                f"observation_matrix has {observation_matrix.shape[0]} rows but tree has "
                f"{tree['num_leaves']} leaves"
            )

    if mutation_priors is not None:
        mutation_priors = np.asarray(mutation_priors, dtype=np.float64)
        if mutation_priors.ndim != 2:
            raise ValueError("mutation_priors must be 2-dimensional (characters, states)")

    if not (NU_LB <= initial_nu <= NU_UB):
        raise ValueError(f"initial_nu must be between {NU_LB} and {NU_UB}")
    if not (PHI_LB <= initial_phi <= PHI_UB):
        raise ValueError(f"initial_phi must be between {PHI_LB} and {PHI_UB}")

    result = _core.topology_search(
        tree=tree,
        character_matrix=character_matrix,
        observation_matrix=observation_matrix,
        mutation_priors=mutation_priors,
        initial_nu=initial_nu,
        initial_phi=initial_phi,
        ultrametric=ultrametric,
        max_iterations=max_iterations,
        temperature=temperature,
        min_branch_length=min_branch_length,
        num_threads=num_threads,
        verbose=verbose
    )

    return SearchResults(result)
