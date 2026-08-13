"""
pylaml - Python bindings for fastLAML

Fast Lineage-Aware Maximum Likelihood for phylogenetic tree optimization.
"""

from typing import Dict, List, Optional, Tuple, Union
import csv
import json
import os
import random
import time

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


class LayoutWarning(UserWarning):
    """Input array is not C-contiguous (still read correctly, via strides)."""


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


# ---------------------------------------------------------------------------
# File I/O helpers for run_lamlpro
# ---------------------------------------------------------------------------

NEGATIVE_INFINITY = -1e8


def _parse_newick_string(newick_str: str) -> Dict:
    """Parse a Newick string into a tree dict.

    Matches the node-ID assignment convention used by the C++ CLI:
    leaves get IDs 0..num_leaves-1 and internal nodes get IDs
    num_leaves..num_nodes-1, assigned in pre-order traversal order.

    Branch lengths of 0 are replaced with 1.0 (matching C++ behaviour).
    """
    newick_str = newick_str.strip()
    # Strip leading tree annotations like [&R]
    if newick_str.startswith("["):
        close = newick_str.index("]")
        newick_str = newick_str[close + 1:].strip()
    if newick_str.endswith(";"):
        newick_str = newick_str[:-1]

    pos = [0]

    def _parse_node():
        children = []
        if pos[0] < len(newick_str) and newick_str[pos[0]] == "(":
            pos[0] += 1  # skip '('
            children.append(_parse_node())
            while pos[0] < len(newick_str) and newick_str[pos[0]] == ",":
                pos[0] += 1  # skip ','
                children.append(_parse_node())
            if pos[0] < len(newick_str) and newick_str[pos[0]] == ")":
                pos[0] += 1  # skip ')'

        # Parse name
        name_start = pos[0]
        while pos[0] < len(newick_str) and newick_str[pos[0]] not in ":,);(":
            pos[0] += 1
        name = newick_str[name_start:pos[0]].strip()

        # Parse branch length
        length = 0.0
        if pos[0] < len(newick_str) and newick_str[pos[0]] == ":":
            pos[0] += 1
            len_start = pos[0]
            while pos[0] < len(newick_str) and newick_str[pos[0]] not in ",);(":
                pos[0] += 1
            try:
                length = float(newick_str[len_start:pos[0]])
            except ValueError:
                length = 1.0

        return {"name": name, "length": length, "children": children}

    root_data = _parse_node()

    # Count leaves and internal nodes
    def _count(node):
        if not node["children"]:
            return 1, 0
        leaves = 0
        internals = 1
        for c in node["children"]:
            cl, ci = _count(c)
            leaves += cl
            internals += ci
        return leaves, internals

    num_leaves, num_internal = _count(root_data)
    num_nodes = num_leaves + num_internal

    edges: List[Tuple[int, int]] = []
    branch_lengths = [0.0] * num_nodes
    node_names = [""] * num_nodes
    leaf_idx = [0]
    internal_idx = [num_leaves]

    def _assign_ids(node):
        if not node["children"]:
            nid = leaf_idx[0]
            leaf_idx[0] += 1
        else:
            nid = internal_idx[0]
            internal_idx[0] += 1

        node_names[nid] = node["name"]
        bl = node["length"]
        if bl == 0.0:
            bl = 1.0
        branch_lengths[nid] = bl

        for child in node["children"]:
            child_id = _assign_ids(child)
            edges.append((nid, child_id))

        return nid

    root_id = _assign_ids(root_data)

    return {
        "num_leaves": num_leaves,
        "num_nodes": num_nodes,
        "root": root_id,
        "edges": edges,
        "branch_lengths": branch_lengths,
        "node_names": node_names,
    }


def read_newick(filepath: str) -> Dict:
    """Read a Newick tree file and return a tree dict.

    Parameters
    ----------
    filepath : str
        Path to the Newick tree file.

    Returns
    -------
    dict
        Tree dictionary ready for use with optimize() or topology_search().
    """
    with open(filepath) as f:
        return _parse_newick_string(f.read())


def _read_character_matrix_csv(filepath: str):
    """Read a character matrix CSV file.

    Returns (taxa_names, matrix) where matrix is a list of lists of ints
    with -1 for missing data.
    """
    taxa_names: List[str] = []
    matrix: List[List[int]] = []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        _header = next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            taxa_names.append(row[0])
            states = []
            for val in row[1:]:
                val = val.strip()
                if val == "?" or val == "":
                    states.append(-1)
                else:
                    states.append(int(val))
            matrix.append(states)

    return taxa_names, matrix


def _read_observation_matrix_csv(filepath: str):
    """Read an observation matrix CSV file.

    Returns (taxa_names, obs_matrix) where obs_matrix is a 3-D numpy array
    of shape (num_taxa, num_characters, num_states).
    """
    taxa_names: List[str] = []
    taxa_index: Dict[str, int] = {}
    character_keys: set = set()
    parsed_rows: list = []

    max_num_states = 0

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        state_cols = [c for c in reader.fieldnames or [] if c.startswith("state") and "_prob" in c]

        for row in reader:
            taxon = row["cell_name"]
            cassette_idx = 0
            target_site = int(float(row["target_site"]))
            key = (cassette_idx, target_site)
            character_keys.add(key)

            probs = []
            for col in state_cols:
                try:
                    probs.append(float(row[col]))
                except (ValueError, KeyError):
                    probs.append(NEGATIVE_INFINITY)
            max_num_states = max(max_num_states, len(probs))

            parsed_rows.append((taxon, key, probs))
            if taxon not in taxa_index:
                taxa_index[taxon] = len(taxa_names)
                taxa_names.append(taxon)

    sorted_keys = sorted(character_keys)
    key_to_idx = {k: i for i, k in enumerate(sorted_keys)}
    num_chars = len(sorted_keys)
    num_taxa = len(taxa_names)

    obs = np.full((num_taxa, num_chars, max_num_states), NEGATIVE_INFINITY, dtype=np.float64)

    for taxon, key, probs in parsed_rows:
        i = taxa_index[taxon]
        j = key_to_idx[key]
        obs[i, j, : len(probs)] = probs

    return taxa_names, obs


def _read_mutation_priors_csv(filepath: str):
    """Read mutation priors from a headerless CSV (character, state, probability)."""
    priors: List[Tuple[int, int, float]] = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            priors.append((int(row[0]), int(row[1]), float(row[2])))
    return priors


def _reorder_character_matrix(
    tree_dict: Dict, taxa_names: List[str], raw_matrix: List[List[int]]
) -> np.ndarray:
    """Reorder a character matrix so row i corresponds to tree leaf i."""
    num_leaves = tree_dict["num_leaves"]
    if not raw_matrix:
        raise ValueError("Character matrix is empty")
    num_chars = max(len(r) for r in raw_matrix)

    taxa_lookup = {name: idx for idx, name in enumerate(taxa_names)}
    reordered = np.full((num_leaves, num_chars), -1, dtype=np.int32)

    for leaf_id in range(num_leaves):
        leaf_name = tree_dict["node_names"][leaf_id]
        if leaf_name not in taxa_lookup:
            raise ValueError(
                f"Tree leaf '{leaf_name}' not found in character matrix"
            )
        orig_row = taxa_lookup[leaf_name]
        for c in range(min(num_chars, len(raw_matrix[orig_row]))):
            reordered[leaf_id, c] = raw_matrix[orig_row][c]

    return reordered


def _reorder_observation_matrix(
    tree_dict: Dict, taxa_names: List[str], obs_matrix: np.ndarray
) -> np.ndarray:
    """Reorder an observation matrix so row i corresponds to tree leaf i."""
    num_leaves = tree_dict["num_leaves"]
    taxa_lookup = {name: idx for idx, name in enumerate(taxa_names)}
    num_chars = obs_matrix.shape[1]
    num_states = obs_matrix.shape[2]
    reordered = np.full((num_leaves, num_chars, num_states), NEGATIVE_INFINITY, dtype=np.float64)

    for leaf_id in range(num_leaves):
        leaf_name = tree_dict["node_names"][leaf_id]
        if leaf_name not in taxa_lookup:
            raise ValueError(
                f"Tree leaf '{leaf_name}' not found in observation matrix"
            )
        orig_row = taxa_lookup[leaf_name]
        reordered[leaf_id] = obs_matrix[orig_row]

    return reordered


def _recode_with_priors(
    char_matrix: np.ndarray, raw_priors: Optional[List[Tuple[int, int, float]]]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Recode character matrix to contiguous states and align mutation priors.

    This matches the recoding logic in the C++ ``process_phylogeny_data``:
    state 0 stays 0, other positive states are mapped to 1, 2, 3, ...
    per character.  Missing data (-1) is preserved.

    If *raw_priors* is not None the prior probabilities are remapped and
    normalised so they can be passed directly to ``_core.optimize()``.
    """
    n_leaves, n_chars = char_matrix.shape
    recoded = np.zeros_like(char_matrix)
    mappings: List[Dict[int, int]] = []
    max_alphabet_size = 0

    for c in range(n_chars):
        col = char_matrix[:, c]
        valid_states = sorted(set(int(s) for s in col if s > 0))
        mapping: Dict[int, int] = {}
        new_idx = 1
        for s in valid_states:
            mapping[s] = new_idx
            new_idx += 1
        mappings.append(mapping)
        max_alphabet_size = max(max_alphabet_size, len(valid_states))

        for i in range(n_leaves):
            v = int(col[i])
            if v > 0 and v in mapping:
                recoded[i, c] = mapping[v]
            elif v == 0:
                recoded[i, c] = 0
            else:
                recoded[i, c] = -1

    if raw_priors is None:
        return recoded, None

    # Build recoded priors array: shape (n_chars, max_alphabet_size)
    recoded_priors = np.zeros((n_chars, max_alphabet_size), dtype=np.float64)
    for character, orig_state, probability in raw_priors:
        if character < 0 or character >= n_chars:
            continue
        if orig_state <= 0:
            continue
        mapping = mappings[character]
        if orig_state not in mapping:
            continue
        new_state = mapping[orig_state] - 1  # 0-indexed into priors
        if new_state < max_alphabet_size:
            recoded_priors[character, new_state] = probability

    # Normalise per character
    for c in range(n_chars):
        total = recoded_priors[c].sum()
        if total > 0:
            recoded_priors[c] /= total

    return recoded, recoded_priors


def _write_newick(tree_dict: Dict) -> str:
    """Convert a tree dict back to a Newick string."""
    children_map: Dict[int, List[int]] = {}
    for parent, child in tree_dict["edges"]:
        children_map.setdefault(parent, []).append(child)

    node_names = tree_dict.get("node_names", [])
    branch_lengths = tree_dict["branch_lengths"]

    def _recurse(nid: int) -> str:
        kids = children_map.get(nid, [])
        result = ""
        if kids:
            result += "(" + ",".join(_recurse(c) for c in kids) + ")"
        if nid < len(node_names) and node_names[nid]:
            result += node_names[nid]
        result += ":" + f"{branch_lengths[nid]:.6f}"
        return result

    return _recurse(tree_dict["root"]) + ";"


def _write_output_files(
    tree_dict: Dict,
    output_prefix: str,
    result,
    data_type: str,
    command_str: str,
    runtime_ms: float,
    log_likelihoods: List[float],
):
    """Write result files matching the CLI output format."""
    # 1. Newick tree
    newick = _write_newick(result.optimized_tree)
    with open(output_prefix + "_tree.newick", "w") as f:
        f.write(newick)

    # 2. Posterior matrices (observation-matrix data type only, matching C++)
    if data_type == "observation-matrix":
        posteriors = result.posterior_probabilities  # (chars, nodes, states)
        num_characters = posteriors.shape[0]
        num_nodes = posteriors.shape[1]
        opt_tree = result.optimized_tree
        opt_names = opt_tree.get("node_names", [f"node_{i}" for i in range(num_nodes)])

        # Posterior probs CSV
        with open(output_prefix + "_posterior_probs.csv", "w") as f:
            f.write("node")
            for c in range(num_characters):
                f.write(f",character_{c}")
            f.write("\n")
            for nid in range(len(opt_names)):
                f.write(opt_names[nid])
                for c in range(num_characters):
                    probs = posteriors[c, nid, :]
                    parts = []
                    for s_idx in range(len(probs)):
                        state_label = s_idx - 1
                        parts.append(f"{state_label}:{probs[s_idx]:.6f}")
                    f.write("," + "/".join(parts))
                f.write("\n")

        # Argmax CSV
        with open(output_prefix + "_posterior_argmax.csv", "w") as f:
            f.write(f"Newick Tree:\n{newick}\n")
            f.write("node")
            for c in range(num_characters):
                f.write(f",character_{c}")
            f.write("\n")
            for nid in range(len(opt_names)):
                f.write(opt_names[nid])
                for c in range(num_characters):
                    probs = posteriors[c, nid, :]
                    best_idx = int(np.argmax(probs))
                    state_label = best_idx - 1
                    f.write(f",{state_label}")
                f.write("\n")

    # 3. JSON summary
    output_json = {
        "phi": result.phi,
        "nu": result.nu,
        "em_iterations": result.num_iterations,
        "best_log_likelihood": result.log_likelihood,
        "command": command_str,
        "runtime_ms": runtime_ms,
        "log_likelihoods": log_likelihoods,
    }
    with open(output_prefix + "_results.json", "w") as f:
        json.dump(output_json, f, indent=4)
        f.write("\n")


def run_lamlpro(
    matrix: str,
    tree: str,
    output: str,
    data_type: str = "character-matrix",
    mutation_priors: str = "",
    mode: str = "optimize",
    seed: int = 73,
    ultrametric: bool = False,
    threads: int = 1,
    max_iterations: int = 20000,
    temp: float = 0.1,
    min_branch_length: float = 0.01,
    verbose: bool = False,
) -> Union[EMResults, SearchResults]:
    """Run LAML-Pro with the same interface as the CLI.

    This mirrors the ``lamlpro`` command-line tool: it reads a Newick tree
    and CSV data files, runs either parameter optimisation or topology
    search, writes output files, and returns the result object.

    Parameters
    ----------
    matrix : str
        Path to the observed data CSV file (``-c`` / ``--matrix``).
    tree : str
        Path to the rooted binary Newick tree file (``-t`` / ``--tree``).
    output : str
        Prefix for output files (``-o`` / ``--output``).
    data_type : str
        ``'character-matrix'`` or ``'observation-matrix'``.
    mutation_priors : str
        Path to mutation priors CSV, or empty string for uniform priors.
    mode : str
        ``'optimize'`` for EM parameter optimisation, ``'search'`` for
        simulated-annealing topology search.
    seed : int
        Random seed for reproducibility.
    ultrametric : bool
        Enforce ultrametric constraint during optimisation.
    threads : int
        Number of threads (used in search mode).
    max_iterations : int
        Maximum iterations for topology search.
    temp : float
        Starting temperature for simulated annealing.
    min_branch_length : float
        Minimum branch length relative to scaled tree height.
    verbose : bool
        Print progress information.

    Returns
    -------
    EMResults or SearchResults
        The optimisation or search result object.
    """
    if mode not in ("optimize", "search"):
        raise ValueError(f"mode must be 'optimize' or 'search', got '{mode}'")
    if data_type not in ("character-matrix", "observation-matrix"):
        raise ValueError(
            f"data_type must be 'character-matrix' or 'observation-matrix', got '{data_type}'"
        )

    # Build a command string for the JSON output (mirrors CLI behaviour)
    command_str = (
        f"pylaml.run_lamlpro(matrix='{matrix}', tree='{tree}', "
        f"output='{output}', data_type='{data_type}', "
        f"mode='{mode}', seed={seed}, ultrametric={ultrametric})"
    )

    # --- 1. Load tree ---
    tree_dict = read_newick(tree)

    # Validate binary tree
    children_count: Dict[int, int] = {}
    for parent, _child in tree_dict["edges"]:
        children_count[parent] = children_count.get(parent, 0) + 1
    for nid, count in children_count.items():
        if count != 2:
            raise ValueError(
                f"Tree node {nid} has {count} children; expected a binary tree"
            )

    # --- 2. Load data ---
    character_matrix_np = None
    observation_matrix_np = None
    priors_np = None

    if data_type == "character-matrix":
        taxa_names, raw_matrix = _read_character_matrix_csv(matrix)
        char_np = _reorder_character_matrix(tree_dict, taxa_names, raw_matrix)

        raw_priors = None
        if mutation_priors:
            raw_priors = _read_mutation_priors_csv(mutation_priors)

        recoded, priors_np = _recode_with_priors(char_np, raw_priors)
        character_matrix_np = recoded
    else:
        taxa_names, obs_np = _read_observation_matrix_csv(matrix)
        observation_matrix_np = _reorder_observation_matrix(tree_dict, taxa_names, obs_np)

        if mutation_priors:
            raw_priors = _read_mutation_priors_csv(mutation_priors)
            # For observation matrix, priors are indexed directly
            num_chars = observation_matrix_np.shape[1]
            num_edit_states = observation_matrix_np.shape[2] - 1
            priors_arr = np.zeros((num_chars, num_edit_states), dtype=np.float64)
            for character, orig_state, probability in raw_priors:
                if 0 <= character < num_chars and orig_state > 0:
                    idx = orig_state - 1
                    if idx < num_edit_states:
                        priors_arr[character, idx] = probability
            for c in range(num_chars):
                total = priors_arr[c].sum()
                if total > 0:
                    priors_arr[c] /= total
                else:
                    priors_arr[c] = 1.0 / num_edit_states
            priors_np = priors_arr

    # --- 3. Randomise initial parameters (matching CLI behaviour) ---
    # Use the C++ RNG (std::mt19937 + uniform_real_distribution<float>)
    # to produce identical initial values as the CLI for a given seed.
    num_bl = len(tree_dict["branch_lengths"]) if mode == "optimize" else 0
    initial_phi, initial_nu, rand_bls = _core.generate_initial_params(seed, num_bl)

    if mode == "optimize":
        tree_dict["branch_lengths"] = list(rand_bls)

    # Label internal nodes (matching CLI behaviour)
    for nid in range(tree_dict["num_nodes"]):
        if not tree_dict["node_names"][nid]:
            tree_dict["node_names"][nid] = f"internal_{nid}"

    # --- 4. Run optimisation or search ---
    start = time.time()

    if mode == "optimize":
        result = _core.optimize(
            tree=tree_dict,
            character_matrix=character_matrix_np,
            observation_matrix=observation_matrix_np,
            mutation_priors=priors_np,
            initial_nu=initial_nu,
            initial_phi=initial_phi,
            ultrametric=ultrametric,
            max_iterations=100,
            min_branch_length=min_branch_length,
            verbose=verbose,
        )
        result = EMResults(result)
        log_likelihoods: List[float] = []
    else:
        result = _core.topology_search(
            tree=tree_dict,
            character_matrix=character_matrix_np,
            observation_matrix=observation_matrix_np,
            mutation_priors=priors_np,
            initial_nu=initial_nu,
            initial_phi=initial_phi,
            ultrametric=ultrametric,
            max_iterations=max_iterations,
            temperature=temp,
            min_branch_length=min_branch_length,
            num_threads=threads,
            verbose=verbose,
            seed=seed,
        )
        result = SearchResults(result)
        log_likelihoods = list(result.log_likelihoods)

    end = time.time()
    runtime_ms = (end - start) * 1000.0

    # --- 5. Write output files ---
    _write_output_files(
        tree_dict, output, result, data_type, command_str, runtime_ms, log_likelihoods
    )

    if verbose:
        print(f"Log likelihood: {result.log_likelihood:.4f}")
        print(f"nu: {result.nu:.4f}, phi: {result.phi:.4f}")
        print(f"Runtime: {runtime_ms:.0f} ms")
        print(f"Output files written with prefix: {output}")

    return result
