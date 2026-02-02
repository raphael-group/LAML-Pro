# laml-pro: joint maximum likelihood estimation of cell genotypes and cell lineage trees

`lamlpro` is a maximum likelihood cell lineage tree inference algorithm
based on the under the Probabilistic Mixed-type Missing Observation (PMMO) 
model. `lamlpro` has two modes and can _i)_ quickly infer branch lengths for a given cell
lineage tree topology and _ii)_ find the most likely cell lineage tree using
topology search.

If you find `lamlpro` useful in your research, please cite the following paper:

```
```

If you are interested in reproducing the analyses in this paper, the reproducibility repository can be found at
[LAML-Pro-experiments](https://github.com/raphael-group/LAML-Pro-experiments).

## Installation

To build `lamlpro` manually requires only a modern C++20 compiler and CMake. We recommend using python 3.10 for the helper scripts.
To install, simply clone the repository and compile the code using CMake, making
sure to initialize all git submodules.

```bash
git clone git@github.com:raphael-group/LAML-Pro.git --recursive
```
We refer the user to the ipopt installation instructions [here](https://coin-or.github.io/Ipopt/INSTALL.html). 
Generally, however, if you are on MacOS you can install with `brew install ipopt` and on Linux you can run `conda install -c conda-forge ipopt`. 
Please change directories into LAML-Pro (`cd LAML-Pro`) and run the following commands:

```bash
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make
mv src/lamlpro ../lamlpro
```
The output files consist of the binary `lamlpro` which can be executed
from the command line. For ease of use, we suggest adding the `lamlpro`
binary to a directory listed in the `PATH` environmental variable.
At the end of these steps, `lamlpro` should sit in the root directory.

### Python Library Installation

We also provide Python bindings (`pylaml`) that expose the core optimization API.
This allows you to use `lamlpro` directly from Python with numpy arrays.

**Requirements:**
- Python >= 3.8
- numpy >= 1.20
- IPOPT (must be installed on your system, see above)

**Installation:**
```bash
pip install .
```

To verify the installation:
```bash
python -c "import pylaml; print(pylaml.__version__)"
```

## Usage

`lamlpro` is a command-line tool to infer a cell lineage tree $\mathcal{T}$
on $n$ cells from a set of observations $X$ at the $n$ cells. The tool
currently supports two types of observated data:
* The observed data $X$ is an $n\text{-by-}m$ character matrix specifying the 
character-states of each of $n$ cells at $m$ characters. 
* The observed data $X$ is an $n\text{-by-}m+2$ matrix, where $n$ specifies the number of observations, $m$ is the number of edit states (including $0$), and $2$ corresponds to additional metadata columns. The columns should be `cell_name,target_site`, followed by columns corresponding to each state `state0_prob,state1_prob,state2_prob,state3_prob`. The state probability values should correspond to the generative probability (e.g. probability of hidden state $0$ having generated this observation), reported be in log space. They should be named with this convention, since we identify these columns as `state*_prob`.

`lamlpro` requires an initial cell lineage tree $\mathcal{T}_0$ as input
to the algorithm and provides two modes:

* mode: `optimize` finds the optimal branch lengths $\delta_e$ and model parameter $\Theta$ for $\mathcal{T}_0$ under the PMMO model.
* mode: `search` finds the most likely tree $\mathcal{T}^*$, branch lengths $\delta_e$ and model parameters $\Theta$ under the PMMO model.

If you are interested in inferring a tree from scratch, we recommend using the `search` mode of
`lamlpro`. If you are interested fitting branch lengths to a tree we recommend using the `optimize` mode
of `lamlpro`. In either case, one is required to specify both the observations $X$ and initial tree
$\mathcal{T}_0$ via the command line flags `--tree` and `--matrix`.

The tool has the following usage format:
```
Usage: lamlpro [--help] [--version] [--mutation-priors VAR] --matrix VAR [--data-type VAR] --tree VAR --output VAR [--verbose] [--ultrametric] [--threads VAR] [--mode VAR] [--seed VAR] [--max-iterations VAR] [--temp VAR] [--min-branch-length VAR]

Optional arguments:
  -h, --help             shows help message and exits
  --version              prints version information and exits
  -m, --mutation-priors  path to the mutation priors file (CSV) [nargs=0..1] [default: ""]
  -c, --matrix           path to the observed data file (CSV) [required]
  -d, --data-type        options are 'character-matrix' or 'observation-matrix'. [nargs=0..1] [default: "character-matrix"]
  -t, --tree             path to the rooted binary tree (newick) [required]
  -o, --output           prefix for output files [required]
  -v, --verbose          save all console logs to a file automatically.
  -u, --ultrametric      enforce ultrametric constraint during optimization.
  --threads              number of threads to use [nargs=0..1] [default: 10]
  --mode                 'optimize' for parameter optimization or 'search' for tree search [nargs=0..1] [default: "optimize"]
  --seed                 random seed for reproducibility [nargs=0..1] [default: 73]
  --max-iterations       maximum number of iterations for hill climbing [nargs=0..1] [default: 20000]
  --temp                 starting temperature for topology search [nargs=0..1] [default: 0.1]
  --min-branch-length    minimum branch length relative to scaled tree with unit height [nargs=0..1] [default: 0.01]
```

There are two main output files of `lamlpro`: the `*.newick` output tree in newick format, and `*results.json` which contains important parameter estimates as well as metadata. 

> [!TIP]
> Use the flag `--ultrametric` to ensure the cell lineage tree has equal
> length root-to-leaf paths.

## Examples

We provide simulated cell lineage trees with $n = 100, 250, 500$ nodes, $400$ characters,
and simulated observations in order to demonstrate `lamlpro`.



### Example 1: Character Matrix
To apply `lamlpro` to character matrix data, we first infer a cell lineage tree
$\mathcal{T}_0$ with $n = 250$ cells using the Neighbor Joining (NJ) algorithm. 
The initial
tree can be inferred using any method, but for the sake of the example,
we use the following command:
```
python scripts/neighbor_joining.py examples/n250_m30_character_matrix/character_matrix.csv examples/n250_m30_character_matrix/initial
```

> [!TIP]
> This script uses the neighbor joining implementation from biopython and pandas. You can install Python dependencies for the other scripts with `pip3 install biopython pandas dendropy scipy`.


This results in two files `examples/n250_m30_character_matrix/initial_hamming_tree.nwk` and
`examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk`. If we compute the
distance from the inferred and true trees, we see that both are quite far away from
the ground truth:
```
python scripts/metrics.py --reference examples/n250_m30_character_matrix/tree.nwk --trees examples/n250_m30_character_matrix/initial_hamming_tree.nwk examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk
Tree                                               RF Distance Normalized RF
examples/n250_m30_character_matrix/initial_hamming_tree.nwk        404     0.817814
examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk        356     0.720648
```

With either initial tree, one can run `lamlpro` with the following command:
```
./lamlpro --matrix examples/n250_m30_character_matrix/character_matrix.csv --tree examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk -o examples/n250_m30_character_matrix/lamlpro --ultrametric --mode search --max-iterations 2500
```
The preceding command enforces that the tree is ultrametric and runs topology search for `2500`
iterations. For practical applications, we recommend setting this value higher and making
sure that the algorithm converges by checking that the log-likelihood improvements have plateaued (by looking at the `.json` output file). The preceding command results in two output files:
`examples/n250_m30_character_matrix/lamlpro_tree.newick` and 
`examples/n250_m30_character_matrix/lamlpro_results.json`.
The first file is the inferred cell lineage tree with branch lengths (in mutation units) and the second contains 
parameter estimates and important metadata, such as the per-iteration log-likelihood.

On this example, `lamlpro` should improve the tree topology and this can be verified by
running:
```
python scripts/metrics.py --reference examples/n250_m30_character_matrix/tree.nwk --trees examples/n250_m30_character_matrix/initial_hamming_tree.nwk examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk examples/n250_m30_character_matrix/lamlpro_tree.newick
```

Example output:
```
Tree                                               RF Distance Normalized RF
examples/n250_m30_character_matrix/initial_hamming_tree.nwk        404     0.817814
examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk        356     0.720648
examples/n250_m30_character_matrix/lamlpro_tree.newick        140     0.283401
```

### Example 2: Observation Matrix

To apply `lamlpro` to observation matrix data, we first infer a cell lineage tree
$\mathcal{T}_0$ with $n = 100$ cells using the Neighbor Joining (NJ) algorithm, as above.

We provide an initial starting tree `argmax_nj.nwk` if you would like to skip directly to testing `lamlpro`. This example corresponds to simulated data condition `k400_s0_sub100_r01_dim3_r0.0439_p0.0`. 

With an initial tree, one can run `lamlpro` with the following command:
```
./lamlpro --matrix examples/n100_m400_observation_matrix/observation_matrix.csv --tree examples/n100_m400_observation_matrix/argmax_nj.nwk -o examples/n100_m400_observation_matrix/lamlpro --ultrametric --mode search --max-iterations 2500 --data-type "observation-matrix"
```
The preceding command enforces that the tree is ultrametric and runs topology search for `2500`
iterations. You can change this number of iterations to be smaller for an initial test. For practical applications, we recommend setting this value higher and making
sure that the algorithm converges by checking that the log-likelihood improvements have plateaued (by looking at the `.json` output file). The preceding command results in four output files:
`examples/n100_m400_observation_matrix/lamlpro_tree.newick` and 
`examples/n100_m400_observation_matrix/lamlpro_results.json` and
`examples/n100_m400_observation_matrix/lamlpro_posterior_probs.csv` and
`examples/n100_m400_observation_matrix/lamlpro_posterior_argmax.csv`.

The first file is the inferred cell lineage tree with branch lengths (in mutation units) and the second contains 
parameter estimates and important metadata, such as the per-iteration log-likelihood.
The third file contains the probabilities over all states at each cell and site on the fixed maximum likelihood tree after convergence, and the fourth file contains the argmax *maximum a posteriori* (MAP) genotypes. 

On this example, `lamlpro` should improve the tree topology and this can be verified by
running:
```
python scripts/metrics.py --reference examples/n100_m400_observation_matrix/tree.nwk --trees examples/n100_m400_observation_matrix/argmax_nj.nwk examples/n100_m400_observation_matrix/example_output/lamlpro_test_tree.newick
Tree                                               RF Distance Normalized RF
examples/n100_m400_observation_matrix/argmax_nj.nwk         24     0.123711
examples/n100_m400_observation_matrix/example_output/lamlpro_test_tree.newick          0     0.000000
```

## Python API

The `pylaml` Python library provides direct access to the optimization algorithms
without file I/O. This is useful for integrating `lamlpro` into Python workflows
or implementing custom topology search algorithms.

### Quick Start

```python
import pylaml
import numpy as np

# Define a tree structure
tree = pylaml.make_tree(
    edges=[(4, 3), (3, 0), (3, 1), (4, 2)],  # (parent, child) pairs
    branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],  # indexed by node
    num_leaves=3
)

# Define character matrix (rows=leaves, cols=characters)
# Use -1 for missing data
char_matrix = np.array([
    [0, 1, 0],  # leaf 0
    [1, 0, 1],  # leaf 1
    [0, 1, 1],  # leaf 2
], dtype=np.int32)

# Run EM optimization
result = pylaml.optimize(
    tree=tree,
    character_matrix=char_matrix,
    max_iterations=100
)

print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Nu (mutation rate): {result.nu:.4f}")
print(f"Phi (dropout rate): {result.phi:.4f}")
```

### Tree Format

Trees are represented as Python dictionaries with the following structure:

```python
tree = {
    "num_leaves": 3,           # Number of leaf nodes
    "num_nodes": 5,            # Total nodes (leaves + internal)
    "root": 4,                 # Root node index
    "edges": [(4, 3), ...],    # List of (parent, child) tuples
    "branch_lengths": [...],   # Branch lengths indexed by node
    "node_names": [...]        # Optional node names
}
```

Leaves must be indexed `0` to `num_leaves - 1`. The character matrix rows
must correspond to leaves in this order.

### API Reference

#### `pylaml.optimize()`

Run EM optimization on a phylogenetic tree.

```python
result = pylaml.optimize(
    tree,                          # Tree dictionary
    character_matrix=None,         # Character matrix (n_leaves, n_chars), int32
    observation_matrix=None,       # OR observation matrix (n_leaves, n_chars, n_states), float64
    mutation_priors=None,          # Prior probabilities (n_chars, n_states), optional
    initial_nu=0.5,                # Initial mutation rate
    initial_phi=0.5,               # Initial dropout rate
    ultrametric=False,             # Enforce ultrametric constraint
    max_iterations=100             # Maximum EM iterations
)
```

Returns an `EMResults` object with:
- `log_likelihood`: Final log-likelihood
- `num_iterations`: Number of iterations performed
- `optimized_tree`: Tree dict with optimized branch lengths
- `nu`, `phi`: Optimized parameters
- `posterior_probabilities`: Array of shape (n_chars, n_nodes, n_states)

#### `pylaml.compute_likelihood()`

Compute log-likelihood without optimization.

```python
llh = pylaml.compute_likelihood(
    tree,
    character_matrix=char_matrix,
    nu=0.3,
    phi=0.1
)
```

#### `pylaml.project_ultrametric()`

Project tree to satisfy ultrametric constraint.

```python
ultrametric_tree = pylaml.project_ultrametric(tree)
```

#### `pylaml.make_tree()`

Helper to create tree dictionaries.

```python
tree = pylaml.make_tree(
    edges=[(4, 3), (3, 0), (3, 1), (4, 2)],
    branch_lengths=[0.1, 0.1, 0.2, 0.1, 0.0],
    num_leaves=3,
    node_names=None,  # Optional
    root=None         # Auto-inferred if not provided
)
```

### Example: Loading Data from Files

Here's how to load a character matrix and tree from files:

```python
import pylaml
import numpy as np
import csv
import dendropy

def load_character_matrix(path):
    """Load character matrix from CSV file."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        rows = list(reader)
        cell_names = [row[0] for row in rows]
        char_matrix = np.array([
            [-1 if x == '?' else int(x) for x in row[1:]]
            for row in rows
        ], dtype=np.int32)
    return cell_names, char_matrix

def load_tree(tree_path):
    """Load tree from Newick file."""
    dtree = dendropy.Tree.get(path=tree_path, schema="newick")

    num_leaves = len(dtree.leaf_nodes())
    num_nodes = len(list(dtree))

    # Map nodes: leaves first (0 to num_leaves-1), then internal
    leaf_map, internal_map = {}, {}
    leaf_idx, internal_idx = 0, num_leaves

    for node in dtree.postorder_node_iter():
        if node.is_leaf():
            leaf_map[node] = leaf_idx
            leaf_idx += 1
        else:
            internal_map[node] = internal_idx
            internal_idx += 1

    node_map = {**leaf_map, **internal_map}

    edges, branch_lengths, node_names = [], [0.0] * num_nodes, [""] * num_nodes

    for node in dtree.postorder_node_iter():
        idx = node_map[node]
        branch_lengths[idx] = node.edge.length or 0.0
        node_names[idx] = node.taxon.label if node.is_leaf() else f"internal_{idx}"
        if node.parent_node:
            edges.append((node_map[node.parent_node], idx))

    root_idx = node_map[dtree.seed_node]
    branch_lengths[root_idx] = 0.0  # Root has no branch

    return {
        "num_leaves": num_leaves, "num_nodes": num_nodes, "root": root_idx,
        "edges": edges, "branch_lengths": branch_lengths, "node_names": node_names
    }

# Load data
cell_names, char_matrix = load_character_matrix(
    "examples/n250_m30_character_matrix/character_matrix.csv"
)
tree = load_tree("examples/n250_m30_character_matrix/tree.nwk")

# Reorder matrix to match tree leaf order
tree_leaf_order = [tree["node_names"][i] for i in range(tree["num_leaves"])]
cell_to_row = {name: i for i, name in enumerate(cell_names)}
reordered_indices = [cell_to_row[name] for name in tree_leaf_order]
char_matrix = char_matrix[reordered_indices]

# Run optimization
result = pylaml.optimize(tree=tree, character_matrix=char_matrix, max_iterations=100)
print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"Nu: {result.nu:.4f}, Phi: {result.phi:.4f}")
```

### Example: Observation Matrix

For probabilistic observation data, pass a 3D array of log-probabilities:

```python
import pylaml
import numpy as np

# observation_matrix shape: (n_leaves, n_characters, n_states)
# Values should be log-probabilities including state 0 (unmutated)
obs_matrix = np.array([...], dtype=np.float64)  # shape (100, 400, 4)

result = pylaml.optimize(
    tree=tree,
    observation_matrix=obs_matrix,  # Full matrix including state 0
    max_iterations=100
)
```

> [!NOTE]
> For observation matrices, include all states (including state 0).
> The library handles state 0 exclusion internally when computing priors.
