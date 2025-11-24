# laml-pro: joint maximum likelihood estimation of cell genotypes and cell lineage trees

`lamlpro` is a maximum likelihood cell lineage tree inference algorithm
based on the under the Probabilistic Mixed-type Missing Observation (PMMO) 
model. `lamlpro` has two modes and can _i)_ quickly infer branch lengths for a given cell
lineage tree topology and _ii)_ find the most likely cell lineage tree using
topology search.

If you find `lamlpro` useful in your research, please cite the following paper:

```
```

## Installation

To build `lamlpro` manually requires only a modern C++20 compiler and CMake. 
To install, simply clone the repository and compile the code using CMake, making
sure to initialize all git submodules.

```bash
$ git clone git@github.com:raphael-group/LAML-Pro.git --recursive
```

To build, run the following commands:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
$ mv src/lamlpro lampro
```

The output files consist of the binary `lamlpro` which can be executed
from the command line. For ease of use, we suggest adding the `lamlpro`
binary to a directory listed in the `PATH` environmental variable.

## Usage

`lamlpro` is a command-line tool to infer a cell lineage tree $\mathcal{T}$
on $n$ cells from a set of observations $X$ at the $n$ cells. The tool
currently supports two types of observated data:
* The observed data $X$ is an $n$-by-$m$ character matrix specifying the character-states of each of $n$ cells at $m$ characters. 
* The observed data $X$ is Gillian fill in details please...
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

There are two main output files of `lamlpro`:
* 
* 

> [!TIP]
> Use the flag `--ultrametric` to ensure the cell lineage tree has equal
> length root-to-leaf paths.

## Examples

We provide simulated cell lineage trees with $n = 100, 250, 500$ nodes
and simulated observations in order to demonstrate `lamlpro`.

### Example 1: Character Matrix

To apply `lamlpro` to character matrix data, we first infer a cell lineage tree
$\mathcal{T}_0$ with $n = 250$ cells using the Neighbor Joining (NJ) algorithm. 
The initial
tree can be inferred using any method, but for the sake of the example,
we use the following command:
```
$ python scripts/neighbor_joining.py examples/n250_m30_character_matrix/character_matrix.csv examples/n250_m30_character_matrix/initial
```
This results in two files `examples/n250_m30_character_matrix/initial_hamming_tree.nwk` and
`examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk`. If we compute the
distance from the inferred and true trees, we see that both are quite far away from
the ground truth:
```
$ python scripts/metrics.py --reference examples/n250_m30_character_matrix/tree.nwk --trees examples/n250_m30_character_matrix/initial_hamming_tree.nwk examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk
Tree                                  RF Distance         Normalized RF
initial_hamming_tree.nwk              404                 0.817814
initial_weighted_hamming_tree.nwk     356                 0.720648
```

With either initial tree, one can run `lamlpro` with the following command:
```
$ lamlpro --matrix examples/n250_m30_character_matrix/character_matrix.csv --tree examples/n250_m30_character_matrix/tree.nwk -o examples/n250_m30_character_matrix/lamlpro --ultrametric --mode search --max-iterations 2500
```
The preceding command enforces that the tree is ultrametric and runs topology search for `2500`
iterations. For practical applications, we recommend setting this value higher and making
sure that the algorithm converges. The preceding command results in two output files:
`examples/n250_m30_character_matrix/lamlpro_tree.newick` and 
`examples/n250_m30_character_matrix/lamlpro_results.json`.
The first file is the inferred cell lineage tree with branch lengths and the second contains 
parameter estimates and important metadata, such as the per-iteration log-likelihood.

On this example, `lamlpro` should improve the tree topology and this can be verified by 
running:
```
$ python scripts/metrics.py --reference examples/n250_m30_character_matrix/tree.nwk --trees examples/n250_m30_character_matrix/initial_hamming_tree.nwk examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk examples/n250_m30_character_matrix/lamlpro_tree.newick
Tree                                  RF Distance         Normalized RF
initial_hamming_tree.nwk              404                 0.817814
initial_weighted_hamming_tree.nwk     356                 0.720648
laml_pro_tree.newick                  140                 0.283401
```