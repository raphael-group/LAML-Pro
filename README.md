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
Tree                                  RF Distance         Normalized RF
initial_hamming_tree.nwk              404                 0.817814
initial_weighted_hamming_tree.nwk     356                 0.720648
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
python scripts/metrics.py --reference examples/n250_m30_character_matrix/tree.nwk --trees examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk examples/n250_m30_character_matrix/initial_weighted_hamming_tree.nwk examples/n250_m30_character_matrix/lamlpro_tree.newick
```

Example output:
```
Tree                                     RF Distance    Normalized RF
initial_hamming_tree.nwk                 404            0.817814
lamlpro_tree.hamming.newick              316            0.753036
initial_weighted_hamming_tree.nwk        356            0.720648
lamlpro_tree.weighted_hamming.newick     316            0.639676
lamlpro_tree.tree.newick                 136            0.275304
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
`examples//n100_m400_observation_matrixlamlpro_tree.newick` and 
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
