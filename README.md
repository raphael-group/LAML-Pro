# fastLAML: Fast Likelihood Approximation for Maximum Likelihood Phylogeny Inference

*fastLAML* is a method for inferring lineage trees from dynamic (e.g. CRISPR-Cas9) 
lineage tracing data. The method utilizes nearest-neighbor interchange (NNI) moves to explore the tree space and find the maximum likelihood phylogeny under the PMM model.

## Installation

#### Option 1: Build from source

`fastLAML` is implemented in C++ and requires a C++17 compliant compiler. The project uses CMake for building.

First, clone the repository with its submodules:
```
$ git clone --recurse-submodules git@github.com:raphael-group/LAML-Pro.git
```

Then build the project:
```
$ mkdir build && cd build
$ cmake ..
$ make
```

The executable will be available at fastlaml.

## Usage

To run *fastLAML*, execute the binary with the appropriate arguments:

```
$ fastlaml [--mode MODE] -t TREE -c CHARACTER_MATRIX [-m MUTATION_PRIORS] -o OUTPUT [--threads N]
```

### Modes

*fastLAML* supports two primary modes:
- `optimize`: Optimizes model parameters on a given tree
- `search`: Searches for the optimal tree by exploring the tree space

### Required Arguments

- `-t, --tree`: Path to the Newick format tree file
- `-c, --character-matrix`: Path to the character matrix file
- `-o, --output`: Path for the output file

### Optional Arguments

- `-m, --mutation-priors`: Path to the mutation priors file
- `--threads`: Number of threads to use (default: number of hardware threads)
- `--mode`: Operating mode: "optimize" or "search" (default: "optimize")

## Input Formats

### Character Matrix

The character matrix should be provided as a CSV file where:
- The first row contains character names (optional)
- The first column contains taxon names
- Each cell contains a character state (integer)
- Missing data is represented by "?" or empty cells

Example:
```
Taxa,Char1,Char2,Char3
TaxonA,1,2,0
TaxonB,1,2,?
TaxonC,0,1,1
```

### Mutation Priors

Mutation priors are specified as a CSV file with the format:
```
character_index,state,probability
```

Example:
```
0,1,0.6
0,2,0.4
1,1,0.3
1,2,0.7
```

If not provided, uniform priors will be used.

### Tree Format

Trees should be provided in Newick format. For example:
```
((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);
```

## Example

To optimize parameters on a given tree:
```
$ fastlaml --mode optimize -t example.tree -c example_matrix.csv -o results
```

To search for the optimal tree:
```
$ fastlaml --mode search -t initial_tree.tree -c example_matrix.csv -m priors.csv -o results --threads 4
```

## Technical Details

*fastLAML* implements a latent evolutionary model that:
1. Accounts for missing data in the character matrix
2. Uses expectation-maximization for parameter estimation
3. Explores tree space using nearest-neighbor interchange (NNI) moves
4. Parallelizes tree search for improved performance

The implementation handles binary phylogenetic trees and automatically resolves polytomies when encountered in the input tree.

## Performance

The runtime of *fastLAML* scales with:
- The number of characters in the matrix
- The number of taxa in the tree
- The number of NNI moves evaluated during tree search

For typical datasets with hundreds of characters and dozens of taxa, analysis completes within minutes on standard hardware when using multiple threads.
