import argparse
import math
import random
import sys
import jax
import phylogeny

import pandas as pd
import networkx as nx
import jax.numpy as jnp
import loguru as lg

### LAML specific lines of code...

""" Alphabet is ordered as {0, 1, 2, ..., max_alphabet_size, -1} """

@jax.jit
def initialize_leaf_inside_log_likelihoods(
    inside_log_likelihoods : jnp.array, 
    leaves : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array
) -> jnp.array:
    """
    Initializes the inside log likelihoods for the leaves of the phylogeny.
    """

    leaf_characters = character_matrix[leaves, :]
    leaf_characters_expanded = jnp.expand_dims(leaf_characters.T, axis=2)  # (C, L, 1)
    
    alphabet_size = inside_log_likelihoods.shape[2]
    alpha_grid = jnp.arange(alphabet_size)  # (A,)
    alpha_grid_expanded = jnp.expand_dims(alpha_grid, axis=(0, 1))  # (1, 1, A)
    
    mask_last_alpha = (alpha_grid_expanded == (alphabet_size - 1))  # (1, 1, A)
    
    value_when_last = jnp.where(
        leaf_characters_expanded == -1,
        1.0,
        model_parameters[1]
    ) # (C, L, 1)
    
    value_when_not_last = jnp.where(
        alpha_grid_expanded == leaf_characters_expanded,
        1.0 - model_parameters[1],
        0.0
    ) # (C, L, A)
    
    initial_values = jnp.where(
        mask_last_alpha,
        value_when_last,
        value_when_not_last
    ) # (C, L, A)
    
    inside_log_likelihoods = inside_log_likelihoods.at[:, leaves, :].set(initial_values)
    return jnp.log(inside_log_likelihoods)

def main(phylo_opt):
    phylogeny = phylo_opt.phylogeny

    leaves = jnp.array([n for n in phylogeny.tree.nodes() if phylogeny.tree.out_degree(n) == 0])
    internal_postorder = [n for n in nx.dfs_postorder_nodes(phylogeny.tree, phylogeny.root) if phylogeny.tree.out_degree(n) > 0]
    internal_postorder = jnp.array(internal_postorder)

    # initialize the inside log likelihoods for the leaves
    import time
    for i in range(10):
        start = time.time()
        phylo_opt.inside_log_likelihoods = initialize_leaf_inside_log_likelihoods(
                phylo_opt.inside_log_likelihoods, 
                leaves, 
                phylo_opt.model_parameters, 
                phylogeny.character_matrix
        )
        end = time.time()
        print(f"Time taken: {end - start}")

    for l in leaves:
        print(f"Leaf {phylogeny.tree.nodes[int(l)]}:\n {phylo_opt.inside_log_likelihoods[:, l, :]}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--character_matrix", help="Character matrix.", required=True)
    p.add_argument("-t", "--tree", help="Newick tree.", required=True)
    p.add_argument("-p", "--priors", help="Mutation priors CSV.", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tree, n = phylogeny.parse_newick(args.tree)

    character_matrix = pd.read_csv(args.character_matrix, sep=",", index_col=0)
    character_matrix.index = character_matrix.index.astype(str)
    character_matrix.replace("?", -1, inplace=True)
    character_matrix = character_matrix.astype(int)

    priors = pd.read_csv(args.priors, sep=",", header=None)
    priors.columns = ["character", "state", "probability"]
    priors.character = priors.character.astype(int)
    priors.state = priors.state.astype(int)
    priors.set_index(["character", "state"], inplace=True)

    if n != character_matrix.shape[0]:
        lg.logger.error("The tree and character matrix have different numbers of taxa.")
        sys.exit(1)

    phylo = phylogeny.build_phylogeny(tree, n, character_matrix, priors)
    branch_lengths = jnp.array([tree.nodes[i]["branch_length"] for i in range(2 * n - 1)])
    model_parameters = jnp.array([0.1, 0.1]) # for LAML the model parameters are [ν, ϕ]
    phylo_opt = phylogeny.PhylogenyOptimization(
        phylogeny=phylo, 
        branch_lengths=branch_lengths, 
        model_parameters=model_parameters,
        inside_log_likelihoods=jnp.zeros((phylo.num_characters, phylo.num_leaves * 2 - 1, phylo.max_alphabet_size + 2), dtype=jnp.float32)
    )

    main(phylo_opt)
