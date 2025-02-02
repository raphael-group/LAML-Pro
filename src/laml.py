import time
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

""" 
LAML: Lineage Analysis with Maximum Likelihood 

Code specific to LAML for computing the inside log likelihoods.
is included in the JIT-compiled functions:
    - `compute_internal_log_likelihoods`.
    - `initialize_leaf_inside_log_likelihoods`.

In the subsequent code, we assume that the alphabet
is ordered {0, 1, ..., A, -1}, where A is the size 
of the alphabet, 0 is the missing state, 1, ..., A
are non-missing states, and -1 is the unknown (?) state.
"""
@jax.jit
def compute_internal_log_likelihoods(
    inside_log_likelihoods: jnp.array,
    internal_postorder: jnp.array,
    internal_postorder_children: jnp.array,
    branch_lengths: jnp.array,
    model_parameters: jnp.array,
    mutation_priors: jnp.array,
    root : int
) -> jnp.array:
    alphabet_size = inside_log_likelihoods.shape[2]

    def compute_child_llh(z, blen, t1, t2, t3):
        # Vectorized computation for each child's contribution
        mask_alpha_last = jnp.arange(alphabet_size) == (alphabet_size - 1)
        llh_case1 = jnp.where(mask_alpha_last, inside_log_likelihoods[:, z, -1][:, None], 0.0)
        
        mask_alpha_zero = jnp.arange(alphabet_size) == 0
        # Compute summands for alpha=0
        summands = jnp.zeros((inside_log_likelihoods.shape[0], alphabet_size))
        summands = summands.at[:, 0].set(-blen * (1 + model_parameters[0]))
        if alphabet_size > 2:
            summands = summands.at[:, 1:(alphabet_size - 1)].set(
                t1 + jnp.log(mutation_priors) + t3
            )
        summands = summands.at[:, -1].set(t2)
        logsumexp_val = jax.nn.logsumexp(summands + inside_log_likelihoods[:, z, :], axis=1)
        llh_case2 = jnp.where(mask_alpha_zero, logsumexp_val[:, None], 0.0)
        
        mask_alpha_other = ~ (mask_alpha_zero | mask_alpha_last)
        term1 = t1 + inside_log_likelihoods[:, z, :]
        term2 = t2 + inside_log_likelihoods[:, z, -1][:, None]
        logaddexp_result = jnp.logaddexp(term1, term2)
        llh_case3 = jnp.where(mask_alpha_other, logaddexp_result, 0.0)
        
        return llh_case1 + llh_case2 + llh_case3

    for i, u in enumerate(internal_postorder):
        v, w = internal_postorder_children[i]
        bv = branch_lengths[v]
        bw = branch_lengths[w]

        t1_v = -bv * model_parameters[0]
        t2_v = jnp.log(1 - jnp.exp(-bv * model_parameters[0]))
        t3_v = jnp.log(1 - jnp.exp(-bv))

        t1_w = -bw * model_parameters[0]
        t2_w = jnp.log(1 - jnp.exp(-bw * model_parameters[0]))
        t3_w = jnp.log(1 - jnp.exp(-bw))

        llh_v = compute_child_llh(v, bv, t1_v, t2_v, t3_v)
        llh_w = compute_child_llh(w, bw, t1_w, t2_w, t3_w)

        # Update the inside log likelihoods for the current node
        inside_log_likelihoods = inside_log_likelihoods.at[:, u, :].set(llh_v + llh_w)

    blength_root = branch_lengths[root]
    t1_root = -blength_root * model_parameters[0]
    t2_root = jnp.log(1 - jnp.exp(-blength_root * model_parameters[0]))
    t3_root = jnp.log(1 - jnp.exp(-blength_root))
    inside_root_llh = compute_child_llh(root, blength_root, t1_root, t2_root, t3_root)

    return inside_log_likelihoods, inside_root_llh

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

    # normalize mutation priors
    phylogeny.mutation_priors = phylogeny.mutation_priors / phylogeny.mutation_priors.sum(axis=1)[:, None]
    leaves = jnp.array([n for n in phylogeny.tree.nodes() if phylogeny.tree.out_degree(n) == 0])
    internal_postorder = [n for n in nx.dfs_postorder_nodes(phylogeny.tree, phylogeny.root) if phylogeny.tree.out_degree(n) > 0]
    internal_postorder = jnp.array(internal_postorder)
    internal_postorder_children = jnp.array([list(phylogeny.tree.successors(int(n))) for n in internal_postorder])

    # initialize the inside log likelihoods for the leaves
    start = time.time()
    phylo_opt.inside_log_likelihoods = initialize_leaf_inside_log_likelihoods(
        phylo_opt.inside_log_likelihoods, 
        leaves, 
        phylo_opt.model_parameters, 
        phylogeny.character_matrix
    )

    phylo_opt.inside_log_likelihoods, inside_root_llh = compute_internal_log_likelihoods(
        phylo_opt.inside_log_likelihoods, 
        internal_postorder,
        internal_postorder_children,
        phylo_opt.branch_lengths,
        phylo_opt.model_parameters,
        phylogeny.mutation_priors,
        phylogeny.root
    )
    end = time.time()
    print(f"Time taken: {end - start}")

    root = [n for n in phylogeny.tree.nodes() if phylogeny.tree.in_degree(n) == 0][0]
    print(f"Node {root}: {phylogeny.tree.nodes[root]}")
    print(inside_root_llh)

    # for n in phylogeny.tree.nodes():
        # if phylogeny.tree.in_degree(n) != 0:
            # print(f"Node {n}: {phylogeny.tree.nodes[n]}")
            # print(jnp.exp(phylo_opt.inside_log_likelihoods[:, n, :]))

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
    model_parameters = jnp.array([0.0, 0.0]) # for LAML the model parameters are [ν, ϕ]
    phylo_opt = phylogeny.PhylogenyOptimization(
        phylogeny=phylo, 
        branch_lengths=branch_lengths, 
        model_parameters=model_parameters,
        inside_log_likelihoods=jnp.zeros((phylo.num_characters, phylo.num_leaves * 2 - 1, phylo.max_alphabet_size + 2), dtype=jnp.float32)
    )

    main(phylo_opt)
