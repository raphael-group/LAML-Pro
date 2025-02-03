import time
import timeit
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
    root: int,
) -> jnp.array:
    ν = model_parameters[0]
    alphabet_size = inside_log_likelihoods.shape[2]
    
    # We'll define a child function that *accepts* the current inside_ll as a parameter
    def compute_child_llh(inside_ll, z, blen, t1, t2, t3):
        # We can hoist these out if we like, but let's keep it inline for clarity:
        mask_alpha_last = jnp.arange(alphabet_size) == (alphabet_size - 1)
        llh_case1 = jnp.where(
            mask_alpha_last,
            inside_ll[:, z, -1][:, None],  # Use inside_ll, not outer scope
            0.0,
        )
        
        mask_alpha_zero = jnp.arange(alphabet_size) == 0
        summands = jnp.zeros((inside_ll.shape[0], alphabet_size))
        summands = summands.at[:, 0].set(-blen * (1 + ν))
        if alphabet_size > 2:
            summands = summands.at[:, 1 : (alphabet_size - 1)].set(
                t1 + jnp.log(mutation_priors) + t3
            )
        summands = summands.at[:, -1].set(t2)
        # Again, read from inside_ll
        logsumexp_val = jax.nn.logsumexp(summands + inside_ll[:, z, :], axis=1)
        llh_case2 = jnp.where(mask_alpha_zero, logsumexp_val[:, None], 0.0)

        mask_alpha_other = ~(mask_alpha_zero | mask_alpha_last)
        term1 = t1 + inside_ll[:, z, :]
        term2 = t2 + inside_ll[:, z, -1][:, None]
        logaddexp_result = jnp.logaddexp(term1, term2)
        llh_case3 = jnp.where(mask_alpha_other, logaddexp_result, 0.0)
        
        return llh_case1 + llh_case2 + llh_case3

    def body_fun(i, inside_ll):
        u = internal_postorder[i]
        v, w = internal_postorder_children[i]
        bv = branch_lengths[v]
        bw = branch_lengths[w]
        
        t1_v = -bv * ν
        t2_v = jnp.log(1.0 - jnp.exp(-bv * ν))
        t3_v = jnp.log(1.0 - jnp.exp(-bv))

        t1_w = -bw * ν
        t2_w = jnp.log(1.0 - jnp.exp(-bw * ν))
        t3_w = jnp.log(1.0 - jnp.exp(-bw))

        # Pass the carry (inside_ll) into compute_child_llh:
        llh_v = compute_child_llh(inside_ll, v, bv, t1_v, t2_v, t3_v)
        llh_w = compute_child_llh(inside_ll, w, bw, t1_w, t2_w, t3_w)

        # Update the carry for the next iteration:
        inside_ll = inside_ll.at[:, u, :].set(llh_v + llh_w)
        return inside_ll

    # Wrap the iteration over internal_postorder with fori_loop
    num_internal = internal_postorder.shape[0]
    inside_log_likelihoods = jax.lax.fori_loop(
        0,
        num_internal,
        body_fun,
        inside_log_likelihoods,
    )

    # Finally compute the root likelihood
    blength_root = branch_lengths[root]
    t1_root = -blength_root * ν
    t2_root = jnp.log(1.0 - jnp.exp(-blength_root * ν))
    t3_root = jnp.log(1.0 - jnp.exp(-blength_root))
    
    # Use the *final* inside_log_likelihoods from the loop:
    inside_root_llh = compute_child_llh(
        inside_log_likelihoods, 
        root, 
        blength_root, 
        t1_root, 
        t2_root, 
        t3_root
    )

    return inside_log_likelihoods, inside_root_llh

@jax.jit
def initialize_leaf_inside_log_likelihoods(
    inside_log_likelihoods : jnp.array, 
    leaves : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array
) -> jnp.array:
    num_characters = inside_log_likelihoods.shape[0]
    alphabet_size  = inside_log_likelihoods.shape[2]
    ϕ = model_parameters[1]
    
    leaf_characters = character_matrix[leaves, :]
    
    alpha_grid = jnp.arange(alphabet_size)
    
    leaf_chars_expanded = leaf_characters[..., None] # shape (L, C, 1)
    alpha_expanded      = alpha_grid[None, None, :]  # shape (1, 1, A)

    cond_1 = ((alpha_expanded == (alphabet_size - 1)) & (leaf_chars_expanded == -1)) # (1) alpha==A-1 & char==-1
    cond_2 = ((alpha_expanded == (alphabet_size - 1)) & (leaf_chars_expanded != -1)) # (2) alpha==A-1 & char!=-1
    cond_3 = (alpha_expanded == leaf_chars_expanded)                                 # (3) alpha==char
    cond_4 = (leaf_chars_expanded == -1)                                             # (4) char==-1           
    
    conditions = [cond_1, cond_2, cond_3, cond_4]
    choices    = [1.0, ϕ, 1.0 - ϕ, ϕ]
    
    leaf_inside_probs = jnp.select(conditions, choices, default=0.0)
    leaf_inside_probs_T = jnp.swapaxes(leaf_inside_probs, 0, 1)  # now (C, L, A)
    
    inside_log_likelihoods = inside_log_likelihoods.at[:, leaves, :].set(
        leaf_inside_probs_T
    )
    
    return jnp.log(inside_log_likelihoods)

@jax.jit
def compute_log_likelihood(
        leaves : jnp.array,
        internal_postorder : jnp.array,
        internal_postorder_children : jnp.array,
        inside_log_likelihoods : jnp.array,
        model_parameters : jnp.array,
        character_matrix : jnp.array,
        branch_lengths : jnp.array,
        mutation_priors : jnp.array,
        root : int
) -> jnp.array:
    inside_log_likelihoods = initialize_leaf_inside_log_likelihoods(
        inside_log_likelihoods, 
        leaves, 
        model_parameters, 
        character_matrix
    )

    inside_log_likelihoods, inside_root_llh = compute_internal_log_likelihoods(
        inside_log_likelihoods, 
        internal_postorder,
        internal_postorder_children,
        branch_lengths,
        model_parameters,
        mutation_priors,
        root
    )

    return inside_root_llh[:, 0].sum()

def compute_llh(phylo_opt):
    phylogeny = phylo_opt.phylogeny

    leaves = jnp.array([n for n in phylogeny.tree.nodes() if phylogeny.tree.out_degree(n) == 0])
    internal_postorder = [n for n in nx.dfs_postorder_nodes(phylogeny.tree, phylogeny.root) if phylogeny.tree.out_degree(n) > 0]
    internal_postorder = jnp.array(internal_postorder)
    internal_postorder_children = jnp.array([list(phylogeny.tree.successors(int(n))) for n in internal_postorder])

    return compute_log_likelihood(
        leaves, 
        internal_postorder, 
        internal_postorder_children, 
        phylo_opt.inside_log_likelihoods, 
        phylo_opt.model_parameters, 
        phylogeny.character_matrix, 
        phylo_opt.branch_lengths, 
        phylogeny.mutation_priors, 
        phylogeny.root
    )

def main(phylo_opt):
    phylogeny = phylo_opt.phylogeny

    leaves = jnp.array([n for n in phylogeny.tree.nodes() if phylogeny.tree.out_degree(n) == 0])
    internal_postorder = [n for n in nx.dfs_postorder_nodes(phylogeny.tree, phylogeny.root) if phylogeny.tree.out_degree(n) > 0]
    internal_postorder = jnp.array(internal_postorder)
    internal_postorder_children = jnp.array([list(phylogeny.tree.successors(int(n))) for n in internal_postorder])

    def llh_helper():
        return compute_log_likelihood(
            leaves, 
            internal_postorder, 
            internal_postorder_children, 
            phylo_opt.inside_log_likelihoods, 
            phylo_opt.model_parameters, 
            phylogeny.character_matrix, 
            phylo_opt.branch_lengths, 
            phylogeny.mutation_priors, 
            phylogeny.root
        )

    llh_helper().block_until_ready()
    NUM_ITER = 1000
    llh = llh_helper()
    runtime = timeit.timeit(lambda: llh_helper(), number=NUM_ITER) # timeit
    avg_runtime = runtime / NUM_ITER

    root = [n for n in phylogeny.tree.nodes() if phylogeny.tree.in_degree(n) == 0][0]
    lg.logger.info(f"Log likelihood at root {root}: {llh}")
    lg.logger.info(f"Average runtime (s): {avg_runtime}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--character_matrix", help="Character matrix.", required=True)
    p.add_argument("-t", "--tree", help="Newick tree.", required=True)
    p.add_argument("-p", "--priors", help="Mutation priors CSV.", required=True)
    p.add_argument("--nu", help="Heritable silencing rate (ν).", type=float, default=0.0)
    p.add_argument("--phi", help="Sequencing dropout rate (ϕ).", type=float, default=0.0)
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
    model_parameters = jnp.array([args.nu, args.phi])
    phylo_opt = phylogeny.PhylogenyOptimization(
        phylogeny=phylo, 
        branch_lengths=branch_lengths, 
        model_parameters=model_parameters,
        inside_log_likelihoods=jnp.zeros((phylo.num_characters, phylo.num_leaves * 2 - 1, phylo.max_alphabet_size + 2), dtype=jnp.float32)
    )

    main(phylo_opt)
