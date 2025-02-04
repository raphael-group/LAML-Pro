import time
import timeit
import argparse
import math
import random
import sys
import jax
import phylogeny
import os

import numpy as np
import pandas as pd
import networkx as nx
import jax.numpy as jnp
import loguru as lg
import optimistix as optx


from collections import defaultdict

EPS = 1e-6
GRAD_STEP_SIZE = 5e-4
GRAD_STEPS = 250

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

def compute_internal_log_likelihoods_depthwise(
    inside_log_likelihoods: jnp.array,
    internal_postorder: jnp.array,
    internal_postorder_children: jnp.array,
    branch_lengths: jnp.array,
    model_parameters: jnp.array,
    mutation_priors: jnp.array,
    root: int,
):
    """Computes log-likelihoods for internal nodes via a depth-based traversal strategy.

    Args:
        inside_log_likelihoods: Array of shape (num_characters, num_nodes, alphabet_size)
            storing current log-likelihoods; will be updated for internal nodes.
        internal_postorder: Array of shape (num_internal_nodes, 2) where each row is 
            (node_id, depth) specifying post-order traversal order and depth information.
        internal_postorder_children: Array of shape (num_internal_nodes, 2) where each row 
            contains the child node indices for the corresponding internal node in internal_postorder.
        branch_lengths: Array of shape (num_nodes,) storing branch lengths for each node.
        model_parameters: Array where model_parameters[1] is ϕ (dropout rate).
        mutation_priors: Array of shape (alphabet_size,) with prior probabilities for mutations.
        root: Integer index of the root node.

    Returns:
        Tuple containing:
        - Updated inside_log_likelihoods array with internal node values.
        - Array of shape (num_characters, alphabet_size) with root node log-likelihoods.

    Notes:
        Uses a depth-based traversal strategy and JAX vectorization for efficient 
        computation, taking O(num_characters * num_nodes * alphabet_size * depth) time,
        rather than the standard O(num_characters * num_nodes * alphabet_size^2) time, but 
        performs each of the O(depth) steps completely in parallel.
    """

    # depth = internal_postorder[:, 1].max()
    depth = 10
    ν = model_parameters[0]
    num_characters, num_nodes, alphabet_size = inside_log_likelihoods.shape

    log_mutation_priors = jnp.log(mutation_priors)

    minus_blen_times_nu = -branch_lengths * ν
    t1_array = minus_blen_times_nu
    t2_array = jnp.log(1.0 - jnp.exp(minus_blen_times_nu))
    t3_array = jnp.log(1.0 - jnp.exp(-branch_lengths))
    alpha0_array = -branch_lengths * (1.0 + ν)  # for alpha=0 summand

    def compute_child_llh(inside_ll, child_idx):
        t1 = t1_array[child_idx]
        t2 = t2_array[child_idx]
        t3 = t3_array[child_idx]
        alpha0_val = alpha0_array[child_idx]

        col = inside_ll[:, child_idx, :]  # => (num_characters, alphabet_size)
        out = jnp.zeros_like(col)

        out = out.at[:, -1].set(col[:, -1])
        summands = jnp.zeros_like(col)
        summands = summands.at[:, 0].set(alpha0_val)
        if alphabet_size > 2:
            summands = summands.at[:, 1:-1].set(t1 + log_mutation_priors + t3)
        summands = summands.at[:, -1].set(t2)

        val_for_alpha0 = jax.nn.logsumexp(summands + col, axis=1)
        out = out.at[:, 0].set(val_for_alpha0)

        if alphabet_size > 2:
            addexp_all = jnp.logaddexp(t1 + col, t2 + col[:, -1, None])
            out = out.at[:, 1:-1].set(addexp_all[:, 1:-1])

        return out

    def body_fun(i, inside_ll):
        u = internal_postorder[i, 0]
        v, w = internal_postorder_children[i]
        llh_v = compute_child_llh(inside_ll, v)
        llh_w = compute_child_llh(inside_ll, w)
        return llh_v + llh_w

    i_vector = jnp.arange(internal_postorder.shape[0])
    update_f = lambda inside_ll: jax.vmap(lambda i: body_fun(i, inside_ll))

    def fori_body(d, inside_ll):
        cond = (internal_postorder[:, 1] == depth - d)[:, None, None]  # shape (249, 1, 1)

        all_updates = jnp.where(
            cond,
            update_f(inside_ll)(i_vector),
            inside_ll[:, internal_postorder[:, 0], :].transpose(1, 0, 2)
        )
        return inside_ll.at[:, internal_postorder[:,0], :].set(all_updates.transpose(1, 0, 2))

    inside_log_likelihoods = jax.lax.fori_loop(0, depth + 1, fori_body, inside_log_likelihoods)

    inside_root_llh = compute_child_llh(inside_log_likelihoods, root)
    return inside_log_likelihoods, inside_root_llh

def initialize_leaf_inside_log_likelihoods(
    inside_log_likelihoods : jnp.array, 
    leaves : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array
) -> jnp.array:
    """Initializes leaf node log-likelihoods based on observed character states.

    Args:
        inside_log_likelihoods: Array of shape (num_characters, num_nodes, alphabet_size)
            to be populated with leaf node likelihoods.
        leaves: Array of leaf node indices to initialize.
        model_parameters: Array with [ν, ϕ] (heritable silencing rate, sequencing dropout rate).
        character_matrix: Array of shape (num_leaves, num_characters) storing observed states,
            where values are in {0 (missing), 1,...,A-1 (valid states), -1 (unknown)}.

    Returns:
        Array of same shape as inside_log_likelihoods with leaf log-likelihoods initialized.
    """

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
    choices    = [1.0, 0.0, 1.0 - ϕ, ϕ]
    choices    = jnp.maximum(jnp.array(choices), EPS)
    
    leaf_inside_probs = jnp.select(conditions, choices, default=0.0)
    leaf_inside_probs_T = jnp.swapaxes(leaf_inside_probs, 0, 1)  # now (C, L, A)
    
    inside_log_likelihoods = inside_log_likelihoods.at[:, leaves, :].set(
        leaf_inside_probs_T
    )
    
    return jnp.log(inside_log_likelihoods)

def compute_log_likelihood(
        branch_lengths : jnp.array,
        mutation_priors : jnp.array,
        leaves : jnp.array,
        internal_postorder : jnp.array,
        internal_postorder_children : jnp.array,
        inside_log_likelihoods : jnp.array,
        model_parameters : jnp.array,
        character_matrix : jnp.array,
        root : int
) -> jnp.array:
    """Computes the total log-likelihood of observed character matrix given 
    a phylogeny with branch lengths and model parameters.

    Args:
        branch_lengths: Array of shape (num_nodes,) with branch lengths.
        mutation_priors: Array of shape (alphabet_size,) with mutation priors.
        leaves: Array of leaf node indices.
        internal_postorder: Post-order traversal info for internal nodes.
        internal_postorder_children: Child mappings for internal nodes.
        inside_log_likelihoods: Pre-allocated array for likelihood computations.
        model_parameters: Array with [ν, ϕ] (heritable silencing rate, sequencing dropout rate).
        character_matrix: Observed states array of shape (num_leaves, num_characters).
        root: Root node index.

    Returns:
        Scalar sum of log-likelihoods over all characters at the root node.
    """

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

def optimize_parameters(
    leaves : jnp.array,
    internal_postorder : jnp.array,
    internal_postorder_children : jnp.array,
    inside_log_likelihoods : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array,
    branch_lengths : jnp.array,
    mutation_priors : jnp.array,
    root : int
    ):

    log_branch_lengths = jnp.log(branch_lengths)
    logit_model_parameters = jnp.log(model_parameters / (1.0 - model_parameters))

    def loss_fn(parameters, args):
        log_branch_lengths, logit_model_parameters = parameters
        return -compute_log_likelihood(
            jnp.exp(log_branch_lengths), 
            mutation_priors, 
            leaves, 
            internal_postorder, 
            internal_postorder_children, 
            inside_log_likelihoods, 
            jax.nn.sigmoid(logit_model_parameters),
            character_matrix, 
            root
        )

    starting_params = (log_branch_lengths, logit_model_parameters)
    solver = optx.BFGS(atol=1e-4, rtol=1e-4)
    optx.minimise(loss_fn, solver, starting_params)

    return jax.exp(log_branch_lengths), jax.nn.sigmoid(logit_model_parameters)

def main(mode, phylo_opt):
    phylogeny = phylo_opt.phylogeny

    leaves = jnp.array([n for n in phylogeny.tree.nodes() if phylogeny.tree.out_degree(n) == 0])
    level_order = nx.single_source_shortest_path_length(phylogeny.tree, phylogeny.root)
    internal_postorder = [[n, level_order[n]] for n in nx.dfs_postorder_nodes(phylogeny.tree, phylogeny.root) if phylogeny.tree.out_degree(n) > 0]
    internal_postorder = jnp.array(internal_postorder)
    internal_postorder_children = jnp.array([list(phylogeny.tree.successors(int(n))) for n in internal_postorder[:, 0]])

    if mode == "score":
        def llh_helper():
            return compute_log_likelihood(
                phylo_opt.branch_lengths,
                phylogeny.mutation_priors, 
                leaves, 
                internal_postorder, 
                internal_postorder_children, 
                phylo_opt.inside_log_likelihoods, 
                phylo_opt.model_parameters,
                phylogeny.character_matrix, 
                phylogeny.root
            )

        llh_helper = jax.jit(llh_helper, backend="cpu")
        llh_helper().block_until_ready()
        NUM_ITER = 100
        llh = llh_helper()
        runtime = timeit.timeit(lambda: llh_helper().block_until_ready(), number=NUM_ITER)
        avg_runtime = runtime / NUM_ITER

        root = [n for n in phylogeny.tree.nodes() if phylogeny.tree.in_degree(n) == 0][0]
        lg.logger.info(f"Log likelihood at root {root}: {llh}")
        lg.logger.info(f"Average runtime (s): {avg_runtime}")
        return llh
    elif mode == "optimize":
        branch_lengths, model_parameters = optimize_parameters(
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
        lg.logger.info(f"Optimized branch lengths: {branch_lengths}")
        lg.logger.info(f"Optimized model parameters: {model_parameters}")
        return branch_lengths, model_parameters

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--character_matrix", help="Character matrix.", required=True)
    p.add_argument("-t", "--tree", help="Newick tree.", required=True)
    p.add_argument("-p", "--priors", help="Mutation priors CSV.")
    p.add_argument("--nu", help="Heritable silencing rate (ν).", type=float, default=0.0)
    p.add_argument("--phi", help="Sequencing dropout rate (ϕ).", type=float, default=0.0)
    p.add_argument("--mode", help="Algorithm mode.", default="score", choices=["score", "optimize", "time_llh"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tree, n = phylogeny.parse_newick(args.tree)

    character_matrix = pd.read_csv(args.character_matrix, sep=",", index_col=0)
    character_matrix.index = character_matrix.index.astype(str)
    character_matrix.replace("?", -1, inplace=True)
    character_matrix = character_matrix.astype(int)

    if args.priors is None:
        lg.logger.info("No mutation priors provided. Assuming uniform priors.")
        rows = []
        for i, c in enumerate(character_matrix.columns):
            states = set(character_matrix[c].unique()) - set([0, -1])
            num_states = len(states)
            for s in states:
                rows.append({"character": i, "state": s, "probability": 1.0 / num_states})
        priors = pd.DataFrame(rows)
        priors.character = priors.character.astype(int)
        priors.state = priors.state.astype(int)
        priors.set_index(["character", "state"], inplace=True)
    else:
        priors = pd.read_csv(args.priors, sep=",", header=None)
        priors.columns = ["character", "state", "probability"]
        priors.character = priors.character.astype(int)
        priors.state = priors.state.astype(int)
        priors.set_index(["character", "state"], inplace=True)

    if n != character_matrix.shape[0]:
        lg.logger.error("The tree and character matrix have different numbers of taxa.")
        sys.exit(1)

    phylo = phylogeny.build_phylogeny(tree, n, character_matrix, priors)

    if any(tree.nodes[i]["branch_length"] is None for i in range(2 * n - 1)):
        lg.logger.error("Some branch lengths are missing. Setting them to 1.0.")
        branch_lengths = jnp.ones(2 * n - 1)
    else:
        branch_lengths = jnp.array([tree.nodes[i]["branch_length"] for i in range(2 * n - 1)])

    model_parameters = jnp.array([args.nu, args.phi])
    phylo_opt = phylogeny.PhylogenyOptimization(
        phylogeny=phylo, 
        branch_lengths=branch_lengths, 
        model_parameters=model_parameters,
        inside_log_likelihoods=jnp.zeros((phylo.num_characters, phylo.num_leaves * 2 - 1, phylo.max_alphabet_size + 2), dtype=jnp.float32)
    )

    main(args.mode, phylo_opt)
