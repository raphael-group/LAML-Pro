import time
import timeit
import argparse
import math
import random
import sys
import jax
import phylogeny
import os
import json

import numpy as np
import pandas as pd
import networkx as nx
import jax.numpy as jnp
import loguru as lg
import optimistix as optx
import equinox.internal as eqxi

import calculations as calc
from collections import defaultdict
from typing import Callable

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

ABSOLUTE_TOLERANCE = 1e-1
RELATIVE_TOLERANCE = 1e-1

def optimize_parameters(
    i,
    leaves : jnp.array,
    internal_postorder : jnp.array,
    internal_postorder_children : jnp.array,
    parent_sibling : jnp.array,
    level_order : jnp.array,
    inside_log_likelihoods : jnp.array,
    model_parameters : jnp.array,
    character_matrix : jnp.array,
    branch_lengths : jnp.array,
    mutation_priors : jnp.array,
    root : int
):

    model_parameters = jnp.maximum(model_parameters, calc.EPS / (i + 1))
    model_parameters = jnp.minimum(model_parameters, 1.0 - calc.EPS / (i + 1))

    logit_model_parameters = jnp.log(model_parameters / (1.0 - model_parameters))
    log_branch_lengths = jnp.log(jnp.maximum(branch_lengths, calc.EPS))

    def loss_fn(parameters, args):
        log_branch_lengths, logit_model_parameters = parameters
        return -calc.compute_log_likelihood(
            jnp.exp(log_branch_lengths), 
            mutation_priors, 
            leaves, 
            internal_postorder, 
            internal_postorder_children, 
            parent_sibling,
            level_order,
            inside_log_likelihoods, 
            jax.nn.sigmoid(logit_model_parameters),
            character_matrix, 
            root
        )

    starting_params = (log_branch_lengths, logit_model_parameters)
    solver = optx.BFGS(atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE)#, verbose=frozenset({"step_size", "loss"}))
    res = optx.minimise(loss_fn, solver, starting_params, max_steps=2500)

    branch_lengths = jnp.exp(res.value[0])
    model_parameters = jax.nn.sigmoid(res.value[1])
    nllh = -loss_fn(res.value, None)
    return nllh, branch_lengths, model_parameters

def main(mode, phylo_opt):
    phylogeny = phylo_opt.phylogeny

    leaves = jnp.array([n for n in phylogeny.tree.nodes() if phylogeny.tree.out_degree(n) == 0])
    level_order = nx.single_source_shortest_path_length(phylogeny.tree, phylogeny.root)
    internal_postorder = [[n, level_order[n]] for n in nx.dfs_postorder_nodes(phylogeny.tree, phylogeny.root) if phylogeny.tree.out_degree(n) > 0]
    internal_postorder = jnp.array(internal_postorder)
    internal_postorder_children = jnp.array([list(phylogeny.tree.successors(int(n))) for n in internal_postorder[:, 0]])
    level_order_jax = jnp.array([level_order[n] for n in range(2 * phylogeny.num_leaves - 1)])
    
    parent_sibling = []
    for i in range(2 * phylogeny.num_leaves - 1):
        if phylogeny.tree.in_degree(i) == 0:
            parent_sibling.append([-1, -1])
            continue
        parent = list(phylogeny.tree.predecessors(i))[0]
        siblings = list(phylogeny.tree.successors(parent))
        siblings.remove(i)
        parent_sibling.append([parent, siblings[0]])
    parent_sibling = jnp.array(parent_sibling)

    if mode == "score":
        def llh_helper():
            return calc.compute_log_likelihood(
                phylo_opt.branch_lengths,
                phylogeny.mutation_priors, 
                leaves, 
                internal_postorder, 
                internal_postorder_children, 
                parent_sibling,
                level_order_jax,
                phylo_opt.inside_log_likelihoods, 
                phylo_opt.model_parameters,
                phylogeny.character_matrix, 
                phylogeny.root
            )

        llh_helper = jax.jit(llh_helper)
        llh_helper().block_until_ready()
        NUM_ITER = 200
        llh = llh_helper()
        runtime = timeit.timeit(lambda: llh_helper().block_until_ready(), number=NUM_ITER)
        avg_runtime = runtime / NUM_ITER

        root = [n for n in phylogeny.tree.nodes() if phylogeny.tree.in_degree(n) == 0][0]
        lg.logger.info(f"Log likelihood at root {root}: {llh}")
        lg.logger.info(f"Average runtime (s): {avg_runtime}")
    elif mode == "optimize":
        def optimize_helper(i):
            return optimize_parameters(
                i,
                leaves, 
                internal_postorder, 
                internal_postorder_children, 
                parent_sibling,
                level_order_jax,
                phylo_opt.inside_log_likelihoods, 
                phylo_opt.model_parameters, 
                phylogeny.character_matrix, 
                phylo_opt.branch_lengths, 
                phylogeny.mutation_priors, 
                phylogeny.root
            )

        start = time.time()
        optimize_helper = jax.jit(optimize_helper)
        optimize_helper(0)[0].block_until_ready()
        end = time.time()
        compile_time = end - start

        end = time.time()

        start = time.time()
        nllh, branch_lengths, model_parameters = optimize_helper(0)
        nllh.block_until_ready()
        end = time.time()

        lg.logger.info(f"Compile time (s): {compile_time}, Optimization time (s): {end - start}")
        lg.logger.info(f"Optimized negative log likelihood(s): {nllh}")
        lg.logger.info(f"Optimized branch lengths: {branch_lengths}")
        lg.logger.info(f"Optimized ν: {model_parameters[0]}, Optimized ϕ: {model_parameters[1]}")

        with open(f"{args.output}_results.json", "w") as f:
            res = {
                "nllh": -nllh.item(),
                "nu": model_parameters[0].item(),
                "phi": model_parameters[1].item(),
                "runtime": end - start,
                "compile_time": compile_time
            }

            f.write(json.dumps(res))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--character_matrix", help="Character matrix.", required=True)
    p.add_argument("-t", "--tree", help="Newick tree.", required=True)
    p.add_argument("-p", "--priors", help="Mutation priors CSV.")
    p.add_argument("-o", "--output", help="Prefix for output files.", default="output")
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

    if any(tree.nodes[i]["branch_length"] is None for i in range(2 * n - 1)) or args.mode == "optimize":
        if args.mode == "optimize":
            lg.logger.info("Optimization mode. Initializing all branch lengths to 1.0.")
        else:
            lg.logger.error("Some branch lengths are missing. Initializing all branch lengths to 1.0.")
        branch_lengths = jnp.ones(2 * n - 1)
    else:
        branch_lengths = jnp.array([tree.nodes[i]["branch_length"] for i in range(2 * n - 1)])

    lg.logger.info(f"Using JAX backend with {jax.devices()} devices.")
    lg.logger.info(f"Using device {jax.devices()[-1]} for computation.")
    jax.config.update("jax_default_device", jax.devices()[-1])

    depths = nx.single_source_shortest_path_length(tree, phylo.root)
    max_depth = max(depths.values())
    calc.DEPTH = max_depth

    lg.logger.info(f"Tree has {n} taxa and {2 * n - 1} nodes.")
    lg.logger.info(f"Tree depth: {max_depth}")
    lg.logger.info(f"Character matrix has {character_matrix.shape[1]} characters and an alphabet size of {phylo.max_alphabet_size}.")
    model_parameters = jnp.array([args.nu, args.phi])
    phylo_opt = phylogeny.PhylogenyOptimization(
        phylogeny=phylo, 
        branch_lengths=branch_lengths, 
        model_parameters=model_parameters,
        inside_log_likelihoods=jnp.zeros((phylo.num_characters, phylo.num_leaves * 2 - 1, phylo.max_alphabet_size + 2), dtype=jnp.float32)
    )

    main(args.mode, phylo_opt)