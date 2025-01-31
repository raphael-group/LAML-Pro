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

def main(phylo_opt, mutation_priors):
    pass

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
    phylo_opt = phylogeny.PhylogenyOptimization(phylogeny=phylo, branch_lengths=branch_lengths, model_parameters=model_parameters)

    main(phylo_opt, priors)
