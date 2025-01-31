import argparse
import math
import random
import sys
import treeswift
import pandas as pd
import networkx as nx

from loguru import logger

"""
Parses a Newick tree into a NetworkX DiGraph, labeling each 
node with an integer index and storing the node  name and 
branch length into each node as attributes. 

Requirement: Each leaf node must have a label.
"""
def parse_newick(newick):
    swift_tree = treeswift.read_tree_newick(args.tree)
    tree = nx.DiGraph()
    for idx, v in enumerate(swift_tree.traverse_preorder()):
        label = v.get_label()
        if label is None and v.is_leaf():
            logger.error("Leaf node has no label.")
            sys.exit(1)
        elif label is None:
            label = f"node_{str(idx)}"

        tree.add_node(idx, name=label, branch_length=v.get_edge_length())
        v.set_label(idx)

        if v.is_root(): continue
        tree.add_edge(v.get_parent().get_label(), idx)

    return tree

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--character_matrix", help="Character matrix.", required=True)
    p.add_argument("-t", "--tree", help="Newick tree.", required=True)
    p.add_argument("-p", "--priors", help="Mutation priors CSV.", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tree = parse_newick(args.tree)
    character_matrix = pd.read_csv(args.character_matrix, sep=",", index_col=0)
    priors = pd.read_csv(args.priors, sep=",", header=None)
    priors.columns = ["character", "state", "probability"]

    n = len(character_matrix)
    if 2*n != len(tree.nodes()):
        logger.error("The tree is not binary: the number of nodes in tree does not match number of taxa.")
        sys.exit(1)

    m = len(character_matrix.columns)
