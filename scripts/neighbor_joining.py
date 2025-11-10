#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

def hamming(a, b):
    return (a != b).sum()

def weighted_hamming(a, b):
    s = 0
    for x, y in zip(a, b):
        if x == y:
            continue
        if x > 0 and y > 0:
            s += 2
        else:
            s += 1
    return s

def build_tree(df, metric):
    names = df.index.tolist()
    data = df.values
    n = len(names)
    mat = [[0.0 if i == j else float(metric(data[i], data[j])) for j in range(i + 1)] for i in range(n)]
    return DistanceTreeConstructor().nj(DistanceMatrix(names, mat))

def root_and_prune(tree, outgroup="OUTGROUP"):
    tree.root_with_outgroup(outgroup)
    tree.prune(outgroup)
    return tree

def main():
    parser = argparse.ArgumentParser(
        description="Construct Hamming and Weighted-Hamming NJ trees from a character matrix."
    )
    parser.add_argument("matrix", help="Input character matrix (CSV or TSV). First column = taxon name.")
    parser.add_argument("output", help="Output prefix.")
    args = parser.parse_args()

    # Auto-detect delimiter
    with open(args.matrix) as f:
        sample = f.read(2048)
    delimiter = '\t' if '\t' in sample else ','

    df = pd.read_csv(args.matrix, index_col=0, delimiter=delimiter)
    df = df.replace("?", -1).astype(int)
    df.loc["OUTGROUP"] = 0  # add all-zero outgroup

    # Build and process trees
    tree_h = root_and_prune(build_tree(df, hamming))
    tree_w = root_and_prune(build_tree(df, weighted_hamming))

    out_h = f"{args.output}_hamming_tree.nwk"
    out_w = f"{args.output}_weighted_hamming_tree.nwk"

    Phylo.write(tree_h, out_h, "newick")
    Phylo.write(tree_w, out_w, "newick")

    print(f"Hamming NJ tree written to {out_h}")
    print(f"Weighted-Hamming NJ tree written to {out_w}")

if __name__ == "__main__":
    main()
