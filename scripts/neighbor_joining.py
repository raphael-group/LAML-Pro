#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
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

def normalized_hamming(a, b, missing=-1):
    """Disagreements over jointly-observed sites, ignoring positions missing in either
    cell. Returns nan when the pair shares no observed site -- build_tree substitutes
    the mean of the defined distances, since 0 would make those pairs the first NJ
    joins and 1 would assert they are maximally diverged."""
    both = (a != missing) & (b != missing)
    k = both.sum()
    if k == 0:
        return float("nan")
    return float((a[both] != b[both]).sum()) / k

def jaccard_edits(a, b, missing=-1, unedited=0):
    """1 - |shared edits| / |edits in either|, over jointly-observed sites only.

    Unlike normalized_hamming, co-absence contributes nothing: two cells both being
    unedited at a site is weak evidence of relatedness in an irreversible system, and
    on this data 54% of jointly-observed positions are 0/0. Two cells count as sharing
    an edit only if both are edited AND carry the same state, so this generalises to
    multi-state matrices.

    Returns nan when the pair shares no observed site, or shares sites but neither is
    edited anywhere in the overlap."""
    both = (a != missing) & (b != missing)
    if both.sum() == 0:
        return float("nan")
    x, y = a[both], b[both]
    ea, eb = x != unedited, y != unedited
    union = (ea | eb).sum()
    if union == 0:
        return float("nan")
    shared = (ea & eb & (x == y)).sum()
    return 1.0 - float(shared) / union

def distance_array(df, metric):
    """Full square distance array. Any nan the metric returns (an undefined pair) is
    replaced by the mean of the defined off-diagonal distances -- 0 would make those
    pairs the first ones joined and 1 would assert they are maximally diverged."""
    names = df.index.astype(str).tolist()
    data = df.values
    n = len(names)
    square = [[0.0 if i == j else float(metric(data[i], data[j])) for j in range(n)] for i in range(n)]
    arr = pd.DataFrame(square).values
    undefined = pd.isna(arr)
    if undefined.any():
        iu = [(i, j) for i in range(n) for j in range(i + 1, n)]
        defined = [arr[i][j] for i, j in iu if not undefined[i][j]]
        fill = sum(defined) / len(defined) if defined else 0.0
        n_undef = sum(1 for i, j in iu if undefined[i][j])
        print(f"  {n_undef} undefined pair(s) with nothing to compare; set to mean distance {fill:.6f}")
        arr[undefined] = fill
    return names, arr

def build_tree(df, metric):
    """Neighbour-joining. Returns (tree, distance_matrix). The distance matrix is the one
    NJ consumed, including the OUTGROUP row/column that main() appends."""
    names, arr = distance_array(df, metric)
    n = len(names)
    mat = [[float(arr[i][j]) for j in range(i + 1)] for i in range(n)]
    dm = DistanceMatrix(names, mat)
    return DistanceTreeConstructor().nj(dm), dm

def build_upgma(df, metric, outgroup="OUTGROUP"):
    """UPGMA (average linkage), following the construction in Martin's
    call_bcs_generate_data notebook: scipy linkage(method='average'), then a newick
    conversion where each branch length is the difference in merge heights.

    No outgroup is used -- average linkage produces a rooted ultrametric tree directly.
    Note that ties in merge height give branch length exactly 0, so a matrix with many
    zero distances yields many zero-length branches; the count is reported."""
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, to_tree

    df = df.drop(index=outgroup) if outgroup in df.index else df
    names, arr = distance_array(df, metric)
    np.fill_diagonal(arr, 0.0)
    arr = (arr + arr.T) / 2.0                      # enforce exact symmetry for squareform
    Z = linkage(squareform(arr, checks=False), method="average")

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(names) + 1000))
    zero_branches = [0]

    def to_newick(node, parent_dist, decimals=6):
        bl = parent_dist - node.dist
        if bl == 0:
            zero_branches[0] += 1
        if node.is_leaf():
            return f"{names[node.id]}:{bl:.{decimals}f}"
        left = to_newick(node.get_left(), node.dist, decimals)
        right = to_newick(node.get_right(), node.dist, decimals)
        return f"({left},{right}):{bl:.{decimals}f}"

    root = to_tree(Z, rd=False)
    newick = to_newick(root, root.dist) + ";"
    print(f"  UPGMA: {zero_branches[0]} zero-length branches of {2 * len(names) - 1}")
    return newick, arr

def write_distance_matrix(dm, path, drop="OUTGROUP"):
    """Write a square, labelled CSV. The outgroup is dropped by default so the matrix
    matches the leaf set of the pruned tree."""
    names = list(dm.names)
    square = [[dm[i, j] for j in names] for i in names]
    out = pd.DataFrame(square, index=names, columns=names)
    if drop is not None and drop in out.index:
        out = out.drop(index=drop, columns=drop)
    out.to_csv(path)
    return out.shape

def root_and_prune(tree, outgroup="OUTGROUP"):
    tree.root_with_outgroup(outgroup)
    tree.prune(outgroup)
    return tree

METRICS = {
    # missing (-1) treated as an ordinary state: ?/? scores as a match, ?/observed as a
    # mismatch. Encodes "shared missingness is evidence of common ancestry".
    "hamming": hamming,
    "weighted_hamming": weighted_hamming,
    # missing ignored: only jointly-observed sites are compared. Encodes "missingness is
    # dropout and carries no lineage information".
    "normalized_hamming": normalized_hamming,
    "jaccard_edits": jaccard_edits,
}

def main():
    parser = argparse.ArgumentParser(
        description="Construct NJ trees and distance matrices from a character matrix."
    )
    parser.add_argument("matrix", help="Input character matrix (CSV or TSV). First column = taxon name.")
    parser.add_argument("output", help="Output prefix.")
    parser.add_argument(
        "--metric", nargs="+", choices=sorted(METRICS), default=["hamming", "weighted_hamming"],
        help="Which distance(s) to build. Default reproduces the original behaviour."
    )
    parser.add_argument(
        "--method", nargs="+", choices=["nj", "upgma"], default=["nj"],
        help="Tree construction method(s). NJ writes <prefix>_<metric>_tree.nwk; "
             "UPGMA writes <prefix>_<metric>_upgma_tree.nwk."
    )
    args = parser.parse_args()

    # Auto-detect delimiter
    with open(args.matrix) as f:
        sample = f.read(2048)
    delimiter = '\t' if '\t' in sample else ','

    df = pd.read_csv(args.matrix, index_col=0, delimiter=delimiter)
    df = df.replace("?", -1).astype(int)
    df.loc["OUTGROUP"] = 0  # add all-zero outgroup

    for name in args.metric:
        print(f"[{name}]")
        if "nj" in args.method:
            tree, dm = build_tree(df, METRICS[name])
            tree = root_and_prune(tree)
            out_tree = f"{args.output}_{name}_tree.nwk"
            out_dmat = f"{args.output}_{name}_distmat.csv"
            Phylo.write(tree, out_tree, "newick")
            shape = write_distance_matrix(dm, out_dmat)
            print(f"  NJ tree written to {out_tree}")
            print(f"  distance matrix {shape} written to {out_dmat}")
        if "upgma" in args.method:
            newick, _ = build_upgma(df, METRICS[name])
            out_upgma = f"{args.output}_{name}_upgma_tree.nwk"
            with open(out_upgma, "w") as fh:
                fh.write(newick)
            print(f"  UPGMA tree written to {out_upgma}")

if __name__ == "__main__":
    main()
