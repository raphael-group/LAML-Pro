#!/usr/bin/env python3
"""Correlate pairwise distance in a tree against pairwise distance in space.

By default distances are TOPOLOGICAL (every branch set to length 1, so the distance
between two leaves is the number of edges between them). This is deliberate: methods
differ wildly in how they assign branch lengths -- UPGMA emits exact zeros, NJ emits
negatives, and LAML-Pro returns an ultrametric tree whose patristic distances compress
toward a constant -- so patristic comparisons across methods are not like-for-like.
Pass --branch-lengths to use them anyway.

Spearman is reported first. Spatial displacement grows roughly as sqrt(divergence time)
under diffusion, so the relationship is monotone but not linear, and the distance
distributions are heavily tied and skewed. Pearson is reported alongside for reference.

Note the p-values are NOT reported, and would be meaningless if they were: n leaves give
n(n-1)/2 pairs but only n independent units. Use a Mantel permutation for significance.

Example:
    python spatial_coherence.py --trees a.nwk b.nwk \\
        --coords cell_metadata.csv --label-col label --coord-cols centroid-1 centroid-2
"""

import argparse
import os

import dendropy
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def pairwise_tree_distances(tree_path, labels, use_branch_lengths=False):
    """Square matrix of leaf-to-leaf distances, ordered to match `labels`."""
    tree = dendropy.Tree.get(path=tree_path, schema="newick", preserve_underscores=True)
    if not use_branch_lengths:
        for edge in tree.edges():
            edge.length = 1.0
    pdm = tree.phylogenetic_distance_matrix()
    taxa = [tree.taxon_namespace.get_taxon(lab) for lab in labels]
    missing = [lab for lab, tx in zip(labels, taxa) if tx is None]
    if missing:
        raise ValueError(f"{len(missing)} label(s) not found in {tree_path}, e.g. {missing[:5]}")
    n = len(labels)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = pdm(taxa[i], taxa[j])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees", nargs="+", required=True, help="Newick tree file(s).")
    p.add_argument("--coords", required=True, help="CSV of cell coordinates.")
    p.add_argument("--label-col", default="label", help="Column holding leaf labels.")
    p.add_argument("--coord-cols", nargs="+", default=["centroid-1", "centroid-2"],
                   help="Coordinate columns. Only include axes on a common scale.")
    p.add_argument("--branch-lengths", action="store_true",
                   help="Use branch lengths instead of topological distance.")
    args = p.parse_args()

    meta = pd.read_csv(args.coords, index_col=0)
    labels = meta[args.label_col].astype(str).tolist()
    spatial = squareform(pdist(meta[args.coord_cols].values))
    iu = np.triu_indices(len(labels), 1)
    sv = spatial[iu]

    kind = "branch-length" if args.branch_lengths else "topological"
    print(f"{len(labels)} cells, {len(sv)} pairs, {kind} distance vs "
          f"euclidean({', '.join(args.coord_cols)})\n")
    print(f"{'tree':<52} {'spearman':>9} {'pearson':>9}")
    for path in args.trees:
        try:
            dist = pairwise_tree_distances(path, labels, args.branch_lengths)
        except ValueError as err:
            print(f"{os.path.basename(path):<52} {err}")
            continue
        dv = dist[iu]
        print(f"{os.path.basename(path):<52} {spearmanr(sv, dv)[0]:9.3f} {pearsonr(sv, dv)[0]:9.3f}")


if __name__ == "__main__":
    main()
