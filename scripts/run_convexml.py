#!/n/fs/ragr-data/users/schmidt/miniconda3/envs/breaked/envs/convexml-env/bin/python

import argparse
import os
import pandas as pd
import convexml
from Bio import Phylo

def main():
    parser = argparse.ArgumentParser(
        description="Computes branch lengths on a fixed tree using ConvexML."
    )
    parser.add_argument("matrix", help="Input character matrix (CSV or TSV).")
    parser.add_argument("tree", help="Path to tree (Newick).")
    parser.add_argument("output", help="Output file.")
    args = parser.parse_args()

    df = pd.read_csv(args.matrix, index_col=0)
    df = df.replace("?", -1).astype(int)

    leaf_sequences = {}
    for i in range(len(df)):
        leaf_name = str(df.iloc[i].name)
        sequence = [int(x) for x in df.iloc[i].to_numpy()]
        leaf_sequences[leaf_name] = sequence

    with open(args.tree, "r") as f:
        tree_newick_str = f.read()
    
    res = convexml.convexml(tree_newick_str, leaf_sequences)
    with open(args.output, "w") as f:
        f.write(res['tree_newick'])

if __name__ == "__main__":
    main()
