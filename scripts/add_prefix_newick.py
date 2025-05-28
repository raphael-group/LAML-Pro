import argparse, sys
from Bio import Phylo

def main():
    p = argparse.ArgumentParser(
        description="Append PREFIX to every label in a Newick tree."
    )
    p.add_argument("newick_file", help="Path to input .nwk/.newick file")
    p.add_argument("prefix",      help="Prefix to prepend to each clade label")
    args = p.parse_args()

    # read → modify labels → write
    tree = Phylo.read(args.newick_file, "newick")
    for clade in tree.find_clades():
        if clade.name:                      # skip unnamed internal nodes
            clade.name = f"{args.prefix}{clade.name}"

    Phylo.write(tree, sys.stdout, "newick")

if __name__ == "__main__":
    main()
