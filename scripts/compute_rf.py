import argparse
import dendropy
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Compute Robinson-Foulds distances between phylogenetic trees and a reference tree.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--reference', required=True,
        help='Reference tree file in Newick format'
    )
    parser.add_argument(
        '--trees', required=True, nargs='+',
        help='List of tree files to compare to reference'
    )
    parser.add_argument(
        '--output', 
        help='Output CSV file (default: print to stdout)'
    )
    args = parser.parse_args()

    # Load reference tree
    try:
        reference = dendropy.Tree.get(path=args.reference, schema='newick')
        reference.deroot()  # Ensure unrooted for RF calculation
    except Exception as e:
        sys.exit(f"Error loading reference tree: {str(e)}")

    # Get reference taxa information
    reference_taxa = set(leaf.taxon.label for leaf in reference.leaf_nodes())
    n_taxa = len(reference_taxa)
    if n_taxa < 4:
        sys.exit("Error: Reference tree must have at least 4 taxa")
    reference_internal = len(reference.internal_edges(exclude_seed_edge=True))

    results = []
    for tree_path in args.trees:
        try:
            # Load tree using reference's taxon namespace
            tree = dendropy.Tree.get(
                path=tree_path,
                schema='newick',
                taxon_namespace=reference.taxon_namespace
            )
            tree.deroot()  # Ensure unrooted for RF calculation
            
            # Verify taxon consistency
            tree_taxa = set(leaf.taxon.label for leaf in tree.leaf_nodes())
            if tree_taxa != reference_taxa:
                sys.exit(f"Error: Tree {tree_path} has different taxa from reference")

            # Calculate RF distances
            rf = dendropy.calculate.treecompare.symmetric_difference(reference, tree)
            tree_internal = len(tree.internal_edges(exclude_seed_edge=True))
            normalized_rf = rf / (reference_internal + tree_internal)

            results.append({
                'tree': tree_path,
                'rf': rf,
                'normalized_rf': normalized_rf
            })

        except Exception as e:
            sys.exit(f"Error processing {tree_path}: {str(e)}")

    # Output results
    if args.output:
        with open(args.output, 'w') as fh:
            fh.write("Tree,RF_Distance,Normalized_RF\n")
            for res in results:
                fh.write(f"{res['tree']},{res['rf']},{res['normalized_rf']:.6f}\n")
    else:
        print(f"{'Tree':<50} {'RF Distance':>10} {'Normalized RF':>12}")
        for res in results:
            print(f"{res['tree']:<50} {res['rf']:10d} {res['normalized_rf']:12.6f}")


if __name__ == "__main__":
    main()
