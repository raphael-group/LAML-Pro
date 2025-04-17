import dendropy
import sys

def compute_rfs(reference_fname, tree_fnames):
    """
    Compute Robinson-Foulds distances between trees and a reference tree.
    
    Args:
        reference_fname (str): Path to reference tree file in Newick format
        tree_fnames (list): List of paths to tree files to compare to reference
        
    Returns:
        list: List of dictionaries with RF distance metrics for each tree
    """
    # Load reference tree
    try:
        reference = dendropy.Tree.get(path=reference_fname, schema='newick')
        reference.deroot()  # Ensure unrooted for RF calculation
    except Exception as e:
        raise ValueError(f"Error loading reference tree: {str(e)}")

    # Get reference taxa information
    reference_taxa = set(leaf.taxon.label for leaf in reference.leaf_nodes())
    n_taxa = len(reference_taxa)
    if n_taxa < 4:
        raise ValueError("Reference tree must have at least 4 taxa")
    reference_internal = len(reference.internal_edges(exclude_seed_edge=True))

    results = []
    for tree_path in tree_fnames:
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
                raise ValueError(f"Tree {tree_path} has different taxa from reference")

            # Calculate RF distances
            rf = dendropy.calculate.treecompare.symmetric_difference(reference, tree)
            tree_internal = len(tree.internal_edges(exclude_seed_edge=True))
            normalized_rf = rf / (reference_internal + tree_internal)

            results.append({
                'tree': tree_path,
                'rf_distance': rf,
                'normalized_rf_distance': normalized_rf
            })

        except Exception as e:
            raise ValueError(f"Error processing {tree_path}: {str(e)}")

    return results

def main():
    import argparse
    
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

    try:
        results = compute_rfs(args.reference, args.trees)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as fh:
                fh.write("Tree,RF_Distance,Normalized_RF\n")
                for res in results:
                    fh.write(f"{res['tree']},{res['rf_distance']},{res['normalized_rf_distance']:.6f}\n")
        else:
            print(f"{'Tree':<50} {'RF Distance':>10} {'Normalized RF':>12}")
            for res in results:
                print(f"{res['tree']:<50} {res['rf_distance']:10d} {res['normalized_rf_distance']:12.6f}")
    
    except ValueError as e:
        sys.exit(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
