import dendropy
import dendropy.calculate.phylogeneticdistance as phylodist
import numpy as np
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
        compute_distance_matrix(reference_fname)
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

def compute_distance_matrix(tree_fname):
    try:
        tree = dendropy.Tree.get(path=tree_fname, schema='newick')
        tree.deroot()
        M = tree.phylogenetic_distance_matrix()
    except Exception as e:
        raise ValueError(f"Error loading reference tree: {str(e)}")

    distance_vec = []
    for t1, t2 in M.distinct_taxon_pair_iter():
        distance_vec.append((t1.label, t2.label, M.distance(t1, t2)))
    sorted_distance_vec = sorted(distance_vec)
    sorted_distance_vec = np.array([x[2] for x in sorted_distance_vec])
    return sorted_distance_vec

def compute_branch_length_distances(reference_fname, tree_fnames):
    true_dists = compute_distance_matrix(reference_fname)
    results = []
    for tree_path in tree_fnames:
        inferred_dists = compute_distance_matrix(tree_path)
        scale_factor = (inferred_dists.T @ inferred_dists) / (inferred_dists.T @ true_dists)
        dist = np.linalg.norm(true_dists * scale_factor - inferred_dists)

        results.append({
            'tree': tree_path,
            'distance': dist
        })

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
    parser.add_argument(
        '--metric',
        help='Metric to compute (default: rf)',
        choices=['rf', 'branch_length_distances']
    )
    args = parser.parse_args()

    if args.metric == 'branch_length_distances':
        res = compute_branch_length_distances(args.reference, args.trees)
        print(f"{'Tree':<50} {'Distance':>10}")
        for r in res:
            print(f"{r['tree']:<50} {r['distance']:10.6f}")
        return
    else:
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

if __name__ == "__main__":
    main()
