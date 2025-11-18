import argparse
import os
import json
import re
import treeswift
import metrics

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import pandas as pd

_FILENAME_RE_FASTLAML = re.compile(
    r"""
    ^k(?P<characters>\d+)
    _s(?P<silencing_rate>\d+)
    _sub(?P<cells>\d+)
    _r(?P<replicate>\d+)
    _r(?P<gaussian_distance>[0-9.]+)
    _p(?P<dropout_rate>[0-9.]+)
    \.(?P<datatype>.+)$
    """,
    re.VERBOSE,
)

_FILENAME_RE = re.compile(
    r"""
    ^k(?P<characters>\d+)
    _s(?P<silencing_rate>\d+)
    _sub(?P<cells>\d+)
    _r(?P<replicate>\d+)
    _r(?P<gaussian_distance>[0-9.]+)
    _p(?P<dropout_rate>[0-9.]+)
    """,
    re.VERBOSE,
)

def parse_directory_name(fname, _RE):
    """
    Extract experiment parameters from a filename.
    """

    m = _RE.match(fname)
    if m is None:
        raise ValueError(f"Filename '{fname}' does not match expected pattern")

    groups = m.groupdict()

    groups["characters"]         = int(groups["characters"])
    groups["silencing_rate"]     = int(groups["silencing_rate"])
    groups["cells"]              = int(groups["cells"])
    groups["gaussian_distance"]  = float(groups["gaussian_distance"])
    groups["dropout_rate"]       = float(groups["dropout_rate"])

    return groups

def parse_timing_txt(path):
    with open(path) as f:
        timing = f.read().strip()
        match = re.search(r'Percent of CPU this job got: (.*)%', timing)
        cpu_usage = match.groups()[0]
        match = re.search(r'Maximum resident set size \(kbytes\): (.*)', timing)
        memory_kb = match.groups()[0]
    return {
        'memory_usage': memory_kb, 'cpu_usage': cpu_usage
    }

def read_branch_lengths(newick):
    swift_tree = treeswift.read_tree_newick(newick)
    node_to_branch_lengths = {}
    for v in swift_tree.traverse_preorder():
        node_to_branch_lengths[v.get_label()] = v.get_edge_length()
    return node_to_branch_lengths

def parse_fast_laml_results(true_character_matrix, argmax_character_matrix, true_newick_file, starting_tree_newick, subdir):
    res_path = os.path.join(subdir, 'fastlaml_results.json')
    row = {}

    with open(res_path, 'r') as f:
        res = json.load(f)
        row['initial_log_likelihood'] = res['log_likelihoods'][0]
        row['best_log_likelihood'] = res['best_log_likelihood']
        row['runtime'] = res['runtime_ms']
        row['est_nu'] = res['nu']
        row['est_phi'] = res['phi']

    timing_file = os.path.join(subdir, 'timing.txt')
    if os.path.exists(timing_file):
        res = parse_timing_txt(timing_file)
        row = row | res

    inferred_cm_file = os.path.join(subdir, 'fastlaml_posterior_argmax.csv')
    if os.path.exists(inferred_cm_file):
        with open(inferred_cm_file, "r") as f:
            next(f)
            next(f)
            inferred_cm = pd.read_csv(f)
        true_cm = pd.read_csv(true_character_matrix)
        true_cm = true_cm.set_index('cell')
        true_cm = true_cm.replace('?', -1).astype(int)

        argmax_cm = pd.read_csv(argmax_character_matrix).set_index('cell')
        argmax_cm = argmax_cm.loc[true_cm.index]
        argmax_cm = argmax_cm.replace('?', -1).astype(int)

        inferred_cm = inferred_cm.set_index('node')
        inferred_cm = inferred_cm.loc[true_cm.index.astype(str)].astype(int)

        true_cm = true_cm.to_numpy()
        all_missing = np.all(true_cm == -1, axis=0)
        true_cm = true_cm[:, ~all_missing]

        inferred_cm = inferred_cm.to_numpy()

        argmax_cm = argmax_cm.to_numpy()
        argmax_cm = argmax_cm[:, ~all_missing]

        row['num_genotype_error'] = np.sum(true_cm != inferred_cm)
        row['fraction_genotype_error'] = row['num_genotype_error'] / (true_cm.shape[0] * true_cm.shape[1])
        row['argmax_num_genotype_error'] = np.sum(true_cm != argmax_cm)
        row['argmax_fraction_genotype_error'] = row['argmax_num_genotype_error'] / (true_cm.shape[0] * true_cm.shape[1])

    estimated_newick_file = os.path.join(subdir, 'fastlaml_tree.nwk')
    rfs = metrics.compute_rfs(true_newick_file, [estimated_newick_file, starting_tree_newick])
    bldistances = metrics.compute_branch_length_distances(true_newick_file, [estimated_newick_file, starting_tree_newick])

    row['rf_distance'] = rfs[0]['rf_distance']
    row['normalized_rf_distance'] = rfs[0]['normalized_rf_distance']
    row['branch_length_error'] = bldistances[0]['distance']
    row['branch_length_r2'] = bldistances[0]['r2']
    row['initial_rf_distance'] = rfs[1]['rf_distance']
    row['initial_normalized_rf_distance'] = rfs[1]['normalized_rf_distance']

    return row

def parse_generic_results(true_newick_file, inferred_tree_file, subdir):
    row = {}

    timing_file = os.path.join(subdir, 'timing.txt')
    if os.path.exists(timing_file):
        res = parse_timing_txt(timing_file)
        row = row | res

    rfs = metrics.compute_rfs(true_newick_file, [inferred_tree_file])
    bldistances = metrics.compute_branch_length_distances(true_newick_file, [inferred_tree_file])

    row['rf_distance'] = rfs[0]['rf_distance']
    row['normalized_rf_distance'] = rfs[0]['normalized_rf_distance']
    row['branch_length_error'] = bldistances[0]['distance']
    row['branch_length_r2'] = bldistances[0]['r2']

    return row

if __name__ == '__main__':
    simulations_dir = "/n/fs/ragr-research/projects/laml-pro/sim_data/set_3d/input/"

    jobs = []
    for directory in os.listdir("nextflow_results/fast-laml/"):
        data = parse_directory_name(directory, _FILENAME_RE_FASTLAML)
        instance_id = f"k{data['characters']}_s{data['silencing_rate']}_sub{data['cells']}_r{data['replicate']}"
        simulation_dir = os.path.join(simulations_dir, instance_id)
        for subdirectory in os.listdir(os.path.join("nextflow_results/fast-laml", directory)):
            starting_tree = f"nextflow_results/starting_trees/{instance_id}_r{data['gaussian_distance']}_p{data['dropout_rate']}"
            starting_tree = os.path.join(starting_tree, subdirectory)
            true_tree = os.path.join(simulation_dir, "tree.nwk")
            true_character_matrix = os.path.join(simulation_dir, "character_matrix.csv")
            argmax_character_matrix = os.path.join(
                simulation_dir, 
                f"k{data['characters']}_s{data['silencing_rate']}_sub{data['cells']}_r{data['replicate']}_dim3_r{data['gaussian_distance']}_p{data['dropout_rate']}_argmax.csv"
            )
            jobs.append({
                "algorithm": "fast-laml",
                "data": data,
                "true_tree": true_tree, 
                "starting_tree": starting_tree, 
                "true_character_matrix": true_character_matrix,
                "argmax_character_matrix": argmax_character_matrix,
                "subdirectory": os.path.join("nextflow_results/fast-laml", directory, subdirectory),
                "starting_tree_type": subdirectory
            })

    for alg in ["neighbor-joining-hamming", "neighbor-joining-weighted-hamming"]:
        for directory in os.listdir(f"nextflow_results/{alg}/"):
            data = parse_directory_name(directory, _FILENAME_RE)
            instance_id = f"k{data['characters']}_s{data['silencing_rate']}_sub{data['cells']}_r{data['replicate']}"
            simulation_dir = os.path.join(simulations_dir, instance_id)
            true_tree = os.path.join(simulation_dir, "tree.nwk")
            inferred_tree = f"nextflow_results/{alg}/{instance_id}_r{data['gaussian_distance']}_p{data['dropout_rate']}/tree.nwk"
            jobs.append({
                "algorithm": alg,
                "data": data,
                "true_tree": true_tree, 
                "inferred_tree": inferred_tree, 
                "subdirectory": os.path.join("nextflow_results/", alg, directory)
            })

    for directory in os.listdir("nextflow_results/convex-ml/"):
        data = parse_directory_name(directory, _FILENAME_RE)
        instance_id = f"k{data['characters']}_s{data['silencing_rate']}_sub{data['cells']}_r{data['replicate']}"
        simulation_dir = os.path.join(simulations_dir, instance_id)
        starting_tree = f"nextflow_results/starting_trees/{instance_id}_r{data['gaussian_distance']}_p{data['dropout_rate']}"
        starting_tree = os.path.join(starting_tree, subdirectory)

        for subdirectory in os.listdir(os.path.join("nextflow_results/convex-ml", directory)):
            true_tree = os.path.join(simulation_dir, "tree.nwk")
            inferred_tree = f"nextflow_results/convex-ml/{directory}/{subdirectory}/tree.nwk"
            jobs.append({
                "algorithm": f"convex-ml-{subdirectory}",
                "data": data,
                "true_tree": true_tree, 
                "inferred_tree": inferred_tree, 
                "subdirectory": os.path.join("nextflow_results/convex-ml", directory, subdirectory),
            })

    def evaluate_job(job):
        if job["algorithm"] == "fast-laml":
            row = parse_fast_laml_results(job['true_character_matrix'], job['argmax_character_matrix'], job['true_tree'], job['starting_tree'], job['subdirectory'])
            row['starting_tree'] = job["starting_tree_type"]
        elif job["algorithm"] == "convex-ml":
            row = parse_generic_results(job['true_tree'], job['inferred_tree'], job['subdirectory'])
        elif "neighbor-joining" in job["algorithm"]:
            row = parse_generic_results(job['true_tree'], job['inferred_tree'], job['subdirectory'])
        else:
            raise ValueError("Invalid algorithm name.")

        row = row | job["data"]
        row["algorithm"] = job["algorithm"]
        return row

    with Pool(56) as p:
        rows = list(tqdm(p.imap_unordered(evaluate_job, jobs)))

    df = pd.DataFrame(rows)
    df.to_csv("topology_search_results.csv")
