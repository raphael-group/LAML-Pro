import argparse
import os
import json
import re
import treeswift
import metrics

from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd

_FILENAME_RE = re.compile(
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

def parse_directory_name(fname):
    """
    Extract experiment parameters from a filename.
    """

    m = _FILENAME_RE.match(fname)
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

def parse_fast_laml_results(true_newick_file, starting_tree_newick, subdir):
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

    estimated_newick_file = os.path.join(subdir, 'fastlaml_ultrametric_tree.nwk')
    rfs = metrics.compute_rfs(true_newick_file, [estimated_newick_file, starting_tree_newick])
    bldistances = metrics.compute_branch_length_distances(true_newick_file, [estimated_newick_file, starting_tree_newick])

    row['rf_distance'] = rfs[0]['rf_distance']
    row['normalized_rf_distance'] = rfs[0]['normalized_rf_distance']
    row['branch_length_error'] = bldistances[0]['distance']
    row['initial_rf_distance'] = rfs[1]['rf_distance']
    row['initial_normalized_rf_distance'] = rfs[1]['normalized_rf_distance']

    return row

if __name__ == '__main__':
    simulations_dir = "/n/fs/ragr-research/projects/laml-pro/sim_data/set_3d/input/"

    jobs = []
    for directory in os.listdir("nextflow_results/fast-laml"):
        data = parse_directory_name(directory)
        instance_id = f"k{data['characters']}_s{data['silencing_rate']}_sub{data['cells']}_r{data['replicate']}"
        simulation_dir = os.path.join(simulations_dir, instance_id)
        for subdirectory in os.listdir(os.path.join("nextflow_results/fast-laml", directory)):
            true_tree = os.path.join(simulation_dir, "tree.nwk")
            starting_tree = f"nextflow_results/starting_trees/{instance_id}_r{data['gaussian_distance']}_p{data['dropout_rate']}"
            starting_tree = os.path.join(starting_tree, subdirectory)
            jobs.append({
                "data": data,
                "true_tree": true_tree, 
                "starting_tree": starting_tree, 
                "subdirectory": os.path.join("nextflow_results/fast-laml", directory, subdirectory),
                "starting_tree_type": subdirectory
            })

    def evaluate_job(job):
        row = parse_fast_laml_results(job['true_tree'], job['starting_tree'], job['subdirectory'])
        row = row | job["data"]
        row['starting_tree'] = job["starting_tree_type"]
        print(row)
        return row

    with Pool(64) as p:
        rows = p.map(evaluate_job, jobs)

    df = pd.DataFrame(rows)
    df.to_csv("observation_model_results.csv")
