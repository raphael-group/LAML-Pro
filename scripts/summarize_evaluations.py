import argparse
import os
import json
import re
import treeswift
from tqdm import tqdm

import pandas as pd

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

def parse_fast_laml_results(subdir):
    res_path = os.path.join(subdir, 'fast_laml_results.json')
    row = {}

    with open(res_path, 'r') as f:
        res = json.load(f)
        row['nllh'] = res['nllh']
        row['runtime'] = res['runtime']
        row['est_nu'] = res['nu']
        row['est_phi'] = res['phi']
        row['compile_time'] = res['compile_time']
        if 'em_iterations' in res:
            row['em_iterations'] = res['em_iterations']

    timing_file = os.path.join(subdir, 'timing.txt')
    if os.path.exists(timing_file):
        res = parse_timing_txt(timing_file)
        row = row | res

    true_newick_file      = os.path.join(subdir, 'simulated_tree.newick')
    estimated_newick_file = os.path.join(subdir, 'fast_laml_tree.newick')
    
    if os.path.exists(true_newick_file) and os.path.exists(estimated_newick_file):
        true_bls = read_branch_lengths(true_newick_file)
        est_bls  = read_branch_lengths(estimated_newick_file)

        l1_bl_error = 0.0
        l2_bl_error = 0.0
        for node, bl in true_bls.items():
            l1_bl_error += abs(bl - est_bls[node])
            l2_bl_error += (bl - est_bls[node]) ** 2

        row['l1_branch_length_error'] = l1_bl_error
        row['l2_branch_length_error'] = l2_bl_error

    return row

def parse_laml_results(subdir):
    row = {}

    with open(os.path.join(subdir, 'LAML_output.log'), 'r') as f:
        lines = f.readlines()
        runtime = float(lines[-1].split()[-1])
        row['runtime'] = runtime

    with open(os.path.join(subdir, 'LAML_output_params.txt'), 'r') as f:
        lines = f.readlines()
        row['est_phi'] = float(lines[0].split()[-1])
        row['est_nu'] = float(lines[1].split()[-1])
        row['nllh'] = eval(lines[2].split(':')[-1].strip())
        row['status'] = row['nllh'][1]
        row['nllh'] = row['nllh'][0]

    timing_file = os.path.join(subdir, 'timing.txt')
    if os.path.exists(timing_file):
        res = parse_timing_txt(timing_file)
        row = row | res

    true_newick_file      = os.path.join(subdir, 'simulated_tree.newick')
    estimated_newick_file = os.path.join(subdir, 'LAML_output_trees.nwk')
    
    if os.path.exists(true_newick_file) and os.path.exists(estimated_newick_file):
        true_bls = read_branch_lengths(true_newick_file)
        est_bls  = read_branch_lengths(estimated_newick_file)

        l1_bl_error = 0.0
        l2_bl_error = 0.0
        for node, bl in true_bls.items():
            l1_bl_error += abs(bl - est_bls[node])
            l2_bl_error += (bl - est_bls[node]) ** 2

        row['l1_branch_length_error'] = l1_bl_error
        row['l2_branch_length_error'] = l2_bl_error

    return row

def evaluate_algo(algo, algo_res_path):
    rows = []

    for subdir in tqdm(os.listdir(algo_res_path)):
        num_chars, alphabet_size, seed_prior, num_cells, seq_prior = re.match(r'k(\d+)M(\d+)p(\d+)_\w+_sub(\d+)_r(\d+)', subdir).groups()
        row = {
            'num_chars': num_chars, 
            'alphabet_size': alphabet_size, 
            'seed_prior': seed_prior, 
            'num_cells': num_cells, 
            'seq_prior': seq_prior,
            'algorithm': algo
        }

        if 'fast' in algo:
            row.update(parse_fast_laml_results(os.path.join(algo_res_path, subdir)))
        else:
            row.update(parse_laml_results(os.path.join(algo_res_path, subdir)))

        rows.append(row)

    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize evaluations')
    parser.add_argument('directory', type=str, help='Directory containing evaluations')
    args = parser.parse_args()

    dataframe_rows = []
    for directory in os.listdir(args.directory):
        algo_res_path = os.path.join(args.directory, directory)
        if not os.path.isdir(algo_res_path):
            continue

        rows = evaluate_algo(directory, algo_res_path)
        dataframe_rows.extend(rows)

    df = pd.DataFrame(dataframe_rows)
    df.to_csv('evaluations.csv', index=False)
