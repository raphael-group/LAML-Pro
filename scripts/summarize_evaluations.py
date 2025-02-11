import argparse
import os
import json
import re

import pandas as pd

def parse_fast_laml_results(res_path):
    row = {}

    with open(res_path, 'r') as f:
        res = json.load(f)
        row['nllh'] = res['nllh']
        row['runtime'] = res['runtime']
        row['est_nu'] = res['nu']
        row['est_phi'] = res['phi']
        row['compile_time'] = res['compile_time']

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

    return row

def evaluate_algo(algo, algo_res_path):
    rows = []

    for subdir in os.listdir(algo_res_path):
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
            res_path = os.path.join(algo_res_path, subdir, 'fast_laml_results.json')
            row.update(parse_fast_laml_results(res_path))
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
