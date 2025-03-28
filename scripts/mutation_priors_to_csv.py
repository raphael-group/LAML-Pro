import argparse
import pickle
import sys
import os

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Process mutation priors file.')
    parser.add_argument('input_file', help='Input mutation priors file (pickle format)')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file (default: stdout)')
    return parser.parse_args()

def load_mutation_priors(input_file):
    try:
        with open(input_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading mutation priors file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    args = parse_args()
        
    # Load the mutation priors file
    mutation_priors = load_mutation_priors(args.input_file)
    rows = []
    for character in mutation_priors:
        for state, prob in mutation_priors[character].items():
            rows.append({
                'character': character,
                'state': state,
                'probability': prob
            })

    mutation_priors_df = pd.DataFrame(rows)
    mutation_priors_df = mutation_priors_df.sort_values(by=['character', 'state'])
    mutation_priors_df = mutation_priors_df.reset_index(drop=True)

    # Output to CSV
    if args.output:
        mutation_priors_df.to_csv(args.output, index=False, header=False)
    else:
        mutation_priors_df.to_csv(sys.stdout, index=False, header=False)
if __name__ == "__main__":
    main()