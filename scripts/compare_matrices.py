import pandas as pd
import numpy as np
import sys
import ast

# Check command-line arguments
if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <character_matrix_pp.csv> <LAMLpro_posterior_argmax.csv>")
    sys.exit(1)

file1 = sys.argv[1]  # character_matrix_pp.csv
file2 = sys.argv[2]  # posterior_argmax.csv

def read_matrix_smart(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    skiprows = 2 if first_line.startswith("Newick Tree:") else 0
    df = pd.read_csv(filepath, index_col=0, skiprows=skiprows)
    return df

# Load the CSVs
char_matrix_df = read_matrix_smart(file1)
lamlpro_df = read_matrix_smart(file2)

# Sort to align rows and columns
char_matrix_df = char_matrix_df.sort_index(axis=0).sort_index(axis=1)
lamlpro_df = lamlpro_df.sort_index(axis=0).sort_index(axis=1)

char_matrix_df.index = char_matrix_df.index.astype(str)
char_matrix_df.columns = char_matrix_df.columns.astype(str)
lamlpro_df.index = lamlpro_df.index.astype(str)
lamlpro_df.columns = lamlpro_df.columns.astype(str)

missing_set = {-1, "-1", "?"}
# Recode all missing entries to -1
char_matrix_df.replace(to_replace=missing_set, value=-1, inplace=True)
lamlpro_df.replace(to_replace=missing_set, value=-1, inplace=True)

# Match only the cell names in char_matrix_df, since we may also output ancestral nodes
shared_cells = char_matrix_df.index.intersection(lamlpro_df.index)

if len(shared_cells) == 0:
    print("Error: No matching cell names found between the two matrices!")
    sys.exit(1)

#print(shared_cells)

# Subset both to only shared cells
char_matrix_df = char_matrix_df.loc[shared_cells]
lamlpro_df = lamlpro_df.loc[shared_cells]

#tuples = [ast.literal_eval(col) for col in char_matrix_df.columns]
#tuples = [tuple(map(int, t)) for t in tuples]
#tuples = 
#char_matrix_df.columns = pd.MultiIndex.from_tuples(
#    tuples,
#    names=["cassette_idx", "target_site"]
#)
#char_matrix_df = char_matrix_df.sort_index(axis=1, level="cassette_idx")

#print("character_matrix_df")
#print(char_matrix_df.columns)
#print("lamlpro_df")
#print(lamlpro_df)

# Confirm dimensions now match
if char_matrix_df.shape != lamlpro_df.shape:
    print(f"Error: Shapes still do not match after subsetting.")
    print(f"Character matrix shape: {char_matrix_df.shape}")
    print(f"LAMLpro matrix shape: {lamlpro_df.shape}")
    sys.exit(1)

# Elementwise comparison

print("Assuming missing values are encoded as -1...")
comparison_matrix = (lamlpro_df.values == char_matrix_df.values)
#print(lamlpro_df)
#print(char_matrix_df)
#print(comparison_matrix)
missing_mask = (lamlpro_df.values == -1) | (char_matrix_df.values == -1)
#print(missing_mask)
total_elements = comparison_matrix.size

# Calculate % imputed
num_missing = np.sum(missing_mask)
frac_missing = num_missing / total_elements

# Identify where a true disagreement occurs
true_disagreement = (~comparison_matrix) & (~missing_mask)
num_true_disagree = np.sum(true_disagreement)

num_agree = np.sum(comparison_matrix)
frac_agreement = (num_agree / total_elements) 
frac_true_disagreement = (num_true_disagree / total_elements) 

print(f"Comparison Results (with imputation handling):")
print(f"  Total elements: {total_elements}")
print(f"  Agree elements: {num_agree}")
print(f"  Disagree elements: {num_true_disagree}")
print(f"  Missing elements: {num_missing}")
print(f"  Frac agreement: {frac_agreement}")
print(f"  Frac true disagreement: {frac_true_disagreement}")

# Generate and save difference matrix
#difference_matrix = (lamlpro_df != char_matrix_df).astype(int)
#difference_matrix.index = shared_cells
#difference_matrix.columns = lamlpro_df.columns

#output_filename = "difference_matrix.csv"
#difference_matrix.to_csv(output_filename)
#print(f"Difference matrix saved to {output_filename}")

