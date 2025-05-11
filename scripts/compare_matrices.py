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

# Load the CSVs
char_matrix_df = pd.read_csv(file1, index_col=0)
lamlpro_df = pd.read_csv(file2, index_col=0, skiprows=2)

# Sort to align rows and columns
char_matrix_df = char_matrix_df.sort_index(axis=0).sort_index(axis=1)
lamlpro_df = lamlpro_df.sort_index(axis=0).sort_index(axis=1)

# Match only the cell names in char_matrix_df
shared_cells = char_matrix_df.index.intersection(lamlpro_df.index)

if len(shared_cells) == 0:
    print("Error: No matching cell names found between the two matrices!")
    sys.exit(1)

# Subset both to only shared cells
char_matrix_df = char_matrix_df.loc[shared_cells]
lamlpro_df = lamlpro_df.loc[shared_cells]

tuples = [ast.literal_eval(col) for col in char_matrix_df.columns]
tuples = [tuple(map(int, t)) for t in tuples]
char_matrix_df.columns = pd.MultiIndex.from_tuples(
    tuples,
    names=["cassette_idx", "target_site"]          # name the levels if you like
)
char_matrix_df = char_matrix_df.sort_index(axis=1, level="cassette_idx")

print("character_matrix_df")
print(char_matrix_df.columns)
print("lamlpro_df")
print(lamlpro_df)

# Confirm dimensions now match
if char_matrix_df.shape != lamlpro_df.shape:
    print(f"Error: Shapes still do not match after subsetting.")
    print(f"Character matrix shape: {char_matrix_df.shape}")
    print(f"LAMLpro matrix shape: {lamlpro_df.shape}")
    sys.exit(1)

# Elementwise comparison
comparison_matrix = (lamlpro_df.values == char_matrix_df.values)
imputed_mask = (lamlpro_df.values == -1) | (char_matrix_df.values == -1)
total_elements = comparison_matrix.size

# Calculate % imputed
num_imputed = np.sum(imputed_mask)
percent_imputed = (num_imputed / total_elements) * 100

# Identify where a true disagreement occurs
true_disagreement = (~comparison_matrix) & (~imputed_mask)

num_agree_including_imputed = np.sum(comparison_matrix | imputed_mask)
num_true_disagree = np.sum(true_disagreement)
num_imputed = np.sum(imputed_mask)

percent_agreement = (num_agree_including_imputed / total_elements) * 100
percent_true_disagreement = (num_true_disagree / total_elements) * 100
percent_imputed = (num_imputed / total_elements) * 100

print(f"Comparison Results (with imputation handling):")
print(f"  Percent agreement (including imputed): {percent_agreement:.2f}%")
print(f"  Percent true disagreement: {percent_true_disagreement:.2f}%")
print(f"  Percent imputed: {percent_imputed:.2f}%")

# Generate and save difference matrix
#difference_matrix = (lamlpro_df != char_matrix_df).astype(int)
#difference_matrix.index = shared_cells
#difference_matrix.columns = lamlpro_df.columns

#output_filename = "difference_matrix.csv"
#difference_matrix.to_csv(output_filename)
#print(f"Difference matrix saved to {output_filename}")

