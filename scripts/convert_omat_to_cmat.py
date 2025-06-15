import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("/Users/gc3045/laml2_experiments/BaseMEM_Magic/gillian/baseMemoir_data_clean_reformatted_pos2.csv")

# Identify the state columns
state_cols = [col for col in df.columns if col.startswith("state")]

# Compute the argmax (most probable state)
df["most_probable_state"] = df[state_cols].values.argmax(axis=1)

df["cassette_idx"] = df["cassette_idx"].astype(int)
df["target_site"] = df["target_site"].astype(int)

pivot = df.pivot_table(
    index="cell_name",
    columns=["cassette_idx", "target_site"],
    values="most_probable_state"
)

pivot = pivot.sort_index(axis=1, level=[0, 1])
pivot.columns = [f"{i}_{j}" for i, j in pivot.columns]
pivot = pivot.fillna(-1).astype(int)


# Save to new CSV
pivot.to_csv("baseMemoir_argmax_states.csv", index=True)

