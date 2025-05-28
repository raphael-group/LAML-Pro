import pandas as pd
import numpy as np
import math
import sys


# Read your data
df = pd.read_csv(sys.argv[1], index_col=0)

### assumes that missing data is '?'
df.replace('?', np.nan, inplace=True)

cassette_idx = 0
states = list(range(4))  # States 0-9
log_low_prob = -1e7 #math.log(0.04) # hardcoding to be the level of white noise
log_high_prob = math.log(1.0)  # This will just be 0.0
# np.random.normal(loc=0, scale=1, size=1000)

rows = []
for cell_name, row in df.iterrows():
    for target_site, observed_state in row.items():
        if pd.isna(observed_state):
            continue  # Skip missing data
        observed_state = int(observed_state)
        target_site = int(target_site[1:])

        entry = {
            'cell_name': cell_name,
            'cassette_idx': cassette_idx,
            'target_site': target_site
        }

        for state in states:
            if state == observed_state:
                logp = log_high_prob
            else:
                logp = log_low_prob
            entry[f'state{state}_probability'] = logp
        rows.append(entry)

obs_df = pd.DataFrame(rows)
obs_df.to_csv(sys.argv[2], index=False)


