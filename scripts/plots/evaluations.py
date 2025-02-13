import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.style.use("~/plotting/paper.mplstyle")   
# use tex
plt.rc('text', usetex=True) 

algorithm_rename_map = {
    'fast-laml': 'fastLAML-JAX',
    'laml': 'LAML',
}

df = pd.read_csv("evaluations.csv")
df['algorithm'] = df['algorithm'].map(algorithm_rename_map)
df['(cells, alphabet size)'] = '(' + df['num_cells'].astype(str) + ', ' + df['alphabet_size'].astype(str) + ')'
df = df.sort_values(['num_cells', 'alphabet_size'])

df_pivot = df.pivot_table(index=['num_chars', 'alphabet_size', 'seed_prior', 'num_cells', 'seq_prior'], 
                          columns='algorithm', values='nllh').reset_index()
df_pivot['nllh_diff'] = df_pivot['fastLAML-JAX'] - df_pivot['LAML']

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

# Plot runtime
sns.boxplot(x='(cells, alphabet size)', y='runtime', hue='algorithm', data=df, ax=ax)
ax.set_xlabel('(cells, alphabet size)')
ax.set_ylabel('Runtime (s)')
ax.set_yscale('log')
ax.legend(title='Algorithm', loc='lower right')
ax.xaxis.set_tick_params(rotation=45)
fig.tight_layout()

fig, axes = plt.subplots(2, 2, figsize=(5, 5))

num_cells = [250, 500, 1000, 10000]
for i, ax in enumerate(axes.flatten()):
    sns.histplot(x='nllh_diff', data=df_pivot[df_pivot['num_cells'] == num_cells[i]], ax=ax, bins=40)
    ax.set_xlabel('$\\Delta$-NLLH (fastLAML-JAX - LAML)')
    ax.set_title(f'$n = {num_cells[i]}$ cells')
    ax.xaxis.set_tick_params(rotation=45)

fig.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.boxplot(x='(cells, alphabet size)', y='est_nu', hue='algorithm', data=df, ax=axes[0])
sns.boxplot(x='(cells, alphabet size)', y='est_phi', hue='algorithm', data=df, ax=axes[1])
axes[0].set_xlabel('(cells, alphabet size)')
axes[0].set_ylabel('Estimated $\\nu$')
axes[1].set_xlabel('(cells, alphabet size)')
axes[1].set_ylabel('Estimated $\\phi$')
axes[0].axhline(y=0.134, color='black', linestyle='--', linewidth=1)
axes[1].axhline(y=0.143, color='black', linestyle='--', linewidth=1)
axes[1].legend().remove()
axes[0].xaxis.set_tick_params(rotation=45)
axes[1].xaxis.set_tick_params(rotation=45)

fig.tight_layout()

print(df)

plt.show()