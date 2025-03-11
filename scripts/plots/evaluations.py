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
    'fast-laml-em': 'fastLAML-EM-GPU',
    'fast-laml-direct': 'fastLAML-Direct',
    'fast-laml-em-cpu': 'fastLAML-EM-CPU',
    'laml': 'LAML',
    'laml-ultrametric': 'LAML-Ultrametric'
}

df = pd.read_csv("evaluations.csv")
df['algorithm'] = df['algorithm'].map(algorithm_rename_map)
df['(cells, alphabet size)'] = '(' + df['num_cells'].astype(str) + ', ' + df['alphabet_size'].astype(str) + ')'
df = df.sort_values(['num_cells', 'alphabet_size'])
df = df[df['algorithm'] != 'fastLAML-Direct']  # remove direct method for now
df = df[df['algorithm'] != 'LAML-Ultrametric']  # remove direct method for now
df = df[df['num_cells'] != 10000]
df = df[(df['status'] == 'optimal') | df['status'].isna()]  # remove infeasible runs

# ONLY for runtime do i need GPU and CPU versions
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
sns.boxplot(x='(cells, alphabet size)', y='runtime', hue='algorithm', data=df, ax=ax)
ax.set_xlabel('(cells, alphabet size)')
ax.set_ylabel('Runtime (s)')
ax.set_yscale('log')
ax.legend(title='Algorithm', loc='center left', bbox_to_anchor=(1.025, 0.5))
ax.xaxis.set_tick_params(rotation=45)
fig.tight_layout(rect=[0, 0, 1.0, 1])
fig.savefig("figures/fast_laml_runtime.pdf", bbox_inches='tight')

df = df[df['algorithm'] != 'fastLAML-EM-GPU']  # remove CPU version for now
df['algorithm'] = df['algorithm'].replace('fastLAML-EM-CPU', 'fastLAML-EM')

df_pivot = df.pivot_table(index=['num_chars', 'alphabet_size', 'seed_prior', 'num_cells', 'seq_prior'], 
                          columns='algorithm', values='nllh').reset_index()
df_pivot['em_nllh_diff'] = df_pivot['fastLAML-EM'] - df_pivot['LAML']
#df_pivot['direct_nllh_diff'] = df_pivot['fastLAML-Direct'] - df_pivot['LAML']
#df_pivot['em_direct_nllh_diff'] = df_pivot['fastLAML-EM'] - df_pivot['fastLAML-Direct']

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.boxplot(x='(cells, alphabet size)', y='em_iterations', data=df[df['algorithm'] == 'fastLAML-EM'], ax=ax)
ax.set_ylabel('EM Iterations')
ax.set_xlabel('(cells, alphabet size)')
ax.xaxis.set_tick_params(rotation=45)
fig.tight_layout()
fig.savefig("figures/fast_laml_em_iterations.pdf", bbox_inches='tight')

fig, axes = plt.subplots(2, 2, figsize=(5, 5))

num_cells = [250, 500, 1000, 10000]
for i, ax in enumerate(axes.flatten()):
    sns.histplot(x='em_nllh_diff', data=df_pivot[df_pivot['num_cells'] == num_cells[i]], ax=ax, bins=40)
    ax.set_xlabel('$\\Delta$-NLLH (fastLAML-EM - LAML)')
    ax.set_title(f'$n = {num_cells[i]}$ cells')
    ax.xaxis.set_tick_params(rotation=45)

fig.tight_layout()
fig.savefig("figures/fast_laml_em_nllh_diff.pdf", bbox_inches='tight')

# fig, axes = plt.subplots(2, 2, figsize=(5, 5))

# num_cells = [250, 500, 1000, 10000]
# for i, ax in enumerate(axes.flatten()):
#     sns.histplot(x='direct_nllh_diff', data=df_pivot[df_pivot['num_cells'] == num_cells[i]], ax=ax, bins=40)
#     ax.set_xlabel('$\\Delta$-NLLH (fastLAML-Direct - LAML)')
#     ax.set_title(f'$n = {num_cells[i]}$ cells')
#     ax.xaxis.set_tick_params(rotation=45)

# fig.tight_layout()
# fig.savefig("figures/fast_laml_direct_nllh_diff.pdf", bbox_inches='tight')

# fig, axes = plt.subplots(2, 2, figsize=(5, 5))

# num_cells = [250, 500, 1000, 10000]
# for i, ax in enumerate(axes.flatten()):
#     sns.histplot(x='em_direct_nllh_diff', data=df_pivot[df_pivot['num_cells'] == num_cells[i]], ax=ax, bins=40)
#     ax.set_xlabel('$\\Delta$-NLLH (EM - Direct)')
#     ax.set_title(f'$n = {num_cells[i]}$ cells')
#     ax.xaxis.set_tick_params(rotation=45)

# fig.tight_layout()
# fig.savefig("figures/fast_laml_em_direct_nllh_diff.pdf", bbox_inches='tight')


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
fig.savefig("figures/fast_laml_estimates.pdf", bbox_inches='tight')

df['normalized_l1_branch_length_estimation_error'] = df['l1_branch_length_error'] / df['num_cells']
df['normalized_l2_branch_length_estimation_error'] = df['l2_branch_length_error'] / df['num_cells']
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.boxplot(
    x='(cells, alphabet size)',
    y='normalized_l1_branch_length_estimation_error',
    hue='algorithm',
    data=df,
    ax=axes[0],
)
sns.boxplot(
    x='(cells, alphabet size)',
    y='normalized_l2_branch_length_estimation_error',
    hue='algorithm',
    data=df,
    ax=axes[1],
)
axes[0].set_xlabel('(cells, alphabet size)')
axes[0].set_ylabel('Normalized L1 Error')
axes[1].set_xlabel('(cells, alphabet size)')
axes[1].set_ylabel('Normalized L2 Error')
axes[1].legend().remove()
axes[0].xaxis.set_tick_params(rotation=45)
axes[1].xaxis.set_tick_params(rotation=45)
fig.tight_layout()
fig.savefig("figures/fast_laml_branch_length_errors.pdf", bbox_inches='tight')

df["memory_usage_mb"] = df["memory_usage"] / 1024
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.boxplot(x="(cells, alphabet size)", y="memory_usage_mb", hue="algorithm", data=df, ax=axes[0])
sns.boxplot(x="(cells, alphabet size)", y="cpu_usage", hue="algorithm", data=df, ax=axes[1])
axes[0].set_xlabel("(cells, alphabet size)")
axes[0].set_ylabel("Memory Usage (MB)")
axes[1].set_xlabel("(cells, alphabet size)")
axes[1].set_ylabel("CPU Usage (\\%)")
axes[1].legend().remove()
axes[0].xaxis.set_tick_params(rotation=45)
axes[1].xaxis.set_tick_params(rotation=45)
fig.tight_layout()
fig.savefig("figures/fast_laml_resources.pdf", bbox_inches="tight")

plt.show()