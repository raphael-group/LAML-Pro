import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os

# Use custom style
plt.style.use(os.path.expanduser("~/plotting/paper.mplstyle"))

# Parse arguments
parser = argparse.ArgumentParser(description='Plot log likelihoods from multiple results.json files')
parser.add_argument('results', nargs='+', type=str, help='Paths to JSON files with log_likelihoods list')
parser.add_argument('--labels', nargs='*', help='Optional labels for each curve')
args = parser.parse_args()

# Set up plot
plt.figure(figsize=(6, 4))

# Iterate through each results file
for i, filepath in enumerate(args.results):
    with open(filepath, 'r') as f:
        data = json.load(f)


    log_likelihoods = data['log_likelihoods']
    iterations = list(range(len(log_likelihoods)))
    print(f"{filepath}: {len(log_likelihoods)} iterations")

    df = pd.DataFrame({
        'Iteration': iterations,
        'Log Likelihood': log_likelihoods,
        'Run': args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(os.path.dirname(filepath))
    })

    plt.plot(df['Iteration'], df['Log Likelihood'], label=df['Run'].iloc[0], linewidth=1.0)

 
    plt.scatter(
        df['Iteration'].iloc[-1],
        df['Log Likelihood'].iloc[-1],
        marker='x',
        color='black',
        zorder=5
    )



# Finalize plot
plt.title('Log Likelihood vs Iteration')
plt.xlabel('Simulated Annealing Iterations')
plt.ylabel('Log Likelihood')
plt.legend(frameon=True, facecolor='white', edgecolor='black')

plt.grid(True)
plt.tight_layout()
plt.show()

