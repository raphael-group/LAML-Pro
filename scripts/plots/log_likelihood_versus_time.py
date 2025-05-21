import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import argparse 
import json

plt.style.use("~/plotting/paper.mplstyle")   
# use tex
plt.rc('text', usetex=True) 

parser = argparse.ArgumentParser(description='Plot log likelihood versus time')
parser.add_argument('results', type=str, help='JSON file with results')
args = parser.parse_args()

with open(args.results, 'r') as f:
    data = json.load(f)

    # Extract data for plotting
    iterations = list(range(len(data['log_likelihoods'])))
    log_likelihoods = data['log_likelihoods']
    initial_ll = log_likelihoods[0]  # Get the initial log likelihood value

    # Create a DataFrame for seaborn
    plot_data = pd.DataFrame({
        'Iteration': iterations,
        'Log Likelihood': log_likelihoods
    })

    # Create the plot
    plt.figure(figsize=(6, 4))
    sns.lineplot(x='Iteration', y='Log Likelihood', data=plot_data, linewidth=1.0, color='orange')
    
    # Add horizontal line at initial value
    plt.axhline(y=initial_ll, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.title('Log Likelihood vs Iteration')
    plt.xlabel('Simulated Annealing Iterations')
    plt.ylabel('Log Likelihood')
    plt.grid(True)
    plt.tight_layout()
    plt.show()