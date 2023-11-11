import copy
import sys
from paretoset import paretoset
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

sys.path.extend([
    "./"
])

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
# sns.set_context("paper")
# sns.set_theme(style="ticks")

# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def identify_pareto(scores):
    """
    Identify the Pareto front in a list of scores where the first column is to be maximized
    and the second column is to be minimized.
    """
    # Count the number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the Pareto front (zero indexed)
    pareto_front = np.ones(population_size, dtype=bool)
    # Compare each pair of scores
    for i in range(population_size):
        for j in range(population_size):
            # Check if score[j] dominates score[i]
            if (scores[j][0] >= scores[i][0] and scores[j][1] <= scores[i][1]) and \
               (scores[j][0] > scores[i][0] or scores[j][1] < scores[i][1]):
                # score[i] is not on the Pareto front
                pareto_front[i] = False
                break
    # Return indices of Pareto front
    return pareto_front

if __name__ == '__main__':

    reports_dir = './reports/'
    dataset_name = 'pneumoniamnist'

    # Filename parameters.
    name_obj = 'dc_inter__dc_intra'
    n_trial = 1000
    pairwise_intra = True
    dim_reduction = False
    synth_samples_reduction = 100000
    eval_backbone = 'SwAV_torch'# 'SwAV_torch'
    post_resizer = 'friendly'
    n_samples = 4708

    # Compose filename.
    filename = f'ensemble_search-n_trial__{n_trial}-name_obj__{name_obj}-pairwise_intra__{pairwise_intra}-dim_reduction__{dim_reduction}-synth_samples_reduction__{synth_samples_reduction}-{eval_backbone}-{post_resizer}-{n_samples}'

    # Load excel data.
    ensemble_dir = os.path.join(reports_dir, dataset_name, 'ensemble', filename)
    df = pd.read_excel(os.path.join(ensemble_dir, 'optuna_study.xlsx'), engine='openpyxl')

    # Find Pareto.
    df_vals = df[['values_0', 'values_1']].to_numpy()
    mask = paretoset(df_vals, sense=["max", "min"])
    df_vals_paretoset = df_vals[mask]
    df_paretoset = df[mask]
    # Save Pareto solutions.
    df_paretoset.to_excel(os.path.join(ensemble_dir, 'pareto_solutions.xlsx'), index=False)

    # Maximise inter.
    best_inter_idx = np.argmax(df_vals_paretoset[:, 0])
    best_inter_sol = df_vals_paretoset[best_inter_idx, :]

    # Minimise intra.
    best_intra_idx = np.argmin(df_vals_paretoset[:, 1])
    best_intra_sol = df_vals_paretoset[best_intra_idx, :]

    # Minimise number of gans.
    df_paretoset['n'] = df_paretoset.filter(like='params_').sum(axis=1)
    best_n_idx = df_paretoset['n'].idxmin()
    best_n_sol = df_paretoset.loc[best_n_idx]

    # Scatter plot.
    fig = plt.figure()
    plt.scatter(df_vals[:, 0], df_vals[:, 1], linewidths=None, color='gray', label='All solutions', alpha=0.5)
    plt.scatter(df_vals_paretoset[:, 0], df_vals_paretoset[:, 1], color='red', label='Pareto solutions')
    plt.scatter(best_inter_sol[0], best_inter_sol[1], color='blue', marker='*', s=200, label='Best DC-Inter')
    plt.scatter(best_intra_sol[0], best_intra_sol[1], color='green', marker='*', s=200, label='Best DC-Intra')
    plt.scatter(best_n_sol['values_0'], best_n_sol['values_1'], color='purple', marker='*', s=200, label='Best N')
    plt.xlabel('DC-Inter')
    plt.ylabel('DC-Intra')
    #plt.title('Pareto frontier')
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(ensemble_dir, 'pareto_frontier.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()

    # Extracting the corresponding indexes for the parameters set to 1 for the solution with minimum parameters set to 1
    gans_pareto = df_paretoset.filter(like='params_')
    gans_idx= [gans_pareto.values[i] for i in range(len(gans_pareto))]

    for sel_idx in gans_idx:
        print(sel_idx)
        print(df_paretoset.loc[(df_paretoset.filter(like='params_') == sel_idx).all(axis=1)])
        # Extract corresponding idx.
        idx = df_paretoset.loc[(df_paretoset.filter(like='params_') == sel_idx).all(axis=1)].index[0]

    print('May the force be with you.')

