import sys
sys.path.extend([
    "./",
])

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.0, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
#sns.set_context("paper")
#sns.set_theme(style="ticks")
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# main
if __name__ == "__main__":

    reports_dir = './reports/'
    dataset_name = 'pneumoniamnist'
    foldername =   '2023-09-26_08-08-09_ensemble_search_fid-obj_name_intra_inter-step_20000,40000,60000,80000,100000-summary_mean-trial_5000-cnn_resnet_50_pneumoniamnist_friendly_50000'# '2023-09-26_08-07-31_ensemble_search_fid-obj_name_intra_inter-step_20000,40000,60000,80000,100000-summary_mean-trial_5000-resnet_ae_50_pneumoniamnist_friendly_50000' # ''
    reports_dir = os.path.join(reports_dir, dataset_name, 'ensemble', foldername)

    # Load dataframe.
    df = pd.read_excel(os.path.join(os.path.join(reports_dir, 'optuna_study_best.xlsx')), engine='openpyxl')
    gan_columns = df.columns[3:-1]
    counts = df[gan_columns].sum()
    sorted_counts = counts.sort_values()
    filtered_counts = sorted_counts[sorted_counts > 0]

    # Plot.
    step_colors = {
        "20000": "blue",
        "40000": "green",
        "60000": "yellow",
        "80000": "orange",
        "100000": "red"
    }
    colors = [step_colors[label.split("_")[-1]] for label in filtered_counts.index]
    simplified_labels = ["_".join(label.split("_")[:-1]).replace("params_", "") for label in filtered_counts.index]

    # Plot the histogram with colors representing the step
    fig = plt.figure(figsize=(10, 15))
    bars = plt.barh(range(len(filtered_counts)), filtered_counts, color=colors)

    # Create legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=step_colors[step]) for step in step_colors]
    plt.legend(handles, step_colors.keys(), title="Steps")

    plt.ylabel("GANs")
    plt.yticks(range(len(simplified_labels)), simplified_labels)
    plt.tight_layout()  # To adjust layout to accommodate the legend
    fig.savefig(os.path.join(reports_dir, f'horizontal_hist.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()

    print('May the force be with you.')