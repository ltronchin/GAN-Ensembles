import copy
import os
import sys

sys.path.extend([
    "./",
])

import pandas as pd
from itertools import combinations
from matplotlib import rc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# For aspect ratio 4:3.
# column_width_pt = 516.0
# pt_to_inch = 1 / 72.27
# column_width_inches = column_width_pt * pt_to_inch
# aspect_ratio = 4 / 3
# sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
sns.set_context("paper")
sns.set_theme(style="ticks")
# # For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters.
reports_dir = f'./reports/'
dataset_name =  'pneumoniamnist'
eval_backbone_list = ['InceptionV3_torch', 'ResNet50_torch', 'SwAV_torch', 'InceptionV3_torch__medical', 'ResNet50_torch__medical']
fitness_name = 'f1_dns_cvg' # 'fid' # 'f1_prc_rec' # 'f1_dns_cvg'
post_resizer = 'friendly'
if fitness_name == 'f1_prc_rec':
    bar_label = 'f1-precision-recall'
elif fitness_name == 'f1_dns_cvg':
    bar_label = 'f1-density-coverage'
elif fitness_name == 'fid':
    bar_label = 'FID'
else:
    raise ValueError(f"Fitness name {fitness_name} not supported.")

n_samples = '4708'
gan_step_list = ['20000', '40000', '60000', '80000', '100000']

# Folders.
reports_dir = os.path.join(reports_dir, dataset_name, 'features')

for eval_backbone in eval_backbone_list:
    filename_intra = f'intra-{eval_backbone}-{post_resizer}-{n_samples}'
    filename_report = f'{fitness_name}-{eval_backbone}-{post_resizer}-{n_samples}'

    # Load dataframe.
    df_intra = pd.read_excel(os.path.join(reports_dir, f'{filename_intra}.xlsx'), engine='openpyxl')

    for gan_step in gan_step_list:
        # Filter df by gan_step.
        df = df_intra[df_intra['step0'] == int(gan_step)]
        df = df[df['step1'] == int(gan_step)]

        # Compute metric_norm and metric_stand.
        mean_metric = df[fitness_name].mean()
        std_metric = df[fitness_name].std()
        df['metric_stand'] = (df[fitness_name] - mean_metric) / std_metric

        min_metric = df[fitness_name].min()
        max_metric = df[fitness_name].max()
        df['metric_norm'] = (df[fitness_name] - min_metric) / (max_metric - min_metric)

        # Drop all columns in df but gan0 gan1 and metric.
        df = df[['gan0', 'gan1', fitness_name, 'metric_norm', 'metric_stand']]

        df_swapped = df.copy()
        df_swapped[['gan0', 'gan1']] = df[['gan1', 'gan0']]
        df_comb = pd.concat([df, df_swapped], ignore_index=True)

        # Create the matrix.
        gan_names = sorted(set(df_comb['gan0'].unique()) | set(df_comb['gan1'].unique()))
        matrix = df_comb.pivot(index='gan0', columns='gan1', values=fitness_name)
        matrix = matrix.reindex(index=gan_names, columns=gan_names)

        fig = plt.figure()
        cmap= sns.color_palette("dark:salmon_r", as_cmap=True)
        # if 'f1_prc_rec' or 'f1_dns_cvg' invert the cmap.
        if fitness_name == 'f1_prc_rec' or fitness_name == 'f1_dns_cvg':
            cmap = cmap.reversed()
        ax = sns.heatmap(matrix, cmap=cmap, cbar=True, fmt=".2f",  cbar_kws={'label': bar_label},  mask=(matrix == np.inf))
        ax.set(xlabel="", ylabel="")
        plt.tight_layout()
        fig.savefig(os.path.join(reports_dir, f'heatmap-gan_step_{gan_step}-{filename_report}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
        plt.show()

print("May the force be with you.")