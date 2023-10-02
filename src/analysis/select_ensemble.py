import sys
sys.path.extend([
    "./",
])

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import json
from itertools import product

from src.general_utils import util_general

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
#sns.set_context("paper")
#sns.set_theme(style="ticks")
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# main
if __name__ == "__main__":

    reports_dir = './reports/'
    dataset_name = 'pneumoniamnist'
    fitness_name = 'fid'
    obj_name = 'intra_inter'
    gan_steps = '20000,40000,60000,80000,100000'
    fitness_summary_flag = 'mean'
    n_trial = '5000'
    eval_backbone_list = ['disc_resnet_50_pneumoniamnist', 'resnet_ae_50_pneumoniamnist', 'SwAV_torch', 'cnn_resnet_50_pneumoniamnist', 'InceptionV3_torch']
    post_resizer = 'friendly'
    n_samples = '50000'
    threshold_list = [50, 75, 90, 95]
    reports_dir = os.path.join(reports_dir, dataset_name, 'ensemble')

    for threshold, eval_backbone in product(threshold_list, eval_backbone_list):

        filename =  f'ensemble_search_{fitness_name}-obj_name_{obj_name}-step_{util_general.parse_separated_list_comma(gan_steps)}-summary_{fitness_summary_flag}-trial_{n_trial}-{eval_backbone}_{post_resizer}_{n_samples}'
        foldernames = [x for x in os.listdir(reports_dir) if filename in x]

        data = {
            'foldername': [],
            'n_best': [],
            'n_gans': [],
            'gans': [],
        }
        for foldername in foldernames:

            # Only dir.
            if not os.path.isdir(os.path.join(reports_dir, foldername)):
                continue

            df = pd.read_excel(os.path.join(os.path.join(reports_dir, foldername, 'optuna_study_best.xlsx')), engine='openpyxl')
            n_best = len(df)


            gan_columns = df.columns[3:-1]
            counts = df[gan_columns].sum()
            sorted_counts = counts.sort_values()

            sorted_counts = sorted_counts[sorted_counts > 0]
            # Delete the GANs that are not in the top 95%.
            filtered_counts = sorted_counts[sorted_counts > np.percentile(sorted_counts.values, threshold)]
            print('\n')
            print(filtered_counts)

            data['foldername'].append(foldername)
            data['n_best'].append(n_best)
            data['n_gans'].append(len(filtered_counts))
            data['gans'].append(list(filtered_counts.index))

        with open(os.path.join(reports_dir, f'{filename}_gan_data_{threshold}.json'), "w") as file:
            json.dump(data, file, indent=4)

        all_gans = [y for x in data['gans'] for y in x]
        common_gans = [y for y in set(all_gans) if all(y in x for x in data['gans'])]
        print(common_gans)
        with open(os.path.join(reports_dir, f'{filename}_common_gan_{threshold}.txt'), "w") as file:
            if common_gans:
                file.write(",".join(common_gans))
            else:
                file.write("No common GANs found in all lists.")

    print('May the force be with you.')