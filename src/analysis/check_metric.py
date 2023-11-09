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

def plot_scatter(bck1, bck2, X):
    plt.scatter(X[bck1], X[bck2], alpha=0.6)
    plt.xlabel(bck1)
    plt.ylabel(bck2)
    plt.grid(True)

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
reports_dir = './reports/'
dataset_name =  'pneumoniamnist'
n_samples = '4708'

fitness_name = 'fid'
eval_backbones = ['InceptionV3_torch', 'ResNet50_torch', 'SwAV_torch', 'InceptionV3_torch__medical', 'InceptionV3_torch__truefake', 'ResNet50_torch__medical', 'ResNet50_torch__truefake']
post_resizer = 'friendly'

reports_dir = os.path.join(reports_dir, dataset_name, 'features')

for eval_backbone in eval_backbones:
    filename = f'intra_test-{eval_backbone}-{post_resizer}-{n_samples}'

    # Load dataframe.
    data = pd.read_excel(os.path.join(reports_dir, f'{filename}.xlsx'), engine='openpyxl')

    fig = plt.figure()
    sns.lineplot(data=data, x='step0', y=fitness_name)
    plt.xlabel('Step')
    plt.ylabel(fitness_name)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, f'{fitness_name}_test_step_{eval_backbone}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    sns.lineplot(data= data.reset_index(), x='index', y=data[fitness_name], lw=2, color='blue')
    plt.ylabel(fitness_name)
    plt.xlabel('')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, f'{fitness_name}_test_{eval_backbone}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()


print("May the force be with you.")