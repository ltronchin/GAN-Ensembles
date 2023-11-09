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
from scipy import stats

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
#sns.set_context("paper")
#sns.set_theme(style="ticks")
# # For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters.
reports_dir = './reports/'
dataset_name =  'pneumoniamnist'
analysis_name = 'intra'
split = 'train'
fitness_name = 'fid'
eval_backbones = ['InceptionV3_torch', 'ResNet50_torch', 'SwAV_torch', 'InceptionV3_torch__medical', 'InceptionV3_torch__truefake', 'ResNet50_torch__medical', 'ResNet50_torch__truefake']
post_resizer = 'friendly'

reports_dir = os.path.join(reports_dir, dataset_name, 'features')

for eval_backbone in eval_backbones:
    print('eval_backbone:', eval_backbone)
    if analysis_name == 'inter':
        filename = f'inter-{split}-{eval_backbone}-{post_resizer}'
    elif analysis_name == 'intra':
        filename = f'intra_{eval_backbone}_{post_resizer}'
    else:
        raise ValueError(f"Analysis name {analysis_name} not supported.")

    # Load dataframe.
    df = pd.read_excel(os.path.join(reports_dir, f'{filename}.xlsx'), engine='openpyxl')

    if analysis_name == 'inter':
        df = df.sort_values(by=[fitness_name])
        print(df[['gan0', 'gan1', 'step0', 'step1']].head(2))
    elif analysis_name == 'intra':
        df = df.sort_values(by=[fitness_name], ascending=False)
        print(df[['gan0', 'gan1', 'step0', 'step1']].head(1))
    else:
        raise ValueError(f"Analysis name {analysis_name} not supported.")
    print('\n')

print("May the force be with you.")