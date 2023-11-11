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

def plot_scatter(bck1, bck2, data, metric_name):
    plt.scatter(data[data['backbone']==bck1][metric_name], data[data['backbone']==bck2][metric_name], alpha=0.6)
    plt.xlabel(bck1)
    plt.ylabel(bck2)
    plt.grid(True)

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
# sns.set_context("paper")
# sns.set_theme(style="ticks")
# # For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters.
reports_dir = './reports/'
dataset_name =  'pneumoniamnist' #'breastmnist' 'retinamnist' 'pneumoniamnist' 'AIforCOVID'

fitness_name = 'f1_dns_cvg' # 'fid' # 'f1_prc_rec' # 'f1_dns_cvg'
eval_backbones = ['InceptionV3_torch', 'ResNet50_torch', 'SwAV_torch', 'InceptionV3_torch__medical', 'ResNet50_torch__medical']
post_resizer = 'friendly'
analysis_name = 'intra' # 'inter' # 'intra'
split = 'train'

n_samples = '4708' # '546' '1080' '4708' '664'

reports_dir = os.path.join(reports_dir, dataset_name, 'features')

df = pd.DataFrame(columns=['backbone', 'metric', 'metric_norm', 'metric_stand'])

for eval_backbone in eval_backbones:
    if analysis_name == 'inter':
        filename = f'inter-{split}-{eval_backbone}-{post_resizer}-{n_samples}'
    elif analysis_name == 'intra':
        filename = f'intra-{eval_backbone}-{post_resizer}-{n_samples}'
    else:
        raise ValueError(f"Analysis name {analysis_name} not supported.")

    # Load dataframe.
    df_bkb = pd.read_excel(os.path.join(reports_dir, f'{filename}.xlsx'), engine='openpyxl')
    df_bkb['metric'] = copy.deepcopy(df_bkb[fitness_name])

    # Standardize the values of metric computing the mean and std.
    mean_fid = df_bkb[fitness_name].mean()
    std_fid = df_bkb[fitness_name].std()
    df_bkb['metric_stand'] = (df_bkb[fitness_name] - mean_fid) / std_fid

    # Normalize the values of the metric computing the min and max.
    min_fid = df_bkb[fitness_name].min()
    max_fid = df_bkb[fitness_name].max()
    df_bkb['metric_norm'] = (df_bkb[fitness_name] - min_fid) / (max_fid - min_fid)

    # Append the total dataframe repeating the value of the backbone.
    df_bkb['backbone'] = eval_backbone

    # Drop everything except the gan0, gan1, metric, metric_norm, backbone columns.
    df_bkb = df_bkb[['backbone', 'metric', 'metric_norm', 'metric_stand']]
    df = df.append(df_bkb, ignore_index=True)

# Save to excel.
if analysis_name == 'inter':
    df.to_excel(os.path.join(reports_dir, f'inter_{fitness_name}-{split}_{post_resizer}_{n_samples}.xlsx'), index=False)
elif analysis_name == 'intra':
    df.to_excel(os.path.join(reports_dir, f'intra_{fitness_name}-{split}_{post_resizer}_{n_samples}.xlsx'), index=False)
else:
    raise ValueError(f"Analysis name {analysis_name} not supported.")

# Define the mapping from old labels to new labels
eval_backbones_label = ['InceptionV3', 'ResNet50', 'SwAV', 'InceptionV3-Med', 'ResNet50-Med']
label_mapping = {
    'InceptionV3_torch': 'InceptionV3',
    'InceptionV3_torch__medical': 'InceptionV3-Med',
    'ResNet50_torch': 'ResNet50',
    'ResNet50_torch__medical': 'ResNet50-Med',
    'SwAV_torch': 'SwAV',
}
df['backbone'] = df['backbone'].map(label_mapping)

#backbones_list = df['backbone'].unique()
#corr = pd.DataFrame(index=backbones_list, columns=backbones_list)
corr = pd.DataFrame(index=eval_backbones_label, columns=eval_backbones_label)
corr_to_plot = 'metric_norm'
for backbone1 in eval_backbones_label:
    for backbone2 in eval_backbones_label:
        #print(len(df[df['backbone'] == backbone1]))
        #print(len(df[df['backbone'] == backbone2]))
        fid_1 =list(df[df['backbone'] == backbone1][corr_to_plot])
        fid_2 = list(df[df['backbone'] == backbone2][corr_to_plot])

        c = stats.spearmanr(fid_1, fid_2)
        corr.at[backbone1, backbone2] = c.statistic

# Convert corr to float64
corr = corr.astype(np.float64)
# Modify the name SwAV to SwAV (DGE) on both axes.
corr = corr.rename(columns={'SwAV': 'SwAV (DGE)'})
corr = corr.rename(index={'SwAV': 'SwAV (DGE)'})
# Plotting the heatmap
fig = plt.figure()
#mask = np.triu(np.ones_like(corr, dtype=bool))
#sns.heatmap(corr,annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
sns.heatmap(corr,annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.xlabel('')
plt.ylabel('')
# Save
plt.tight_layout()
fig.savefig(os.path.join(reports_dir, f'heatmap_corr_{analysis_name}-{fitness_name}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
plt.show()

# Plotting scatter plots for each bacbkone against the others.
for eval_backbone_label in eval_backbones_label:

    fig = plt.figure(figsize=(15, 15))
    # select all the remains backbones in lab_new and remove the current backbone.
    temp = copy.deepcopy(eval_backbones_label)
    temp.remove(eval_backbone_label)

    for i, lab in enumerate(temp):
        plt.subplot(3, 2, i+1)
        plot_scatter(eval_backbone_label, lab, df, metric_name='metric')
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, f'scatter_corr_{analysis_name}-{fitness_name}-{eval_backbone_label}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()

print("May the force be with you.")