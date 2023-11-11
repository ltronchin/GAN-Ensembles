import sys
import os
sys.path.extend([
    "./",
])

import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np
from itertools import product
import cv2
import re
import torch
def count_ensemble(file_path):
    import re
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            gan_with_steps = re.findall(r"[\w-]+(?=__[0-9]+)", content)
            return len(gan_with_steps)
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {e}"

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

# Parameters
dataset_name =  'pneumoniamnist' # 'AIforCOVID' 'pneumoniamnist' 'retinamnist' 'breastmnist'
foldername = 'downstream_task_competitors'
reports_dir = os.path.join('./reports', dataset_name, foldername)
gan_models = ['MHGAN','SNGAN','StyleGAN2-D2DCE','ReACGAN-ADA','ReACGAN-ADC','ReACGAN-DiffAug','ACGAN-Mod','ReACGAN','BigGAN-DiffAug','BigGAN-Info','StyleGAN2-DiffAug','ACGAN-Mod-TAC','BigGAN','ReACGAN-TAC','BigGAN-ADA','StyleGAN2-Info','ACGAN-Mod-ADC','StyleGAN2-ADA','ReACGAN-Info','StyleGAN2','ContraGAN','SAGAN']
gan_steps = ['20000', '40000', '60000', '80000', '100000']
metric_name_list = ['ACC', 'recall', 'precision', 'specificity', 'f1_score', 'g_mean']

# Results dictionary.
data = {f'mean_{metric_name}': [] for metric_name in metric_name_list}
data.update({f'std_{metric_name}': [] for metric_name in metric_name_list})
data.update({'exp_name': []})
data.update({'models': []})
data.update({'steps': []})
data.update({'name_obj': []})
data.update({'n': []})

folders = os.listdir(reports_dir)

for folder in folders:

    # Skip if not folder.
    if not os.path.isdir(os.path.join(reports_dir, folder)):
        continue
    # Read the results.xlsx file
    try:
        results_path = os.path.join(reports_dir, folder, "results.xlsx")
        results_df = pd.read_excel(results_path)
    except FileNotFoundError:
        print(f'FileNotFoundError: {folder}')
        continue

    # Capture the filenames.
    if folder == 'real':
        exp_name = 'Real'
        models = 'NA'
        steps = 'NA'
        n = 1
    elif 'random' in folder:
          n = folder.split('__')[-1]
          exp_name = f'Random {n}'
          models = 'NA'
          steps = 'NA'
    else:
        p = folder.split('--')
        if p[0] == 'single_gan':
            exp_name = 'Mean GAN'
            n = 1
        elif p[0] == 'naive_models':
            exp_name = 'Naive models'
            n = 22
        elif p[0] == 'naive_steps':
            exp_name = 'Naive steps'
            n = 5
        elif p[0] == 'naive_models_steps':
            exp_name = 'Naive'
            n = 110
        else:
            raise ValueError(f'p[0]={p[0]} not supported')
        models = p[1].split('__')[-1]
        steps = p[2].split('__')[-1]
    name_obj = 'NA'

    # Extract metric values.
    for metric_name in metric_name_list:

        values = results_df[metric_name].values[:-2]
        mean_values = np.mean(values)
        std_values = np.std(values)

        data[f'mean_{metric_name}'].append(mean_values)
        data[f'std_{metric_name}'].append(std_values)

    data['exp_name'].append(exp_name)
    data['models'].append(models)
    data['steps'].append(steps)
    data['name_obj'].append(name_obj)
    data['n'].append(n)

report_dir_ensemble = os.path.join('./reports', dataset_name, 'downstream_task_ensemble')
folders = os.listdir(report_dir_ensemble)
# filter out all the alements that has dim_reduction == False
folders = [folder for folder in folders if 'dim_reduction__True' not in folder]

for folder in folders:
    # Skip if not folder.
    if not os.path.isdir(os.path.join(report_dir_ensemble, folder)):
        continue

    # Read the results.xlsx file
    try:
        results_path = os.path.join(report_dir_ensemble, folder, "results.xlsx")
        results_df = pd.read_excel(results_path)
    except FileNotFoundError:
        print(f'FileNotFoundError: {folder}')
        continue

    # Red ensemble.txt file to load the number of GANs.
    n = count_ensemble(os.path.join(report_dir_ensemble, folder, "ensemble.txt"))

    # Capture the filenames.
    p = folder.split('-')
    name_obj = p[2].split('__')
    if  len(name_obj) == 2:
        name_obj = name_obj[-1]
    elif len(name_obj) == 3:
        name_obj = name_obj[1] + '__' +  name_obj[2]
    else:
        raise ValueError(f'len(name_obj)={len(name_obj)} not supported')
    eval_backbone = p[-3]
    if eval_backbone == 'InceptionV3_torch':
        exp_name = 'InceptionV3'
    elif eval_backbone == f'ResNet50_torch':
        exp_name = 'ResNet50'
    elif eval_backbone == 'SwAV_torch':
        exp_name = 'SwAV'
    elif eval_backbone == f'InceptionV3_torch__medical':
        exp_name = 'InceptionV3-Med'
    elif eval_backbone == f'ResNet50_torch__medical':
        exp_name = 'ResNet50-Med'
    else:
        raise ValueError(f'eval_backbone={eval_backbone} not supported')
    models = 'ensemble'
    steps = 'ensemble'

    for metric_name in metric_name_list:

        values = results_df[metric_name].values[:-2]
        mean_values = np.mean(values)
        std_values = np.std(values)

        data[f'mean_{metric_name}'].append(mean_values)
        data[f'std_{metric_name}'].append(std_values)

    data['exp_name'].append(exp_name)
    data['models'].append(models)
    data['steps'].append(steps)
    data['name_obj'].append(name_obj)
    data['n'].append(n)

# Flatten all the list inside data.
df = pd.DataFrame(data=data)
df.to_excel(os.path.join(reports_dir, 'history_performances.xlsx'), index=False)

# ------------------------------------------------------------------------------------
# Random analysis.
# ------------------------------------------------------------------------------------
backbone = 'SwAV' # 'SwAV'
rand_exps = df[df['exp_name'].str.contains('Random') | (df['exp_name'] == 'Naive')]

rand_exps = rand_exps.assign(exp_nums=rand_exps['exp_name'].replace({'Random ': '', 'Naive': '110'}, regex=True).astype(int))
rand_exps = rand_exps.sort_values(by='exp_nums')
dge_exp = df[(df['exp_name'] == backbone) & (df['name_obj'] == 'dc_inter__dc_intra')]

plt.figure()
sns.lineplot(data=rand_exps[rand_exps['exp_nums'] < 110], x='exp_nums', y='mean_g_mean', marker='o', color='gray', label='Naive (R)')
sns.scatterplot(data=rand_exps[rand_exps['exp_nums'] == 110], x='exp_nums', y='mean_g_mean', color='blue', s=150, marker='*', label='Naive (A)')
sns.scatterplot(data=dge_exp, x='n', y='mean_g_mean', color='red', s=150, marker='*', label='DGE')

plt.xlabel('Number of GANs')
plt.ylabel('g-mean')
plt.xticks(rand_exps['exp_nums'].unique(), rotation=45)
#plt.xticks(list(plt.xticks()[0]) + [dge_exp['n'].values[0]])
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./reports/history_performances_random_{dataset_name}_{backbone}.pdf', bbox_inches='tight', dpi=300)
plt.show()

# ------------------------------------------------------------------------------------
# Backbones.
# ------------------------------------------------------------------------------------
backbone_exps = df[df['name_obj'] == 'dc_inter__dc_intra']
backbone_exps = backbone_exps[backbone_exps['exp_name'].isin(['InceptionV3', 'ResNet50', 'SwAV', 'InceptionV3-Med', 'ResNet50-Med'])]

# Standardise.
mu_perf = backbone_exps['mean_g_mean'].mean()
sigma_perf = backbone_exps['mean_g_mean'].std()
backbone_exps['z_score'] = (backbone_exps['mean_g_mean'] - mu_perf) / sigma_perf

# Referral z-score value.
dge_z_score = backbone_exps.loc[backbone_exps['exp_name'] == backbone, 'z_score'].values[0]
backbone_exps['z_score_diff'] = backbone_exps['z_score'] - dge_z_score
backbone_exps = backbone_exps.sort_values('z_score_diff')

# Vertical barplot.
backbone_exps['exp_name'].replace({backbone: f'{backbone} (DGE)'}, inplace=True)
backbone_color_mapping = {
    'InceptionV3': '#4285F4',
    'ResNet50': '#DB4437',
    'SwAV (DGE)': '#E7E6E6',  # DGE
    'InceptionV3-Med': '#0F9D58',
    'ResNet50-Med': '#F4B400'
}

backbone_exps['color'] = backbone_exps['exp_name'].map(backbone_color_mapping)

plt.figure()
bar = sns.barplot(x='z_score_diff', y='exp_name', data=backbone_exps, palette=backbone_exps['color'], orient='h')
bar.set(ylabel=None)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Z-Score Difference from DGE')
#plt.ylabel('Backbones')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./reports/z_score_backbones_{dataset_name}_{backbone}.pdf', bbox_inches='tight', dpi=300)
plt.show()

# ------------------------------------------------------------------------------------
# Metrics.
# ------------------------------------------------------------------------------------
metric_exps = df[df['exp_name'] == backbone]
metric_exps = metric_exps[metric_exps['name_obj'].isin(['dc_inter__dc_intra', 'dc_inter', 'fid_inter', 'fid_inter__fid_intra'])]

# Standardise.
mu_perf = metric_exps['mean_g_mean'].mean()
std_perf = metric_exps['mean_g_mean'].std()
metric_exps['z_score'] = (metric_exps['mean_g_mean'] - mu_perf) / std_perf

# Referral z-score value.
dge_metric_z_score = metric_exps.loc[metric_exps['name_obj'] == 'dc_inter__dc_intra', 'z_score'].values[0]
metric_exps['z_score_diff'] = metric_exps['z_score'] - dge_metric_z_score
metric_exps = metric_exps.sort_values('z_score_diff')

# Vertical barplot.
metric_exps['name_obj'].replace({'dc_inter': 'DC-Inter'}, inplace=True)
metric_exps['name_obj'].replace({'fid_inter': 'FID-Inter'}, inplace=True)
metric_exps['name_obj'].replace({'dc_inter__dc_intra': 'DC-Inter_Intra (DGE)'}, inplace=True)
metric_exps['name_obj'].replace({'fid_inter__fid_intra': 'FID-Inter_Intra'}, inplace=True)

metric_color_mapping = {
    'FID-Inter': '#4285F4',
    'FID-Inter_Intra': '#DB4437',
    'DC-Inter_Intra (DGE)': '#E7E6E6',  # DGE
    'DC-Inter': '#0F9D58',
}
metric_exps['color'] = metric_exps['name_obj'].map(metric_color_mapping)

plt.figure()
# barplot (do not show ylabel
bar = sns.barplot(x='z_score_diff', y='name_obj', data=metric_exps, palette=metric_color_mapping, orient='h')
bar.set(ylabel=None)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Z-Score Difference from DGE')
#plt.ylabel('Metric')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./reports/z_score_metrics_{dataset_name}_{backbone}.pdf', bbox_inches='tight', dpi=300)
plt.show()

print("May the force be with you.")