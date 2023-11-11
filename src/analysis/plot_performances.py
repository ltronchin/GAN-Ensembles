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
# column_width_pt = 516.0
# pt_to_inch = 1 / 72.27
# column_width_inches = column_width_pt * pt_to_inch
# aspect_ratio = 4 / 3
# sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
sns.set_context("paper")
sns.set_theme(style="ticks")
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters
dataset_name = 'AIforCOVID' #'retinamnist' 'pneumoniamnist' 'breastmnist'
foldername = 'downstream_task_competitors'
reports_dir = os.path.join('./reports', dataset_name, foldername)
gan_models = ['MHGAN','SNGAN','StyleGAN2-D2DCE','ReACGAN-ADA','ReACGAN-ADC','ReACGAN-DiffAug','ACGAN-Mod','ReACGAN','BigGAN-DiffAug','BigGAN-Info','StyleGAN2-DiffAug','ACGAN-Mod-TAC','BigGAN','ReACGAN-TAC','BigGAN-ADA','StyleGAN2-Info','ACGAN-Mod-ADC','StyleGAN2-ADA','ReACGAN-Info','StyleGAN2','ContraGAN','SAGAN']
gan_steps = ['20000', '40000', '60000', '80000', '100000']
metric_name = 'g_mean'

# Results dictionary.
data = {
    metric_name: [],
    'exp_name': [],
    'models': [],
    'steps': [],
    'name_obj': [],
    'n': []
}
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
    values = results_df[metric_name].values[:-2]
    data[metric_name].append(values)
    data['exp_name'].append(np.repeat(exp_name, len(values)))
    data['models'].append(np.repeat(models, len(values)))
    data['steps'].append(np.repeat(steps, len(values)))
    data['name_obj'].append(np.repeat(name_obj, len(values)))
    data['n'].append(np.repeat(n, len(values)))

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

    values = results_df[metric_name].values[:-2]
    data[metric_name].append(values)
    data['exp_name'].append(np.repeat(exp_name, len(values)))
    data['models'].append(np.repeat(models, len(values)))
    data['steps'].append(np.repeat(steps, len(values)))
    data['name_obj'].append(np.repeat(name_obj, len(values)))
    data['n'].append(np.repeat( n, len(values)))

# Flatten all the list inside data.
data = {k: np.concatenate(data[k]) for k in list(data.keys())}
df = pd.DataFrame(data=data)

# Drop some the rows.
df = df[~df.exp_name.str.contains('Random')]
# Drop naive models and steps.
df = df[~df.exp_name.str.contains('Naive steps')]
df = df[~df.exp_name.str.contains('Naive models')]

# Turn the columns n to int.
df['n'] = df['n'].astype(int)

for name_obj in ['dc_inter', 'fid_inter', 'dc_inter__dc_intra', 'fid_inter__fid_intra']:

    df_obj = df[(df.name_obj == name_obj) | (df.name_obj == 'NA')]

    mean_gan = np.mean(df_obj[df_obj.exp_name == 'Mean GAN'][metric_name])
    mean_val = df_obj.groupby(['exp_name'])[metric_name].median().reset_index()
    df_n = df_obj.groupby(['exp_name'])['n'].mean().reset_index()
    sorted_exp = mean_val.sort_values(metric_name, ascending=True)['exp_name'].values

    # Order df_n according sorted_exp.
    mean_n = []
    for exp_name in sorted_exp:
        mean_n.append(df_n[df_n.exp_name == exp_name]['n'].values[0])

    fig, ax1 = plt.subplots()

    # Boxplot for the metric
    sns.boxplot(data=df_obj, x="exp_name", y=metric_name, order=sorted_exp, showfliers=False, ax=ax1)
    ax1.axhline(mean_gan, color='r', linestyle='--', label=f'Mean {metric_name}')
    ax1.set_ylabel(metric_name)
    ax1.set_xlabel('Experiments')
    ax1.tick_params(axis='x', rotation=45)

    # Secondary axis for the number of GANs
    ax2 = ax1.twinx()
    ax2.plot(sorted_exp, mean_n, color='b', linestyle='--', label='Number of GANs')
    ax2.set_ylabel('Number of GANs')

    # Labels & Legend
    fig.tight_layout()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2)

    # Save the plot
    plot_filename = f'plot_folder-{foldername}-metric__{metric_name}-name_obj__{name_obj}.pdf'
    plot_path = os.path.join(reports_dir, plot_filename)
    fig.savefig(plot_path, dpi=400, format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

print("May the force be with you.")