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
dataset_name = 'pneumoniamnist'
foldername = 'downstream_task_randomness_warmup_weighted' #'downstream_task_randomness_warmup'
reports_dir = f'./reports/{dataset_name}/{foldername}/'
filename = f'{dataset_name}_overall_reports_all.xlsx'
eval_backbone = f'cnn_resnet_50_{dataset_name}' # InceptionV3_torch, ResNet50_torch, SwAV_torch, resnet_ae_50_pneumoniamnist, disc_resnet_50_pneumoniamnist, cnn_resnet_50_pneumoniamnist

step_to_plot_list = ['100000'] # ['20000', '40000', '60000', '80000', '100000']
metric_name_list = ['ACC', 'f1_score', 'auc', 'geometric_mean'] # 'f1_score' #'ACC'
plot_mean_gan_list =  [False] # [True, False]

folders = os.listdir(reports_dir)
pattern = re.compile(
    r'(?P<dataset>\w+)-'
    r'(?P<gan>.+?)-'
    r'(?P<step>[\d,]+)-'
    r'(?P<fitness_name>\w+)-'
    r'(?P<cost_name>\w+)-'
    r'(?P<eval_backbone>[\w_]+)'
)

for step_to_plot, metric_name, plot_mean_gan in product(step_to_plot_list, metric_name_list, plot_mean_gan_list):

    naive = []
    real = []
    dge_cnn = []
    dge_ae = []
    dge_swav = []
    dge_inceptionv3 = []
    mean_gan = []

    data = {
        metric_name: [],
        'exp': [],
        'fitness_name': [],
        'cost_name': [],
        'eval_backbone': []
    }

    for folder in folders:

        try:
            # Read the results.xlsx file
            results_path = os.path.join(reports_dir, folder, "results.xlsx")
            results_df = pd.read_excel(results_path)

            # Extract metric values
            if 'real' in folder:
                print(folder)
                real.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('Real', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat('Real', len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat('Real', len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat('Real', len(results_df[metric_name].values[:-2])))
                continue

            match = re.search(pattern, folder)
            gan = match.group("gan")
            step = match.group("step")
            fitness_name = match.group("fitness_name")
            cost_name = match.group("cost_name")
            eval_backbone = match.group("eval_backbone")

            if step != step_to_plot:
                continue
            if eval_backbone == 'disc_resnet_50_pneumoniamnist_friendly':
                continue

            print('\n')
            print(f"gan={gan}")
            print(f"step={step}")
            print(f"fitness_name={fitness_name}")
            print(f"cost_name={cost_name}")
            print(f"eval_backbone={eval_backbone}")


            if 'all' in folder:
                naive.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('Naive', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='resnet_ae_50_pneumoniamnist_friendly'  and cost_name == 'ratio':
                dge_ae.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Ratio DDTI', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='SwAV_torch_friendly'  and cost_name == 'ratio':
                dge_swav.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Ratio DITI', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='InceptionV3_torch_friendly'  and cost_name == 'ratio':
                dge_inceptionv3.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Ratio DITD', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='cnn_resnet_50_pneumoniamnist_friendly'  and cost_name == 'ratio':
                dge_cnn.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Ratio DDTD', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='resnet_ae_50_pneumoniamnist_friendly' and cost_name == 'intra':
                dge_ae.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Intra DDTI', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='SwAV_torch_friendly' and cost_name == 'intra':
                dge_swav.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Intra DITI', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='InceptionV3_torch_friendly' and cost_name == 'intra':
                dge_inceptionv3.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Intra DITD', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone=='cnn_resnet_50_pneumoniamnist_friendly' and cost_name == 'intra':
                dge_cnn.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Intra DDTD', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone == 'resnet_ae_50_pneumoniamnist_friendly' and cost_name == 'inter':
                dge_ae.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Inter DDTI', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone == 'SwAV_torch_friendly' and cost_name == 'inter':
                dge_swav.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Inter DITI', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone == 'InceptionV3_torch_friendly' and cost_name == 'inter':
                dge_inceptionv3.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Inter DITD', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            elif fitness_name == 'fid' and eval_backbone == 'cnn_resnet_50_pneumoniamnist_friendly' and cost_name == 'inter':
                dge_cnn.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('DGE-Inter DDTD', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

            else:
                mean_gan.append(results_df[metric_name].values[:-2])
                data[metric_name].append(results_df[metric_name].values[:-2])
                data['exp'].append(np.repeat('Mean GAN', len(results_df[metric_name].values[:-2])))
                data['fitness_name'].append(np.repeat(fitness_name, len(results_df[metric_name].values[:-2])))
                data['cost_name'].append(np.repeat(cost_name, len(results_df[metric_name].values[:-2])))
                data['eval_backbone'].append(np.repeat(eval_backbone, len(results_df[metric_name].values[:-2])))

        except Exception as e:
            print('\n')
            print(e)
            print("ERROR in folder:")
            print(folder)

            continue

    # Flatten all the list inside data.
    data = {k: np.concatenate(data[k]) for k in list(data.keys())}
    df = pd.DataFrame(data=data)
    median_val_gan_mean = np.median(df[df.exp == 'Mean GAN'][metric_name])
    #df.drop(df[df[metric_name] > 0.85].index & df[df.exp == 'Mean GAN'].index, inplace=True)
    if not plot_mean_gan:
        df.drop(df[df.exp == 'Mean GAN'].index, inplace=True)
    #df.drop(df[df.eval_backbone == 'SwAV_torch_friendly'].index, inplace=True)
    #df.drop(df[df.eval_backbone == 'cnn_resnet_50_pneumoniamnist_friendly'].index, inplace=True)
    #df.drop(df[df.eval_backbone == 'resnet_ae_50_pneumoniamnist_friendly'].index, inplace=True)
    #df.drop(df[df.eval_backbone == 'InceptionV3_torch_friendly'].index, inplace=True)
    medians = df.groupby(['exp'])[metric_name].median().reset_index()
    sorted_exp = medians.sort_values(metric_name, ascending=True)['exp'].values

    # Plot
    fig = plt.figure()
    sns.boxplot(data=df, x="exp", y=metric_name, order=sorted_exp)
    plt.axhline(median_val_gan_mean,color='r', linestyle='--', label='Mean GAN')
    plt.tight_layout()  # To adjust layout to accommodate the legend
    plt.xticks(rotation=90)
    if metric_name == 'ACC':
        plt.ylabel('Accuracy')
    elif metric_name == 'f1_score':
        plt.ylabel('F1 score')
    elif metric_name == 'precision':
        plt.ylabel('Precision')
    elif metric_name == 'recall':
        plt.ylabel('Recall')
    elif metric_name == 'auc':
        plt.ylabel('AUC')
    elif metric_name == 'geometric_mean':
        plt.ylabel('Geometric mean')
    plt.xlabel('Experiments')
    fig.savefig(os.path.join(reports_dir, f'plot__folder_{foldername}__metric_{metric_name}__ganstep_{step_to_plot}__addmeangan_{plot_mean_gan}.pdf'), dpi=400, format='pdf', bbox_inches='tight')
    plt.show()

print("May the force be with you.")