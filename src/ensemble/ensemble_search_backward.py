import copy
import sys
sys.path.extend([
    "./",
])
import argparse
import os
from itertools import combinations
import pandas as pd
import pickle
from tqdm import tqdm
import torch

from src.general_utils import util_general
from src.general_utils import util_path
from src.general_utils import util_ensemble

import optuna
def get_parser():

    parser = argparse.ArgumentParser(description='Ensemble search.')

    parser.add_argument('--source_dir', type=str, default='./reports/pneumoniamnist', help='Directory name to fake samples.')
    parser.add_argument('--dataset_name', type=str, default='pneumoniamnist',  choices=['chestmnist', 'pneumoniamnist', 'breastmnist'], help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # GANs params.
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list,   default='MHGAN,SNGAN,StyleGAN2-D2DCE,ReACGAN-ADA,ReACGAN-ADC,ReACGAN-DiffAug,ACGAN-Mod,ReACGAN,BigGAN-DiffAug,BigGAN-Info,StyleGAN2-DiffAug,ACGAN-Mod-TAC,BigGAN,ReACGAN-TAC,BigGAN-ADA,StyleGAN2-Info,ACGAN-Mod-ADC,StyleGAN2-ADA,ReACGAN-Info,StyleGAN2,ContraGAN,SAGAN', help='List of GANs to enable in the ensemble') # MHGAN,SNGAN,StyleGAN2-D2DCE,ReACGAN-ADA,ReACGAN-ADC,ReACGAN-DiffAug,ACGAN-Mod,ReACGAN,BigGAN-DiffAug,BigGAN-Info,StyleGAN2-DiffAug,ACGAN-Mod-TAC,BigGAN,ReACGAN-TAC,BigGAN-ADA,StyleGAN2-Info,ACGAN-Mod-ADC,StyleGAN2-ADA,ReACGAN-Info,StyleGAN2,ContraGAN,SAGAN
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='20000,40000,60000,80000,100000',help='Iter or Iters to sample each GAN') # 20000,40000,60000,80000,100000
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='InceptionV3_torch', help="[InceptionV3_torch, ResNet50_torch, SwAV_torch, resnet_ae_50_pneumoniamnist, disc_resnet_50_pneumoniamnist]")
    parser.add_argument('--n_samples', type=int, default='4708', help='Total number of images generated by the ensemble.')

    parser.add_argument('--obj_name', type=str, default='intra_inter', choices=['intra_inter', 'intra', 'inter'])
    parser.add_argument('--fitness_summary_flag', type=str, default='mean', choices=['mean'])
    parser.add_argument('--fitness_name', type=str, default='fid', choices=['fid', 'rec'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])


    return parser

if __name__ == '__main__':

    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # Directories.
    source_dir = args.source_dir
    dataset_name = args.dataset_name
    reports_dir = os.path.join(source_dir, 'ensemble')
    features_dir = os.path.join(source_dir, 'features')
    util_path.create_dir(reports_dir)

    # Parameters.
    gan_models = args.gan_models
    gan_steps = args.gan_steps
    eval_backbone= args.eval_backbone
    post_resizer= args.post_resizer
    n_samples = args.n_samples
    obj_name = args.obj_name
    fitness_name = args.fitness_name
    fitness_summary_flag = args.fitness_summary_flag
    split = args.split

    filename = f'ensemble_search_backward_{fitness_name}_{obj_name}-step_{util_general.parse_separated_list_comma(gan_steps)}-summary_{fitness_summary_flag}-{eval_backbone}_{post_resizer}_{n_samples}'
    import datetime
    now = datetime.datetime.now()
    filename = f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_{filename}'

    ensemble_dir = os.path.join(reports_dir, filename)
    util_path.create_dir(ensemble_dir)

    # Load the dataframe containing the binary FID.
    try:
        if fitness_name in ['rec', 'prc', 'dns', 'cvg']:
            df_synth = pd.read_excel(os.path.join(features_dir, f'history_prdc_bin_combinations_{eval_backbone}_{post_resizer}_{n_samples}.xlsx'))
            df_real = pd.read_excel(os.path.join(features_dir, f'history_prdc_bin_real_{split}_{eval_backbone}_{post_resizer}_{n_samples}.xlsx'))
        else:
            df_synth = pd.read_excel(os.path.join(features_dir, f'history_{fitness_name}_bin_combinations_{eval_backbone}_{post_resizer}_{n_samples}.xlsx'))
            df_real = pd.read_excel(os.path.join(features_dir, f'history_{fitness_name}_bin_real_{split}_{eval_backbone}_{post_resizer}_{n_samples}.xlsx'))
    except FileNotFoundError:
        raise FileNotFoundError

    # Backward Search
    search_space = set([f"{model}_{step}" for model in gan_models for step in gan_steps])
    df = pd.DataFrame(columns=['step', 'step_comb', 'ensemble', 'fitness', 'n_gans'])

    gan_ensemble_temp = copy.deepcopy(search_space)
    gan_ensemble_optimal = copy.deepcopy(gan_ensemble_temp)
    obj_gan_ensemble_temp = util_ensemble.backward_objective_pairwise(
        sel_gans=gan_ensemble_temp, obj_name=obj_name,  fitness_name=fitness_name, summary_flag=fitness_summary_flag, data_synth=df_synth, data_real=df_real
    )
    print(gan_ensemble_optimal)
    print(obj_gan_ensemble_temp)
    print('\n')

    # Save the initial history.
    step, step_comb = 0, 0
    df.loc[len(df)] = [step, step_comb, list(gan_ensemble_temp), obj_gan_ensemble_temp, len(gan_ensemble_temp)]

    # Backward search.
    while len(gan_ensemble_optimal) > 2:

        for i in range(len(gan_ensemble_optimal)):
            gan_ensemble = gan_ensemble_optimal - set([list(gan_ensemble_optimal)[i]])
            #print(gan_ensemble)
            obj_gan_ensemble = util_ensemble.backward_objective(
                sel_gans=gan_ensemble,  obj_name=obj_name, fitness_name=fitness_name, summary_flag=fitness_summary_flag, data_synth=df_synth, data_real=df_real
            )
            #print(obj_gan_ensemble)

            df.loc[len(df)] = [step + 1, i, list(gan_ensemble), obj_gan_ensemble, len(gan_ensemble)]

            if obj_gan_ensemble <= obj_gan_ensemble_temp:
                gan_ensemble_temp = gan_ensemble
                obj_gan_ensemble_temp = obj_gan_ensemble

        if gan_ensemble_temp != gan_ensemble_optimal:
            gan_ensemble_optimal = gan_ensemble_temp
            # print('\n')
            # print('Update optimal ensemble.')
            # print(gan_ensemble_optimal)
            # print('\n')
        else:
            break
        step += 1

    # plot fitness against number of gans using df.
    df.to_excel(os.path.join(ensemble_dir, f'history_gans.xlsx'), index=False)
    import matplotlib.pyplot as plt

    # Remove column ensemble
    df = df.drop(columns=['ensemble'])

    # Group by number of gans and compute the mean fitness value.
    df_ngans = df.groupby(['n_gans']).min().reset_index()
    plt.plot(df_ngans['n_gans'], df_ngans['fitness'])
    plt.xlabel('Number of GANs')
    plt.ylabel(f'{fitness_name}')
    plt.title('Min Fitness vs Number of GANs')
    plt.savefig(os.path.join(ensemble_dir, f'min_n_gans.png'))
    plt.show()

    print(gan_ensemble_optimal)
    print(obj_gan_ensemble_temp)

    print('May the force be with you.')