import sys
sys.path.extend([
    "./",
])

import os
import torch
from tqdm import tqdm
import time
import numpy as np
import argparse
import umap
import yaml
import json
import pandas as pd
import re
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import src.custom_metrics.preparation as pp
from src.general_utils import util_data
from src.general_utils import util_path
from src.custom_metrics import features
from src.general_utils import util_general
from src.general_utils import util_reports

from src.data_util import Dataset_

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
    parser.add_argument('--source_dir', type=str, default='./reports/pneumoniamnist', help='Directory name to fake samples.') #/home/lorenzo/GAN-Ensembles/reports/
    parser.add_argument('--dataset_name', type=str, default='pneumoniamnist', choices=['pneumoniamnist'],  help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000', help='Iter or Iters to sample each GAN')
    parser.add_argument('--gan_steps_ensemble', type=util_general.parse_comma_separated_list, default='100000', help='Iter or Iters to sample each GAN')
    parser.add_argument('--gan_models_ensemble', type=util_general.parse_comma_separated_list, default='ACGAN-Mod-ADC,SAGAN', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps_ensemble', type=util_general.parse_comma_separated_list, default='100000', help='Iter or Iters to sample each GAN')
    parser.add_argument('--gan_models_steps_ensemble', type=util_general.parse_comma_separated_list, default=None, help='Combinations models_steps.')
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='ResNet50_torch', help="[InceptionV3_torch, ResNet50_torch, SwAV_torch, resnet_ae_50_pneumoniamnist, disc_resnet_50_pneumoniamnist, cnn_resnet_50_pneumoniamnist")
    parser.add_argument('--n_samples', type=int, default='50000',  help='Total number of images generated by the ensemble.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])

    return parser


# main
if __name__ == "__main__":

    parser = get_parser()
    args, unknown = parser.parse_known_args()

    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Directories.
    split= args.split
    source_dir = args.source_dir
    dataset_name = args.dataset_name
    samples_dir = os.path.join(source_dir, 'samples')
    cache_dir = os.path.join(source_dir, 'features')
    reports_dir = os.path.join(source_dir, 'umap_embeddings')
    util_path.create_dir(reports_dir)

    # Parameters.
    eval_backbone= args.eval_backbone
    post_resizer= args.post_resizer
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_max_dim_red = np.inf

    # Device.
    gpu_ids = int(args.gpu_ids)
    print(f'gpu_ids: {gpu_ids}')

    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            raise ValueError('GPU specified but not available.')
        device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')

    # GANs
    gan_models_steps = args.gan_models_steps
    if len(gan_models_steps) != 0:
        gan_models = [x.split('_')[0] for x in gan_models_steps]
        gan_steps = [x.split('_')[1] for x in gan_models_steps]
    else:
        gan_models = args.gan_models
        gan_steps = args.gan_steps
    gan_models_steps_ensemble = args.gan_models_steps_ensemble
    if len(gan_models_steps) != 0:
        gan_models_ensemble = [x.split('_')[0] for x in gan_models_steps_ensemble]
        gan_steps_ensemble = [x.split('_')[1] for x in gan_models_steps_ensemble]
    else:
        gan_models_ensemble = args.gan_models_ensemble
        gan_steps_ensemble = args.gan_steps_ensemble
    gan_aval = os.listdir(samples_dir)
    filename = f'umap_{split}_{eval_backbone}_{post_resizer}_{n_samples}'

    # Create the entire path for each gan.
    gan_folders = [os.path.join(root_dir, x, f'fake__{split}') for x in gan_aval if any(f'{gan_model}-train-' in x for gan_model in gan_models)]
    gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]

    gan_folders_ensemble = [os.path.join(samples_dir, x, 'fake') for x in gan_aval if any(y in x for y in gan_models_ensemble)]
    gan_folders_ensemble = [os.path.join(x, f"step={y}") for x in gan_folders_ensemble for y in gan_steps_ensemble]

    eval_model = pp.LoadEvalModel(
        eval_backbone=eval_backbone,
        post_resizer=post_resizer,
        device=device
    )
    tot_comb = len(gan_folders)

    real_dataset = Dataset_(data_name=cfg['DATA']['name'],
                            data_dir=cfg['DATA']['data_dir'],
                            train=True if split == 'train' else False,
                            crop_long_edge=cfg['PRE']['crop_long_edge'],
                            resize_size=cfg['PRE']['resize_size'],
                            resizer=cfg['PRE']['pre_resizer'],
                            random_flip=cfg['PRE']['apply_rflip'],
                            normalize=cfg['PRE']['normalize'],
                            cfgs=cfg)
    n_max_dim_red = min(n_max_dim_red, len(real_dataset))  # take the minimum between n_max_dim_red and the length of real dataset

    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    # real
    print('Load feature real.')
    stats1 = features.compute_feature(
        dataloader=real_loader,
        eval_model=eval_model,
        batch_size=batch_size,
        quantize=True,
        device=device,
        cache_dir=cache_dir + f'/real_{split}',
        capture_all=True,
        capture_mean_cov=True
    )
    feat_real = stats1.get_all()

    # Main cycle.
    feat_all = []
    y_all = []
    feat_all.append(feat_real)
    y_all.append(np.zeros(feat_real.shape[0], dtype=int))
    gan_names = {
       'real': 0
    }

    # Compute features for all possible GANs.
    with tqdm(total=tot_comb) as pbar:
        for idx, gan_sel in enumerate(gan_folders):

            print("\n")
            print(gan_sel)

            tik = time.time()
            # Dataset
            gan0_dataset = util_data.EnsembleDataset(folders=[gan_sel], weights=[1.0])

            # Dataloder
            dataloader0 = torch.utils.data.DataLoader(gan0_dataset, batch_size=batch_size, shuffle=False)

            # Compute the features representations.
            stats0 = features.compute_feature(
                dataloader=dataloader0,
                eval_model=eval_model,
                batch_size=batch_size,
                quantize=True,
                device=device,
                cache_dir=cache_dir + gan_sel.split('samples')[-1],
                max_items=n_samples,
                capture_all=True,
                capture_mean_cov=True
            )
            feat_synth = stats0.get_all()
            feat_synth = feat_synth[:n_max_dim_red]  # select only n_max_dim_red samples
            feat_all.append(feat_synth)
            y_all.append(np.full(feat_synth.shape[0], idx + 1, dtype=int))
            gan_names[gan_sel.split('samples')[-1]] = idx + 1

    with open(os.path.join(reports_dir, f'{filename}_gan_label_map.txt'), "w") as f:
        json.dump(gan_names, f, indent=4)

    feat_real = feat_all[0]
    feat_real = np.array(feat_real).astype(np.float32)
    y_real = y_all[0]
    y_real = np.array(y_real).astype(np.int32)

    feat_all_gans = feat_all[1:]
    feat_all_gans = np.concatenate(feat_all_gans, axis=0).astype(np.float32)
    y_all_gans = y_all[1:]
    y_all_gans = np.concatenate(y_all_gans, axis=0).astype(np.int32)

    feat_ensemble = [feat_all[gan_names.get(x.split('samples')[-1])] for x in gan_folders_ensemble]
    feat_ensemble = np.concatenate(feat_ensemble, axis=0).astype(np.float32)
    y_ensemble = [y_all[gan_names.get(x.split('samples')[-1])] for x in gan_folders_ensemble]
    y_ensemble = np.concatenate(y_ensemble, axis=0).astype(np.int32)

    # Fit umap.
    print('Fit umap.')
    reducer = umap.UMAP(random_state=42)
    output_name = filename + '_' + 'fit_real'
    transformer = reducer.fit(feat_real)

    # Project latent points.
    print('Project latent points.')
    embedding_real = transformer.transform(feat_real)
    embedding_all_gans = transformer.transform(feat_all_gans)
    embedding_ensemble = transformer.transform(feat_ensemble)

    # Create a dataframe from embeddings.
    df_real = pd.DataFrame(embedding_real, columns=['umap-1', 'umap-2'])
    df_real['exp'] = 'real'

    df_naive = pd.DataFrame(embedding_all_gans, columns=['umap-1', 'umap-2'])
    df_naive['exp'] = 'naive'

    df_DGE = pd.DataFrame(embedding_ensemble, columns=['umap-1', 'umap-2'])
    df_DGE['exp'] = 'DGE'

    # Concatenate all these dataframes
    df = pd.concat([df_real, df_naive, df_DGE], ignore_index=True)

    # Save to the disk.
    df.to_excel(os.path.join(reports_dir, f'{filename}.xlsx'), index=False)
    df_real.to_excel(os.path.join(reports_dir, f'{filename}_real.xlsx'), index=False)
    df_naive.to_excel(os.path.join(reports_dir, f'{filename}_naive.xlsx'), index=False)
    df_DGE.to_excel(os.path.join(reports_dir, f'{filename}_DGE.xlsx'), index=False)

    print('May the force be with you.')