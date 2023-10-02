import sys
sys.path.extend([
    "./",
])
import os
import torch
from itertools import combinations
from tqdm import tqdm
import math
import time
import scipy
import numpy as np
import re
import pandas as pd
import uuid
import argparse
import  yaml

from src.general_utils import util_data
from src.custom_metrics import features
from src.general_utils import util_path
from src.general_utils import util_general
import src.custom_metrics.preparation as pp
from src.data_util import Dataset_

def get_parser():

    parser = argparse.ArgumentParser(description='Compute features.')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
    parser.add_argument('--source_dir', type=str, default='./reports/pneumoniamnist', help='Directory name to fake samples.') #/home/lorenzo/GAN-Ensembles/reports/
    parser.add_argument('--dataset_name', type=str, default='pneumoniamnist', choices=['pneumoniamnist', 'retinamnist', 'breastmnist'],  help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000', help='Iter or Iters to sample each GAN')
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='ResNet50_torch', help="InceptionV3_torch, SwAV_torch, resnet_ae_50_pneumoniamnist, disc_resnet_50_pneumoniamnist, cnn_resnet50_pneumoniamnist, resnet_ae_50_retinamnist, disc_resnet_50_retinamnist, cnn_resnet50_retinamnist, resnet_ae_50_breastmnist, disc_resnet_50_breastmnist, cnn_resnet50_breastmnist ")
    parser.add_argument('--n_samples', type=int, default='50000',  help='Total number of images generated by the ensemble.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--rescale', type=bool, default=True)

    return parser

# main
if __name__ == '__main__':

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

    # Parameters.
    eval_backbone= args.eval_backbone
    post_resizer= args.post_resizer
    n_samples = args.n_samples
    batch_size = args.batch_size
    rescale = args.rescale

    # Device.
    gpu_ids = int(args.gpu_ids)
    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')

    # GANs
    gan_models = args.gan_models
    gan_steps = args.gan_steps
    gan_aval = os.listdir(samples_dir)
    filename = f'history_fid_bin_real_{split}_{eval_backbone}_{post_resizer}_{n_samples}'

    # Create the entire path for each gan.
    gan_folders = [os.path.join(samples_dir, x, 'fake') for x in gan_aval if any(y in x for y in gan_models)]
    gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]

    eval_model = pp.LoadEvalModel(
        eval_backbone=eval_backbone,
        post_resizer=post_resizer,
        device=device
    )

    df = pd.DataFrame(columns=['folder0', 'folder1', 'gan0', 'gan1', 'step0', 'step1', 'time', 'fid'])
    tot_comb = len(gan_folders)

    real_dataset = Dataset_(data_name=cfg['DATA']['name'],
                            data_dir=cfg['DATA']['data_dir'],
                            train=True if split == 'train' else False,
                            split = split,
                            crop_long_edge=cfg['PRE']['crop_long_edge'],
                            resize_size=cfg['PRE']['resize_size'],
                            resizer=cfg['PRE']['pre_resizer'],
                            random_flip=cfg['PRE']['apply_rflip'],
                            normalize=cfg['PRE']['normalize'],
                            cfgs=cfg)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    # Main cycle.
    with tqdm(total=tot_comb) as pbar:
        for gan_sel in gan_folders:

            print("\n")
            print(gan_sel, 'vs', 'REAL')

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
            mu0, sigma0 = stats0.get_mean_cov()
            # dset0_feats= stats0.get_all_torch()
            # dset0_feats = dset0_feats.detach().cpu().numpy().astype(np.float64)
            # mu0 = np.mean(dset0_feats, axis=0)
            # sigma0 = np.cov(dset0_feats, rowvar=False)

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
            mu1, sigma1 = stats1.get_mean_cov()
            # dset1_feats = stats1.get_all_torch()
            # dset1_feats = dset1_feats.detach().cpu().numpy().astype(np.float64)
            # mu1 = np.mean(dset1_feats, axis=0)
            # sigma1 = np.cov(dset1_feats, rowvar=False)

            # Compute the fid.
            m = np.square(mu1 - mu0).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma0), disp=False)  # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma1 + sigma0 - s * 2))
            tok = time.time()

            if rescale:
                data_iter = iter(dataloader0)
                # Get image shapes and types.
                images, _ = next(data_iter)
                img = images[0].cpu().detach().numpy()
                img_size = img.shape[1]
                img_type = img.dtype
                filename_dummy_dataset = f'{dataset_name}_{split}-img_size_{img_size}-img_type_{img_type}-range_11-dummy_dataset'

                # Load the rescale factor from the cache (.xlsx file).
                rescale_df = pd.read_excel(os.path.join(cache_dir, f'{filename_dummy_dataset}.xlsx'))
                # Select the row according to eval_backbone.
                rescale_factor = rescale_df.loc[rescale_df['eval_backbone'] == eval_backbone]['FID'].values[0]

            # Append to dataframe.
            new_row = {
                'folder0': gan_sel,
                'folder1': 'REAL',
                'gan0': re.search(f'{dataset_name}-(.*?)-train', gan_sel).group(1),
                'gan1': 'REAL',
                'step0': re.search('step=(\d+)', gan_sel).group(1),
                'step1': 'REAL',
                'time': float(tok - tik),
                'fid': float(fid),
                'fid_rescaled': float(fid / rescale_factor) if rescale else None,
                'rescale_factor': rescale_factor if rescale else None,
            }
            df_new = pd.DataFrame([new_row])
            df = pd.concat([df, df_new], ignore_index=True)

            pbar.update(1)

    df.to_excel(os.path.join(cache_dir, f'{filename}.xlsx'), index=False)

    print("May be the force ")





