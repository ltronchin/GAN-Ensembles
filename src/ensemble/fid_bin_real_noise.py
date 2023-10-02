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
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='InceptionV3_torch', help="InceptionV3_torch, SwAV_torch, resnet_ae_50_pneumoniamnist, disc_resnet_50_pneumoniamnist, cnn_resnet50_pneumoniamnist, resnet_ae_50_retinamnist, disc_resnet_50_retinamnist, cnn_resnet50_retinamnist, resnet_ae_50_breastmnist, disc_resnet_50_breastmnist, cnn_resnet50_breastmnist ")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])

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
    batch_size = args.batch_size

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

    # Noise folder.

    eval_model = pp.LoadEvalModel(
        eval_backbone=eval_backbone,
        post_resizer=post_resizer,
        device=device
    )
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
    n_samples = len(real_dataset)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    real_iter = iter(real_loader)
    # Get image shapes and types.
    real_images, real_labels = next(real_iter)
    img = real_images[0].cpu().detach().numpy()
    n_channels = img.shape[0]
    img_size = img.shape[1]
    img_type = img.dtype

    filename = f'{dataset_name}_{split}-img_size_{img_size}-img_type_{img_type}-range_11-dummy_dataset'

    # Main cycle.
    print("\n")
    print('Dummy Dataset', 'vs', 'REAL')
    tik = time.time()
    # Dataset
    dummy_dataset = util_data.DummyDataset(folder=os.path.join(samples_dir, filename))

    # Dataloder
    dataloader0 = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)

    # Compute the features representations.
    stats0 = features.compute_feature(
        dataloader=dataloader0,
        eval_model=eval_model,
        batch_size=batch_size,
        quantize=True,
        device=device,
        cache_dir=os.path.join(cache_dir, filename),
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

    print(eval_backbone)
    print(f'FID: {fid:.3f}')

    # Save to excel file as .xlsx file with colums split,eval_backbone,post_resizer,FID,Time.
    if os.path.isfile(os.path.join(cache_dir, f'{filename}.xlsx')):
        df = pd.read_excel(os.path.join(cache_dir, f'{filename}.xlsx'))
        df = df.append({'split': split, 'eval_backbone': eval_backbone, 'post_resizer': post_resizer, 'FID': fid, 'Time': tok-tik}, ignore_index=True)
    else:
        df = pd.DataFrame([[split, eval_backbone, post_resizer, fid, tok-tik]], columns=['split', 'eval_backbone', 'post_resizer', 'FID', 'Time'])
    df.to_excel(os.path.join(cache_dir, f'{filename}.xlsx'), index=False)

    print("May be the force with you.")