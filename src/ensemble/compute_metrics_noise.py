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
from src.general_utils import util_metric
from src.custom_metrics import features
from src.general_utils import util_path
from src.general_utils import util_general
import src.custom_metrics.preparation as pp
from src.data_util import Dataset_

def get_parser():

    parser = argparse.ArgumentParser(description='Compute metrics noise.')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
    parser.add_argument('--source_dir', type=str, default='./reports/pneumoniamnist', help='Directory name to fake samples.') #/home/lorenzo/GAN-Ensembles/reports/
    parser.add_argument('--dataset_name', type=str, default='pneumoniamnist', choices=['pneumoniamnist', 'retinamnist', 'breastmnist', 'AIforCOVID'],  help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list,  default='ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000',   help='Iter or Iters to sample each GAN')
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='InceptionV3_torch', choices=['InceptionV3_torch', 'ResNet50_torch', 'SwAV_torch', 'InceptionV3_torch__medical', 'InceptionV3_torch__truefake', 'ResNet50_torch__medical', 'ResNet50_torch__truefake'])
    parser.add_argument('--metrics_name',  type=util_general.parse_comma_separated_list, default='fid,pr,dc', help='List of metrics to compute')
    parser.add_argument('--n_samples', type=int, required=True, help='Total number of images generated by the ensemble.')
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
    n_classes = cfg['DATA']['num_classes']
    img_size = cfg['DATA']['img_size']
    samples_dir = os.path.join(source_dir, 'samples')
    cache_dir = os.path.join(source_dir, 'features')
    backbone_dir = os.path.join(source_dir, 'backbone')

    # Parameters.
    metrics_name = args.metrics_name
    eval_backbone= args.eval_backbone
    post_resizer= args.post_resizer
    batch_size = args.batch_size
    n_samples = args.n_samples

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
        device=device,
        eval_backbone_dir=backbone_dir,
        n_classes=n_classes
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
                            hdf5_path=os.path.join(cfg['DATA']['hdf5'], f'{dataset_name}_{img_size}_{split}.hdf5') if cfg['DATA']['hdf5'] is not None else None,
                            load_data_in_memory=cfg['DATA']['load_data_in_memory'],
                            cfgs=cfg)

    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    real_iter = iter(real_loader)
    # Get image shapes and types.
    real_images, real_labels = next(real_iter)
    img = real_images[0].cpu().detach().numpy()
    n_channels = img.shape[0]
    img_size = img.shape[1]
    img_type = img.dtype

    filename_dummy_dataset = f'{dataset_name}_{split}-img_size_{img_size}-img_type_{img_type}-range_11-dummy_dataset'
    filename = f'inter_dummy_dataset-{split}-{eval_backbone}-{post_resizer}-{n_samples}'

    # Main cycle.
    print("\n")
    print('Dummy Dataset', 'vs', 'REAL')
    tik = time.time()

    # Dataset
    dummy_dataset = util_data.DummyDataset(folder=os.path.join(samples_dir, filename_dummy_dataset))

    # Dataloder
    dataloader0 = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)

    # Compute the features representations.
    stats0 = features.compute_feature(
        dataloader=dataloader0,
        eval_model=eval_model,
        batch_size=batch_size,
        quantize=True,
        device=device,
        cache_dir=os.path.join(cache_dir, filename_dummy_dataset),
        max_items=n_samples,
        capture_all=True,
        capture_mean_cov=True
    )
    mu0, sigma0 = stats0.get_mean_cov()
    feats0 = stats0.get_all_torch()
    feats0 = feats0.detach().cpu().numpy().astype(np.float64)

    stats1 = features.compute_feature(
        dataloader=real_loader,
        eval_model=eval_model,
        batch_size=batch_size,
        quantize=True,
        device=device,
        cache_dir=cache_dir + f'/real_{split}',
        max_items=n_samples,
        capture_all=True,
        capture_mean_cov=True
    )
    mu1, sigma1 = stats1.get_mean_cov()
    feats1 = stats1.get_all_torch()
    feats1 = feats1.detach().cpu().numpy().astype(np.float64)

    # Compute the metrics.
    eps = 1e-6
    if 'fid' in metrics_name:
        fid = util_metric.FID(mu0, sigma0, mu1, sigma1)
        fid = float(fid)
    else:
        fid = 'NA'
    if 'pr' in metrics_name:
        prc, rec = util_metric.compute_pr(feats0, feats1, nearest_k=5)
        prc, rec = float(prc), float(rec)
        f1_prc_rec = 2 * prc * rec / ((prc + rec) + eps)
        f1_prc_rec = float(f1_prc_rec)
    else:
        prc = 'NA'
        rec = 'NA'
        f1_prc_rec = 'NA'
    if 'dc' in metrics_name:
        dns, cvg = util_metric.compute_dc(feats0, feats1, nearest_k=5)
        dns, cvg = float(dns), float(cvg)
        # Avoid division by 0
        f1_dns_cvg = 2 * dns * cvg / ((dns + cvg) + eps)
        f1_dns_cvg = float(f1_dns_cvg)
    else:
        dns = 'NA'
        cvg = 'NA'
        f1_dns_cvg = 'NA'

    tok = time.time()

    # Save to excel file as .xlsx file with colums split,eval_backbone,post_resizer,FID,Time.
    if os.path.isfile(os.path.join(cache_dir, f'{filename}.xlsx')):
        df = pd.read_excel(os.path.join(cache_dir, f'{filename}.xlsx'))
        df = df.append(
            {
                'split': split,
                'eval_backbone': eval_backbone,
                'post_resizer': post_resizer,
                'time': float(tok - tik),
                'fid': fid,
                'prc': prc,
                'rec': rec,
                'f1_prc_rec': f1_prc_rec,
                'dns': dns,
                'cvg': cvg,
                'f1_dns_cvg': f1_dns_cvg

            }, ignore_index=True
        )
    else:
        df = pd.DataFrame([
            [split, eval_backbone, post_resizer, float(tok-tik), fid, prc, rec, f1_prc_rec, dns, cvg, f1_dns_cvg]
        ], columns=['split', 'eval_backbone', 'post_resizer', 'time', 'fid', 'prc','rec','f1_prc_rec', 'dns', 'cvg', 'f1_dns_cvg'])
    df.to_excel(os.path.join(cache_dir, f'{filename}.xlsx'), index=False)

    print("May be the force with you.")