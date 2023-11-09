import copy
import sys
sys.path.extend(["./"])

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import yaml
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.utils.data import random_split
import collections

import src.general_utils.util_general as util_general
from src.general_utils import util_path
import src.general_utils.util_autoencoder as util_model
from src.cnn_models.models import ResNet18, ResNet50
from src.general_utils import util_cnn
from src.general_utils.util_data import DiscDataset, MergedDataset
from src.data_util import Dataset_

def has_tiff_files(dir_path):
    """
    Recursively checks if the directory contains any .tiff files.
    """
    for root, dirs, files in os.walk(dir_path):
        if any(x.endswith('.tiff') for x in files):
            return True
    return False

def has_npy_files(dir_path):
    """
    Recursively checks if the directory contains any .npy files.
    """
    for root, dirs, files in os.walk(dir_path):
        if any(x.endswith('.npy') for x in files):
            return True
    return False

def get_parser():
    parser = argparse.ArgumentParser(description='Train resnet autoencoder.')

    parser.add_argument('--source_dir', type=str, default='./reports/',  help='Directory name to fake samples.')
    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument('--gpu_ids', type=str, default=1, help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list,  default='ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000', help='Iter or Iters to sample each GAN')
    parser.add_argument("--num_workers", type=int, default=8)
    return parser

if __name__ == "__main__":

    # Configuration file.
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything.
    util_general.seed_all(cfg['SEED'])

    # Parameters.
    source_dir = args.source_dir
    gan_models = args.gan_models
    gan_steps = args.gan_steps
    num_workers = args.num_workers
    model_name = cfg['MODEL']['name']
    task_name = cfg['MODEL']['task']
    dataset_name = cfg['DATA']['name']
    batch_size = cfg['TRAINER']['batch_size']
    gpu_ids = int(args.gpu_ids)

    # Device.
    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')

    # Files and Directories.
    filename = f'{model_name}__{task_name}'
    report_dir = os.path.join(args.save_dir, 'backbone', filename)
    util_path.create_dir(report_dir)

    # Select only the GAN included in gan_models and available to the disk.
    samples_dir = os.path.join(source_dir, 'samples')

    # Dataset.
    datasets_real = {
        step: Dataset_(data_name=cfg['DATA']['name'],
                            data_dir=cfg['DATA']['data_dir'],
                            train=True if step == 'train' else False,
                            split=step,
                            crop_long_edge=cfg['PRE']['crop_long_edge'],
                            resize_size=cfg['PRE']['resize_size'],
                            resizer=cfg['PRE']['pre_resizer'],
                            random_flip=cfg['PRE']['apply_rflip'],
                            normalize=cfg['PRE']['normalize'],
                            cfgs=cfg)  for step in ["train", "val", "test"]
    }
    datasets_synth = {
        step: DiscDataset(root_dir=samples_dir, split=step, gan_models=gan_models, gan_steps=gan_steps, max_samples_per_gan=len(datasets_real[step]))  for step in ["train", "val", "test"]
    }
    n_classes = len(datasets_synth['train'].classes) + 1

    # Merge the two datasets.
    datasets = {
        step: MergedDataset(copy.deepcopy(datasets_synth[step]), copy.deepcopy(datasets_real[step]), model_name=model_name)  for step in ["train", "val", "test"]
    }
    class_real = datasets['train'].class_real
    data_loaders = {
        step: torch.utils.data.DataLoader(datasets[step], batch_size=batch_size, shuffle=True if step=='train' else False, num_workers=num_workers) for step in ["train", "val", "test"]
    }

    dataiter = iter(data_loaders["train"])
    x, _ = next(dataiter)
    print("\n")
    print("Shape")
    print(x.shape)

    # Model.
    print('Building and training model.')
    if model_name == 'ResNet50_torch':
        model = torch.hub.load("pytorch/vision:v0.10.0", 'resnet50', weights='ResNet50_Weights.DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif model_name == 'InceptionV3_torch':
        model = torch.hub.load("pytorch/vision:v0.10.0", 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif model_name == 'ResNet18_torch':
        model = torch.hub.load("pytorch/vision:v0.10.0", 'resnet18', weights='ResNet18_Weights.DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif model_name == 'ResNet50_custom':
        model = ResNet50(input_channels=cfg['DATA']['img_channels'], num_classes=n_classes)
    elif model_name == 'ResNet18_custom':
        model = ResNet18(input_channels=cfg['DATA']['img_channels'], num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    # Loss function.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['TRAINER']['optimizer']['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['TRAINER']['scheduler']['mode'], patience=cfg['TRAINER']['scheduler']['patience'])

    # Train model.
    model, history = util_cnn.train_model(
        model=model,
        data_loaders=data_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg['TRAINER']['max_epochs'],
        early_stopping=cfg['TRAINER']['early_stopping'],
        warmup_epoch=cfg['TRAINER']['warmup_epoch'],
        model_dir=report_dir,
        device=device,
        transfer_learning_inception_v3=True if model_name == 'InceptionV3_torch' else False
    )

    #Plot Training.
    util_cnn.plot_training(history=history, plot_training_dir=report_dir)

    # Test model.
    idx_to_class = {idx: x for idx, x in enumerate(data_loaders['test'].dataset.dataset_synth.classes)}
    class_real = data_loaders['test'].dataset.class_real
    idx_to_class[class_real] = 'real'
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    test_results = util_cnn.evaluate(model=model, data_loader=data_loaders['test'], device=device, idx_to_class=idx_to_class)

    # Update report.
    results = collections.defaultdict(lambda: [])
    results["ACC"].append(test_results['all'])
    for c in classes:
        results["ACC %s" % str(c)].append(test_results[c])
    results['recall'].append(test_results['recall'])
    results['precision'].append(test_results['precision'])
    results['specificity'].append(test_results['specificity'])
    results['f1_score'].append(test_results['f1_score'])
    results['g_mean'].append(test_results['g_mean'])

    # Save Results
    report_file = os.path.join(report_dir, 'results.xlsx')
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    results_frame.to_excel(report_file, index=False)

    print("May the force be with you!")