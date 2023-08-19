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

import src.general_utils.util_general as util_general
from src.general_utils import util_path
import src.general_utils.util_autoencoder as util_model
from src.cnn_models.models import ResNet18, ResNet50
from src.general_utils import util_cnn
from src.general_utils.util_data import DiscDataset, MergedDataset
from src.data_util import Dataset_


RUN_NAME_FORMAT = ("{data_name}-" "{framework}-" "{phase}-" "{timestamp}")

def make_run_name(format, data_name, framework, phase):
    return format.format(data_name=data_name,
                         framework=framework,
                         phase=phase,
                         timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

def has_tiff_files(dir_path):
    """
    Recursively checks if the directory contains any .tiff files.
    """
    for root, dirs, files in os.walk(dir_path):
        if any(x.endswith('.tiff') for x in files):
            return True
    return False

def get_parser():
    parser = argparse.ArgumentParser(description='Train resnet autoencoder.')

    parser.add_argument('--source_dir', type=str, default='/home/lorenzo/GAN-Ensembles/reports/pneumoniamnist',  help='Directory name to fake samples.')
    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument('--gpu_ids', type=str, default=1, help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list,  default='ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000',   help='Iter or Iters to sample each GAN')
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
    dataset_name = cfg['DATA']['name']
    batch_size = cfg['TRAINER']['batch_size']
    gpu_ids = int(args.gpu_ids)

    # Device.
    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            raise ValueError('GPU specified but not available.')
        device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')

    # Files and Directories.
    run_name = make_run_name(
        RUN_NAME_FORMAT,
        data_name=dataset_name,
        framework=model_name,
        phase="train"
    )
    report_dir = os.path.join(args.save_dir, 'backbone', run_name)
    util_path.create_dir(report_dir)

    # Select only the GAN included in gan_models and available to the disk.
    samples_dir = os.path.join(source_dir, 'samples')
    gan_aval = os.listdir(samples_dir)

    gan_folders = [os.path.join(samples_dir, x, 'fake') for x in gan_aval if any(y in x for y in gan_models)]
    gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]
    # Filter out empty directories
    gan_folders = [x for x in gan_folders if has_tiff_files(x)]

    n_classes = len(gan_folders) + 1

    # Dataset.

    datasets_real = {
        step: Dataset_(data_name=cfg['DATA']['name'],
                            data_dir=cfg['DATA']['data_dir'],
                            train=True if step == 'train' else False,
                            crop_long_edge=cfg['PRE']['crop_long_edge'],
                            resize_size=cfg['PRE']['resize_size'],
                            resizer=cfg['PRE']['pre_resizer'],
                            random_flip=cfg['PRE']['apply_rflip'],
                            normalize=cfg['PRE']['normalize'],
                            cfgs=cfg)  for step in ["train", "val"]
    }

    dataset_synth = DiscDataset(root_dir=samples_dir, classes=gan_folders, max_samples_per_gan=len(datasets_real['train']))
    train_dataset = MergedDataset(copy.deepcopy(dataset_synth), copy.deepcopy(datasets_real['train']))
    class_real = train_dataset.class_real

    data_loaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(datasets_real['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Model.
    print('Building and training model')
    model = ResNet50(in_channels=cfg['DATA']['img_channels'], num_classes=n_classes)
    model = model.to(device)

    # Loss function.
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=cfg['TRAINER']['optimizer']['lr'], weight_decay=cfg['TRAINER']['optimizer']['weight_decay'])

    # LR Scheduler.
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
        model_dir=report_dir,
        device=device,
        class_real = train_dataset.class_real
    )

    # Plot Training.
    util_cnn.plot_training(history=history, plot_training_dir=report_dir)

    # Save dataframe history as excel.
    df = pd.DataFrame.from_dict(history)
    df.to_excel(os.path.join(report_dir, 'history_perf.xlsx'), index=False)

    print("May the force be with you!")
