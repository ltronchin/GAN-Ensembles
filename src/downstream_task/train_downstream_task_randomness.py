import copy
import sys

import numpy as np

sys.path.extend([
    "./"
])
import argparse
import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
from datetime import datetime
import pandas as pd
import collections
import seaborn as sns

from src.cnn_models.models import ResNet18, ResNet50
from src.general_utils import util_general
from src.general_utils import util_cnn
from src.data_util import Dataset_
from src.general_utils import util_path
from src.general_utils import util_ensemble

from src.general_utils.util_data import EnsembleDataset

# For aspect ratio 4:3.
sns.set_context("paper")
sns.set_theme(style="ticks")

def get_parser():

    parser = argparse.ArgumentParser(description='Train CNN')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs_downstream_task/")
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument('--gpu_ids', type=str, default=1, help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--real_flag', default=1, type=int)
    parser.add_argument('--random_flag', default=0, type=int)
    parser.add_argument('--num_random_flag', default=10, type=int)
    parser.add_argument('--n_train', type=int, default=20, help='Number of times the experiment is repeated.')

    # GANs params.
    parser.add_argument('--init_w', type=str, default='uniform', choices=['uniform', 'random', 'fid'], help='Weight initialization')
    parser.add_argument('--samples_dir', type=str, default='./reports/',  help='Directory name to fake samples.')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='StyleGAN2', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000', help='Epoch or Epochs to sample each GAN')
    parser.add_argument('--n_times', type=int, default=1, help='Number of times of training real dataset.')

    return parser

if __name__ == "__main__":

    # Configuration file
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything.
    util_general.seed_all(cfg['SEED'])

    # Parameters.
    save_dir = args.save_dir
    samples_dir = args.samples_dir
    folder_name = 'downstream_task_competitors'

    gpu_ids = int(args.gpu_ids)
    num_workers = args.num_workers

    dataset_name = cfg['DATA']['name']
    n_channels = cfg['DATA']['img_channels']
    n_classes = cfg['DATA']['num_classes']
    img_size = cfg['DATA']['img_size']
    model_name = cfg['MODEL']['name']
    batch_size = cfg['TRAINER']['batch_size']
    n_train = args.n_train

    # Parameters GAN.
    gan_models = args.gan_models
    gan_steps = args.gan_steps
    init_w = args.init_w
    n_times = args.n_times
    real_flag = args.real_flag
    random_flag = args.random_flag
    num_random_flag = args.num_random_flag

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
    if real_flag:
        filename = f'real'
    elif random_flag:
        filename = f'random--num_random_flag__{num_random_flag}'
    else:
        if len(gan_models) == 22 and len(gan_steps) == 5:
            filename = f'naive_models_steps--models__naive--steps__{util_general.parse_separated_list_comma(gan_steps)}--n_times__{n_times}'
        elif len(gan_models) == 22 and len(gan_steps) != 5:
            filename = f'naive_models--models__naive--steps__{util_general.parse_separated_list_comma(gan_steps)}--n_times__{n_times}'
        elif len(gan_models) != 22 and len(gan_steps) == 5:
            filename = f'naive_steps--models__{util_general.parse_separated_list_comma(gan_models)}--steps__{util_general.parse_separated_list_comma(gan_steps)}--n_times__{n_times}'
        else:
            filename = f'single_gan--models__{util_general.parse_separated_list_comma(gan_models)}--steps__{util_general.parse_separated_list_comma(gan_steps)}--n_times__{n_times}'

    report_dir = os.path.join(save_dir, f'{folder_name}', filename)
    util_path.create_dir(report_dir)

    # Preparing data.
    datasets_real = {
        step: Dataset_(
                data_name=cfg['DATA']['name'],
                data_dir=cfg['DATA']['data_dir'],
                train=True if step == 'train' else False,
                split=step,
                crop_long_edge=cfg['PRE']['crop_long_edge'],
                resize_size=cfg['PRE']['resize_size'],
                resizer=cfg['PRE']['pre_resizer'],
                random_flip=cfg['PRE']['apply_rflip'],
                normalize=cfg['PRE']['normalize'],
                hdf5_path=os.path.join(cfg['DATA']['hdf5'], f'{dataset_name}_{img_size}_{step}.hdf5') if cfg['DATA']['hdf5'] is not None else None,
                load_data_in_memory=cfg['DATA']['load_data_in_memory'],
                cfgs=cfg) for step in ['train', 'val', 'test']
    }
    if dataset_name == 'AIforCOVID':
        n_samples = len(datasets_real['train'])
        classes = cfg['DATA']['classes']
        class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        weight = [
            len(datasets_real['train']) / (len(classes) * len(datasets_real['train'].labels[datasets_real['train'].labels == class_to_idx[c]])) for c in classes
        ]
    elif dataset_name in ['pneumoniamnist', 'breastmnist', 'retinamnist', 'organamnist']:
        n_samples = len(datasets_real['train'].data)
        idx_to_class = datasets_real['test'].data.info['label']
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        classes = [idx_to_class[i] for i in range(len(idx_to_class))]
        weight = [
            len(datasets_real['train'].data) / (len(classes) * len(datasets_real['train'].data.labels[datasets_real['train'].data.labels == class_to_idx[c]])) for c in classes
        ]
    else:
        raise NotImplementedError

    if not real_flag:
        print('Using synthetic data...')
        gan_aval = os.listdir(samples_dir)

        gan_folders = [os.path.join(samples_dir, x, f'fake__train') for x in gan_aval if any(f'{gan_model}-train-' in x for gan_model in gan_models)]
        gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]
        if random_flag:
            #num_random_flag = np.random.randint(2, len(gan_models) * len(gan_steps))
            gan_folders = [str(x) for x in np.random.choice(gan_folders, size=num_random_flag, replace=False)]
            # Save to the disk the txt of gan_folders.
            with open(os.path.join(report_dir, 'sel_gans.txt'), 'w') as f:
                for item in gan_folders:
                    f.write("%s\n" % item)

        weights = util_ensemble.initialize_ensemble_weights(init_w=init_w, gan_list=gan_folders)
        dataset_synth = EnsembleDataset(folders=gan_folders, weights=weights)
        datataset_train = copy.deepcopy(dataset_synth)
        n_samples = n_samples * n_times
    else:
        datataset_train = copy.deepcopy(datasets_real['train'])

    datasets = {
        'train': datataset_train,
        'val': datasets_real['val'],
        'test': datasets_real['test']
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['TRAINER']['batch_size'], shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers)
    }

    # List to save results for each training.
    results = collections.defaultdict(lambda: [])
    for idx_train in np.arange(n_train):
        results['Training'].append(idx_train)
        print(f"Training {idx_train+1}/{n_train}...")

        # Model.
        print('==> Building and training model...')
        if model_name == 'resnet18':
            model = ResNet18(input_channels=n_channels, num_classes=n_classes)
        elif model_name == 'resnet50':
            model = ResNet50(input_channels=n_channels, num_classes=n_classes)
        else:
            raise NotImplementedError
        model = model.to(device)

        # Optimizer.
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight)).to(device)
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
            n_samples=n_samples,
            to_disk=False
        )

        # Plot Training.
        util_cnn.plot_training(history=history, plot_training_dir=report_dir, plot_name_loss=f'Loss_training_{idx_train}', plot_name_acc=f'Acc_training_{idx_train}')

        # Test model.
        test_results = util_cnn.evaluate(model=model, data_loader=data_loaders['test'], device=device, idx_to_class=idx_to_class)

        # Update report.
        results["ACC"].append(test_results['all'])
        for c in classes:
            results["ACC %s" % str(c)].append(test_results[c])
        results['recall'].append(test_results['recall'])
        results['precision'].append(test_results['precision'])
        results['specificity'].append(test_results['specificity'])
        results['f1_score'].append(test_results['f1_score'])
        results['g_mean'].append(test_results['g_mean'])

        # Save results
        report_file = os.path.join(report_dir, f'results_training_{idx_train}.xlsx')
        results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
        results_frame.to_excel(report_file, index=False)

    # Compute mean and std.
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    results_frame.loc['mean'] = results_frame.mean()
    results_frame.loc['std'] = results_frame.std()

    # Save Results
    report_file = os.path.join(report_dir, 'results.xlsx')
    results_frame.to_excel(report_file, index=True)

    print('May the force be with you!')