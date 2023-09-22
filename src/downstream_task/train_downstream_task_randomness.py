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

from src.cnn_models.models import ResNet18, ResNet50
from src.general_utils import util_general
from src.general_utils import util_cnn
from src.data_util import Dataset_
from src.general_utils import util_path
from src.general_utils import util_ensemble

from src.general_utils.util_data import EnsembleDataset

RUN_NAME_FORMAT = ("{data_name}-" "{gan_models}-" "{gan_steps}-" "{metric_name}-" "{cost_name}-" "{backbone}-" "{n_times}-" "{timestamp}")

def make_run_name(format, data_name, gan_models, gan_steps, metric_name=None, cost_name=None, backbone=None, n_times=1):
    return format.format(data_name=data_name,
                         gan_models=gan_models,
                         gan_steps=gan_steps,
                         metric_name=metric_name,
                         cost_name=cost_name,
                         backbone=backbone,
                         n_times=f'n_times-{n_times}',
                         timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

def get_parser():

    parser = argparse.ArgumentParser(description='Train CNN')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs_downstream_task/")
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument('--gpu_ids', type=str, default=1, help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--real_flag', default=1, type=int)
    parser.add_argument('--n_train', type=int, default=20, help='Number of times the experiment is repeated.')

    # GANs params.
    parser.add_argument('--init_w', type=str, default='uniform', choices=['uniform', 'random', 'fid'], help='Weight initialization')
    parser.add_argument('--samples_dir', type=str, default='./reports/pneumoniamnist',  help='Directory name to fake samples.')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='StyleGAN2', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000', help='Epoch or Epochs to sample each GAN')
    parser.add_argument('--gan_models_steps', type=util_general.parse_comma_separated_list, default=None, help='Combinations models_steps.')
    parser.add_argument('--fitness_name', type=str, default=None)
    parser.add_argument('--cost_name', type=str, default=None)
    parser.add_argument('--eval_backbone', type=str, default=None)
    parser.add_argument('--n_times', type=int, default=5, help='Number of times of training real dataset.')

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
    data_dir = cfg['DATA']['data_dir']
    samples_dir = args.samples_dir
    folder_name = 'downstream_task_randomness_warmup_weighted_times5' # 'downstream_task_randomness_warmup_weighted' # 'downstream_task_randomness_warmup'

    gpu_ids = int(args.gpu_ids)
    num_workers = args.num_workers

    dataset_name = cfg['DATA']['name']
    n_channels = cfg['DATA']['img_channels']
    n_classes = cfg['DATA']['num_classes']
    model_name = cfg['MODEL']['name']
    batch_size = cfg['TRAINER']['batch_size']
    n_train = args.n_train

    # Parameters GAN.
    gan_models_steps = args.gan_models_steps
    if len(gan_models_steps) != 0:
        gan_models = [x.split('_')[0] for x in gan_models_steps]
        gan_steps = [x.split('_')[1] for x in gan_models_steps]
    else:
        gan_models = args.gan_models
        gan_steps = args.gan_steps
    init_w = args.init_w
    n_times = args.n_times
    real_flag = args.real_flag
    # Parameter ensemble search.
    fitness_name = args.fitness_name
    cost_name = args.cost_name
    eval_backbone = args.eval_backbone

    # Device.
    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            raise ValueError('GPU specified but not available.')
        device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')

    # Files and Directories.
    if real_flag:
        run_name = make_run_name(
            RUN_NAME_FORMAT,
            data_name=dataset_name,
            gan_models='real',
            gan_steps='real',
            n_times=1,
        )
    else:
        # One GAN
        if len(gan_models) == 1:
            run_name = make_run_name(
                RUN_NAME_FORMAT,
                data_name=dataset_name,
                gan_models=util_general.parse_separated_list_comma(gan_models),
                gan_steps=util_general.parse_separated_list_comma(gan_steps),
                n_times=n_times,
            )
        elif len(gan_models) == 22:
            # All GANs
            run_name = make_run_name(
                RUN_NAME_FORMAT,
                data_name=dataset_name,
                gan_models='all',
                gan_steps=util_general.parse_separated_list_comma(gan_steps),
                n_times=n_times,
            )
        else:
            # Ensembles
            run_name = make_run_name(
                RUN_NAME_FORMAT,
                data_name=dataset_name,
                gan_models=util_general.parse_separated_list_comma(gan_models),
                gan_steps=util_general.parse_separated_list_comma(gan_steps),
                metric_name=fitness_name,
                cost_name=cost_name,
                backbone=eval_backbone,
                n_times=n_times,
            )

    report_dir = os.path.join(save_dir, f'{folder_name}', run_name)
    util_path.create_dir(report_dir)

    # Preparing data.
    dataset_train =  Dataset_(
        data_name=cfg['DATA']['name'],
        data_dir=data_dir,
        train=True,
        split='train',
        crop_long_edge=cfg['PRE']['crop_long_edge'],
        resize_size=cfg['PRE']['resize_size'],
        resizer=cfg['PRE']['pre_resizer'],
        random_flip=cfg['PRE']['apply_rflip'],
        normalize=cfg['PRE']['normalize'],
        cfgs=cfg
    )
    n_samples = len(dataset_train)
    dataset_val = Dataset_(
            data_name=cfg['DATA']['name'],
            data_dir=data_dir,
            train=False,
            split='val',
            crop_long_edge=cfg['PRE']['crop_long_edge'],
            resize_size=cfg['PRE']['resize_size'],
            resizer=cfg['PRE']['pre_resizer'],
            random_flip=cfg['PRE']['apply_rflip'],
            normalize=cfg['PRE']['normalize'],
            cfgs=cfg)
    dataset_test = Dataset_(
            data_name=cfg['DATA']['name'],
            data_dir=data_dir,
            train=False,
            split='test',
            crop_long_edge=cfg['PRE']['crop_long_edge'],
            resize_size=cfg['PRE']['resize_size'],
            resizer=cfg['PRE']['pre_resizer'],
            random_flip=cfg['PRE']['apply_rflip'],
            normalize=cfg['PRE']['normalize'],
            cfgs=cfg
        )
    idx_to_class = dataset_test.data.info['label']
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    weight = [
        len(dataset_train.data) / (
                    len(classes) * len(dataset_train.data.labels[dataset_train.data.labels == class_to_idx[c]]))
        for c in classes
    ]

    if not real_flag:
        print('Using synthetic data...')
        gan_aval = os.listdir(samples_dir)

        gan_folders = [os.path.join(samples_dir, x, 'fake') for x in gan_aval if any(y in x for y in gan_models)]
        gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]

        weights = util_ensemble.initialize_ensemble_weights(init_w=init_w, gan_list=gan_folders)
        dataset_train = EnsembleDataset(folders=gan_folders, weights=weights)
        n_samples = n_samples * n_times

    datasets = {
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['TRAINER']['batch_size'], shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers)
    }

    # List to save results for each training.
    results = collections.defaultdict(lambda: [])
    for i_train in np.arange(n_train):
        results['Training'].append(i_train)
        print(f"Training {i_train+1}/{n_train}...")

        # Model.
        print('==> Building and training model...')
        if model_name == 'resnet18':
            model = ResNet18(in_channels=n_channels, num_classes=n_classes)
        elif model_name == 'resnet50':
            model = ResNet50(in_channels=n_channels, num_classes=n_classes)
        else:
            raise NotImplementedError
        model = model.to(device)

        # Loss function.
        if 'weighted' in folder_name:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight)).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        # Optimizer.
        optimizer = optim.Adam(model.parameters(), lr=cfg['TRAINER']['optimizer']['lr'])
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
            warmup_epoch=cfg['TRAINER']['warmup_epoch'],
            model_dir=report_dir,
            device=device,
            n_samples=n_samples,
            to_disk=False
        )

        # Plot Training.
        util_cnn.plot_training(history=history, plot_training_dir=report_dir, plot_name_loss=f'Loss_training_{i_train}', plot_name_acc=f'Acc_training_{i_train}')

        # Test model.
        test_results = util_cnn.evaluate(dataset_name=dataset_name, model=model, data_loader=data_loaders['test'], device=device)

        # Update report.
        results["ACC"].append(test_results['all'])
        for c in classes:
            results["ACC %s" % str(c)].append(test_results[c])
        results['recall'].append(test_results['recall'])
        results['precision'].append(test_results['precision'])
        results['f1_score'].append(test_results['f1_score'])
        results['geometric_mean'].append(test_results['geometric_mean'])
        results['auc'].append(test_results['auc'])

        # Save results
        report_file = os.path.join(report_dir, f'results_training_{i_train}.xlsx')
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

