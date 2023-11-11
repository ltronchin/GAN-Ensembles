import sys
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
import copy

from src.cnn_models.models import ResNet18, ResNet50
from src.general_utils import util_general
from src.general_utils import util_cnn
from src.data_util import Dataset_
from src.general_utils import util_path
from src.general_utils import util_data

def get_parser():

    parser = argparse.ArgumentParser(description='Train CNN')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs_downstream_task/")
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument('--gpu_ids', type=str, default=1, help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument("--num_workers", type=int, default=8)

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

    gpu_ids = int(args.gpu_ids)
    num_workers = args.num_workers

    dataset_name = cfg['DATA']['name']
    n_channels = cfg['DATA']['img_channels']
    n_classes = cfg['DATA']['num_classes']
    model_name = cfg['MODEL']['name']
    task_name = cfg['MODEL']['task']
    batch_size = cfg['TRAINER']['batch_size']
    img_size = cfg['DATA']['img_size']
    weighted_loss = True

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

    report_dir = os.path.join(save_dir, 'backbone', filename)
    util_path.create_dir(report_dir)

    # Preparing data.
    datasets = {
        step: Dataset_(data_name=cfg['DATA']['name'],
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
                            cfgs=cfg) for step in ["train", "val", "test"]
    }

    if model_name in ["InceptionV3_torch", "ResNet50_torch", "ResNet18_torch"]:
        datasets = {
            step: util_data.ImageNetDataset(copy.deepcopy(datasets[step]), model_name=model_name) for step in ["train", "val", "test"]
        }
        if dataset_name == 'AIforCOVID':
            datasets_data_train = datasets['train'].dataset
        elif dataset_name in ['pneumoniamnist', 'breastmnist', 'retinamnist', 'organamnist']:
            datasets_data_train = datasets['train'].dataset.data
        else:
            raise NotImplementedError
    else:
        if dataset_name == 'AIforCOVID':
            datasets_data_train = copy.deepcopy(['train'])
        elif dataset_name in ['pneumoniamnist', 'breastmnist', 'retinamnist', 'organamnist']:
            datasets_data_train = datasets['train'].data
        else:
            raise NotImplementedError

    n_samples = len(datasets['train'])
    if dataset_name == 'AIforCOVID':
        classes = cfg['DATA']['classes']
        class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
    elif dataset_name in ['pneumoniamnist', 'breastmnist', 'retinamnist', 'organamnist']:
        idx_to_class = datasets_data_train.info['label']
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        raise NotImplementedError

    weight = [
        len(datasets_data_train) / (
                len(classes) * len(datasets_data_train.labels[datasets_data_train.labels == class_to_idx[c]]))
        for c in classes
    ]

    data_loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['TRAINER']['batch_size'], shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers)
    }

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
    if weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight)).to(device)
    else:
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
        warmup_epoch = cfg['TRAINER']['warmup_epoch'],
        model_dir=report_dir,
        device=device,
        n_samples=n_samples,
        transfer_learning_inception_v3 = True if model_name == 'InceptionV3_torch' else False
    )

    results = collections.defaultdict(lambda: [])
    util_cnn.plot_training(history=history, plot_training_dir=report_dir)
    test_results = util_cnn.evaluate(model=model, data_loader=data_loaders['test'], idx_to_class=idx_to_class, device=device)
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

    print('May the force be with you!')

