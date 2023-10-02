import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import collections
import yaml
from datetime import datetime
import argparse

import src.general_utils.util_general as util_general
from src.general_utils import util_path
import src.general_utils.util_data as util_data
import src.general_utils.util_autoencoder as util_model
from src.data_util import Dataset_

RUN_NAME_FORMAT = ("{data_name}-" "{framework}-" "{phase}-" "{timestamp}")

def make_run_name(format, data_name, framework, phase):
    return format.format(data_name=data_name,
                         framework=framework,
                         phase=phase,
                         timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

def get_parser():
    parser = argparse.ArgumentParser(description='Train resnet autoencoder.')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
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
    data_dir = cfg['DATA']['data_dir']
    model_name = cfg['MODEL']['name']
    dataset_name = cfg['DATA']['name']
    num_workers = args.num_workers
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
    run_name = make_run_name(
        RUN_NAME_FORMAT,
        data_name=dataset_name,
        framework=model_name,
        phase="train"
    )
    report_dir = os.path.join(args.save_dir, 'backbone', run_name)
    util_path.create_dir(report_dir)

    # Dataset.
    datasets = {
        step: Dataset_(data_name=cfg['DATA']['name'],
                            data_dir=data_dir,
                            train=True if step == 'train' else False,
                            split=step,
                            crop_long_edge=cfg['PRE']['crop_long_edge'],
                            resize_size=cfg['PRE']['resize_size'],
                            resizer=cfg['PRE']['pre_resizer'],
                            random_flip=cfg['PRE']['apply_rflip'],
                            normalize=cfg['PRE']['normalize'],
                            cfgs=cfg) for step in ["train", "val", "test"]
    }
    data_loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['TRAINER']['batch_size'], shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers)
    }

    # Model
    dataiter = iter(data_loaders["train"])
    input, _ = next(dataiter)

    input_dim = input[0].shape[1]
    model = util_model.get_img_autoencoder(model_name=model_name, input_dim=input_dim, h_dim=cfg['MODEL']['h_dim'], input_channels=cfg['DATA']['img_channels'])
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=cfg['TRAINER']['optimizer']['lr'], weight_decay=cfg['TRAINER']['optimizer']['weight_decay'])

    # LR Scheduler.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['TRAINER']['scheduler']['mode'], patience=cfg['TRAINER']['scheduler']['patience'])

    # Train model.
    model, history = util_model.train_autoencoder(
        model=model,
        data_loaders=data_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg['TRAINER']['max_epochs'],
        early_stopping=cfg['TRAINER']['early_stopping'],
        warmup_epoch=cfg['TRAINER']['warmup_epoch'],
        model_dir=report_dir,
        device=device
    )

    # Plot Training.
    util_model.plot_training(history=history, plot_training_dir=report_dir)
    util_model.evaluate(model=model, data_loader=data_loaders['val'], report_dir=report_dir, device=device, split='val')
    util_model.evaluate(model=model, data_loader=data_loaders['test'], report_dir=report_dir, device=device, split='test')

    # Plot.
    rec_dir = os.path.join(report_dir, 'rec')
    util_path.create_dir(rec_dir)
    util_model.plot_evaluate_img_autoencoder(model=model, data_loader=data_loaders['test'], plot_dir=rec_dir, device=device)

    # Save dataframe history as excel.
    df = pd.DataFrame.from_dict(history)
    df.to_excel(os.path.join(report_dir, 'history.xlsx'), index=False)

    print("May the force be with you!")
