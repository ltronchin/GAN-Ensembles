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

from src.cnn_models.models import ResNet18, ResNet50
from src.general_utils import util_general
from src.general_utils import util_cnn
from src.data_util import Dataset_
from src.general_utils import util_path

RUN_NAME_FORMAT = ("{data_name}-" "{framework}-" "{phase}-" "{timestamp}")

def make_run_name(format, data_name, framework, phase):
    return format.format(data_name=data_name,
                         framework=framework,
                         phase=phase,
                         timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
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
    data_dir = cfg['DATA']['data_dir']

    gpu_ids = int(args.gpu_ids)
    num_workers = args.num_workers

    dataset_name = cfg['DATA']['name']
    n_channels = cfg['DATA']['img_channels']
    n_classes = cfg['DATA']['num_classes']
    model_name = cfg['MODEL']['name']
    batch_size = cfg['TRAINER']['batch_size']

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
    report_dir = os.path.join(save_dir, 'backbone', run_name)
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


    datasets = {
        'train': dataset_train,
        'val': Dataset_(
            data_name=cfg['DATA']['name'],
            data_dir=data_dir,
            train=False,
            split='val',
            crop_long_edge=cfg['PRE']['crop_long_edge'],
            resize_size=cfg['PRE']['resize_size'],
            resizer=cfg['PRE']['pre_resizer'],
            random_flip=cfg['PRE']['apply_rflip'],
            normalize=cfg['PRE']['normalize'],
            cfgs=cfg
        ),
        'test': Dataset_(
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
    }
    idx_to_class = datasets['test'].data.info['label']
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    data_loaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['TRAINER']['batch_size'], shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['TRAINER']['batch_size'], shuffle=False, num_workers=num_workers)
    }

    # Model.
    print('==> Building and training model...')
    model = ResNet50(in_channels=n_channels, num_classes=n_classes)

    model = model.to(device)

    # Loss function.
    criterion = nn.CrossEntropyLoss().to(device)
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=cfg['TRAINER']['optimizer']['lr'])
    # LR Scheduler.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['TRAINER']['scheduler']['mode'], patience=cfg['TRAINER']['scheduler']['patience'])

    # Train model.
    results = collections.defaultdict(lambda: [])
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
    )

    # Plot Training.
    util_cnn.plot_training(history=history, plot_training_dir=report_dir)

    # Test model.
    test_results = util_cnn.evaluate(dataset_name=dataset_name, model=model, data_loader=data_loaders['test'], device=device)

    # Update report.
    results["ACC"].append(test_results['all'])
    for c in classes:
        results["ACC %s" % str(c)].append(test_results[c])

    # Save Results
    report_file = os.path.join(report_dir, 'results.xlsx')
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    results_frame.to_excel(report_file, index=False)

    print('May the force be with you!')

