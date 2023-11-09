import copy
import sys
sys.path.extend([
    "./"
])
import numpy as np
import os
import optuna
import seaborn as sns
import hashlib
import uuid
import re
import argparse
import copy
from tqdm import tqdm
import pandas as pd
import datetime

from src.custom_metrics.features import FeatureStats
from src.custom_metrics.features import load_stats

from src.general_utils import util_path
from src.general_utils import util_general
from src.general_utils import util_ensemble
from src.general_utils import util_metric

# For aspect ratio 4:3.
sns.set_context("paper")
sns.set_theme(style="ticks")

def get_parser():

    parser = argparse.ArgumentParser(description='Ensemble search.')

    parser.add_argument('--source_dir', type=str, default='./reports/AIforCOVID', help='Directory name to fake samples.')
    parser.add_argument('--dataset_name', type=str, default='AIforCOVID',  choices=['retinamnist', 'pneumoniamnist', 'breastmnist'], help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # GANs params.
    #parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='MHGAN,SNGAN,StyleGAN2-D2DCE,ReACGAN-ADA,ReACGAN-ADC,ReACGAN-DiffAug,ACGAN-Mod,ReACGAN,BigGAN-DiffAug,BigGAN-Info,StyleGAN2-DiffAug,ACGAN-Mod-TAC,BigGAN,ReACGAN-TAC,BigGAN-ADA,StyleGAN2-Info,ACGAN-Mod-ADC,StyleGAN2-ADA,ReACGAN-Info,StyleGAN2,ContraGAN,SAGAN', help='List of GANs to enable in the ensemble') #ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA,StyleGAN2-DiffAug,StyleGAN2  # MHGAN,SNGAN,StyleGAN2-D2DCE,ReACGAN-ADA,ReACGAN-ADC,ReACGAN-DiffAug,ACGAN-Mod,ReACGAN,BigGAN-DiffAug,BigGAN-Info,StyleGAN2-DiffAug,ACGAN-Mod-TAC,BigGAN,ReACGAN-TAC,BigGAN-ADA,StyleGAN2-Info,ACGAN-Mod-ADC,StyleGAN2-ADA,ReACGAN-Info,StyleGAN2,ContraGAN,SAGAN
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='MHGAN,SNGAN,StyleGAN2-D2DCE,ReACGAN-ADA', help='List of GANs to enable in the ensemble') #ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA,StyleGAN2-DiffAug,StyleGAN2  # MHGAN,SNGAN,StyleGAN2-D2DCE,ReACGAN-ADA,ReACGAN-ADC,ReACGAN-DiffAug,ACGAN-Mod,ReACGAN,BigGAN-DiffAug,BigGAN-Info,StyleGAN2-DiffAug,ACGAN-Mod-TAC,BigGAN,ReACGAN-TAC,BigGAN-ADA,StyleGAN2-Info,ACGAN-Mod-ADC,StyleGAN2-ADA,ReACGAN-Info,StyleGAN2,ContraGAN,SAGAN
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='20000,40000',help='Iter or Iters to sample each GAN') #20000,40000 # 20000,40000,60000,80000,100000
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='ResNet50_torch',  choices=['InceptionV3_torch', 'ResNet50_torch', 'SwAV_torch', 'InceptionV3_torch__medical',  'InceptionV3_torch__truefake', 'ResNet50_torch__medical', 'ResNet50_torch__truefake'])
    parser.add_argument('--n_samples', type=int, default='664', help='Total number of images generated by the ensemble.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    # Ensemble search params.
    parser.add_argument('--name_obj', type=str, default='dc_inter', choices=['dc_inter', 'pr_inter', 'fid_inter', 'dc_inter__dc_intra', 'pr_inter__pr_intra', 'fid_inter__fid_intra'])
    parser.add_argument('--pairwise_intra', type=bool, default=True)
    #parser.add_argument('--pairwise_intra_df', type=bool, default=True)
    parser.add_argument('--n_trial', type=int, default=10)
    parser.add_argument('--dim_reduction', type=bool, default=False)
    parser.add_argument('--synth_samples_reduction', type=int, default=100000)
    # Downstream task params.
    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs_downstream_task/pneumoniamnist.yaml")
    parser.add_argument('--init_w', type=str, default='uniform', choices=['uniform', 'random', 'fid'],  help='Weight initialization')
    parser.add_argument('--n_train', type=int, default=20, help='Number of times the experiment is repeated.')
    parser.add_argument("--num_workers", type=int, default=8)

    return parser

if __name__ == '__main__':


    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # Directories.
    source_dir = args.source_dir
    dataset_name = args.dataset_name
    samples_dir = os.path.join(source_dir, 'samples')
    ensemble_dir = os.path.join(source_dir, 'ensemble')
    reports_dir = os.path.join(source_dir, 'downstream_task')
    cache_dir = os.path.join(source_dir, 'features')
    util_path.create_dir(reports_dir)

    # Parameters features.
    post_resizer = args.post_resizer
    eval_backbone = args.eval_backbone
    n_samples = args.n_samples
    split = args.split

    # Parameters ensemble.
    name_obj = args.name_obj
    n_obj = len(name_obj.split('__'))
    n_trial = args.n_trial
    pairwise_intra = args.pairwise_intra
    dim_reduction = args.dim_reduction
    synth_samples_reduction = args.synth_samples_reduction

    # Add to the filename the current time.
    filename = f'ensemble_search-n_trial__{n_trial}-name_obj__{name_obj}-pairwise_intra__{pairwise_intra}-dim_reduction__{dim_reduction}-synth_samples_reduction__{synth_samples_reduction}-{eval_backbone}-{post_resizer}-{n_samples}'
    now = datetime.datetime.now()
    #filename = f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_{filename}'

    ensemble_dir = os.path.join(ensemble_dir, filename)
    util_path.create_dir(ensemble_dir)

    directions = []
    for i in name_obj.split('__'):
        if i == 'pr_inter':
            directions.append('maximize')
        elif i == 'pr_intra':
            directions.append('minimize')
        elif i == 'dc_inter':
            directions.append('maximize')
        elif i == 'dc_intra':
            directions.append('minimize')
        elif i=='fid_inter':
            directions.append('minimize')
        elif i=='fid_intra':
            directions.append('maximize')
        else:
            raise NotImplementedError

    # Load real features.
    data_real_list = []
    stats_real = load_stats(cache_dir=os.path.join(cache_dir, f'real_{split}'), eval_backbone=eval_backbone, post_resizer=post_resizer, max_items=n_samples, capture_all=True, capture_mean_cov=True)
    feats_real = stats_real.get_all_torch()
    feats_real = feats_real.detach().cpu().numpy().astype(np.float64)
    data_real_list.append(feats_real)
    data_real = copy.deepcopy(feats_real)

    # Load all synthetic distributions -> naive.
    gan_models = args.gan_models
    gan_steps = args.gan_steps
    gan_aval = os.listdir(samples_dir)
    gan_folders = [os.path.join(samples_dir, x, f'fake__{split}') for x in gan_aval if any(f'{gan_model}-train-' in x for gan_model in gan_models)]
    gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]
    pattern = fr"(?<={dataset_name}-)(\w+(?:-\w+)*)(?=-{split})"
    data_synthetic_list = []
    search_space = []
    with tqdm(total=len(gan_folders)) as pbar:
        for gan_sel in gan_folders:
            stats_synth = load_stats(cache_dir=cache_dir + gan_sel.split('samples')[-1], eval_backbone=eval_backbone, post_resizer=post_resizer,  max_items=n_samples, capture_all=True, capture_mean_cov=True)
            feats_synth = stats_synth.get_all_torch()
            feats_synth = feats_synth.detach().cpu().numpy().astype(np.float64)
            data_synthetic_list.append(feats_synth)
            search_space.append( re.search(pattern, gan_sel.split('/')[-3]).group(1) + '__' + gan_sel.split('/')[-1].split('=')[-1])
            pbar.update(1)
    data_synth = np.vstack(data_synthetic_list)

    # Load the dataframe containing the binary FID.
    try:
        df_intra = pd.read_excel(os.path.join(cache_dir, f'intra-{eval_backbone}-{post_resizer}-{n_samples}.xlsx'))
    except FileNotFoundError:
        pass

    # Dimensionality reduction using UMAP.
    if dim_reduction:

        # Check if data_real and data_synth already exist on the disk.
        flag_real = os.path.exists(os.path.join(os.path.join(source_dir,'ensemble'), f'data_real_umap-{eval_backbone}-{post_resizer}-{n_samples}.npy'))
        flag_synth = os.path.exists(os.path.join(os.path.join(source_dir,'ensemble'), f'data_synth_umap-{eval_backbone}-{post_resizer}-{n_samples}.npy'))
        if flag_real and flag_synth:
            print('Loading data_real and data_synth from the disk.')
            data_real = np.load(os.path.join(os.path.join(source_dir,'ensemble'), f'data_real_umap-{eval_backbone}-{post_resizer}-{n_samples}.npy'))
            data_synth = np.load(os.path.join(os.path.join(source_dir,'ensemble'), f'data_synth_umap-{eval_backbone}-{post_resizer}-{n_samples}.npy'))
        else:
            print('Reducing the dimensionality.')
            # First find the number of dimension to retain all the Variance.
            from sklearn.decomposition import PCA
            pca = PCA(n_components=0.95)
            pca.fit(data_real)
            n_components = pca.n_components_
            print(f"Number of components to retain 95% of the variance: {n_components}")

            # Plot the variance.
            import matplotlib.pyplot as plt
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            # Plot a line definining the number of components to retain 95% of the variance.
            plt.axvline(x=n_components, color='r', linestyle='--')
            # Write in the legend the number of components to retain 95% of the variance.
            plt.legend(['Cumulative explained variance', f'Number of components to retain 95% of the variance: {n_components}'], loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(source_dir,'ensemble'), f'cumulative_explained_variance-{eval_backbone}-{post_resizer}-{n_samples}.png'), dpi=300, bbox_inches='tight')
            plt.show()

            # Apply UMAP.
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            data_real = reducer.fit_transform(data_real)
            data_synth = reducer.transform(data_synth)

            # Save to the disk.
            np.save(os.path.join(os.path.join(source_dir,'ensemble'), f'data_real_umap-{eval_backbone}-{post_resizer}-{n_samples}.npy'), data_real)
            np.save(os.path.join(os.path.join(source_dir,'ensemble'), f'data_synth_umap-{eval_backbone}-{post_resizer}-{n_samples}.npy'), data_synth)

        # Override data_real_list and data_synthetic_list.
        data_real_list = [data_real[i : i + n_samples] for i in range(0, n_samples * len(data_real_list), n_samples)]
        data_synthetic_list = [data_synth[i : i + n_samples] for i in  range(0, n_samples * len(data_synthetic_list), n_samples)]

    if synth_samples_reduction < n_samples:
        print('Reducing the number of synthetic samples.')
        # Reduce the number of samples for each distribution with random sampling.
        data_synthetic_list = [x[np.random.choice(x.shape[0], synth_samples_reduction, replace=False)] for x in data_synthetic_list]
        data_synth = np.vstack(data_synthetic_list)

    # Ensemble search.
    study = optuna.create_study(directions=directions, sampler=optuna.samplers.TPESampler(multivariate=True))

    func = lambda trial: util_ensemble.objective(
        trial=trial,
        samples_real=data_real,
        samples_naive_list=data_synthetic_list,
        search_space=search_space,
        name_obj=name_obj,
        pairwise=pairwise_intra,
        df_intra = df_intra
    )

    study.optimize(func, n_trials=n_trial)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    # Save the study ready to be uploaded.
    df = df[df['state'] == "COMPLETE"]
    df.reset_index(inplace=True, drop=True)
    df.to_excel(os.path.join(ensemble_dir, 'optuna_study.xlsx'), index=False)

    best_trials_idx, best_trials_name = util_ensemble.extract_best_trial(ensemble_dir=ensemble_dir, name_obj=name_obj, study=study)
    print('\n')
    print('Selected Ensemble.')
    print(best_trials_name)
    sel_idx = copy.deepcopy(best_trials_idx)
    sel_folders = [x for x in gan_folders if sel_idx[gan_folders.index(x)] == 1]

    # *
    # **
    # ***
    # ***
    # **
    # *
    # Downstream task.
    import yaml
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import collections

    from src.data_util import Dataset_
    from src.general_utils.util_data import EnsembleDataset
    from src.cnn_models.models import ResNet18, ResNet50
    from src.general_utils import util_cnn

    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    init_w = args.init_w
    n_train = args.n_train
    img_size = cfg['DATA']['img_size']

    gpu_ids = int(args.gpu_ids)
    num_workers = args.num_workers

    # Device.
    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')

    # Directories.
    filename = f'downstream_task_ensemble-n_trial__{n_trial}-name_obj__{name_obj}-pairwise_intra__{pairwise_intra}-dim_reduction__{dim_reduction}-synth_samples_reduction__{synth_samples_reduction}-{eval_backbone}-{post_resizer}-{n_samples}'
    #filename = f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_{filename}'
    reports_dir = os.path.join(reports_dir, filename)
    util_path.create_dir(reports_dir)
    with open(os.path.join(reports_dir, f"ensemble.txt"), 'w') as f:
        f.write(f"\nEnsemble: {best_trials_name}")
        f.write("\n")
        f.write(f"\nFolder ensemble: {sel_folders}")

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
        classes = cfg['DATA']['classes']
        class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        weight = [
            len(datasets_real['train']) / (len(classes) * len(datasets_real['train'].labels[datasets_real['train'].labels == class_to_idx[c]])) for c in classes
        ]
    elif dataset_name in ['pneumoniamnist', 'breastmnist', 'retinamnist']:
        idx_to_class = datasets_real['test'].data.info['label']
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        classes = [idx_to_class[i] for i in range(len(idx_to_class))]
        weight = [
            len(datasets_real['train'].data) / (len(classes) * len(datasets_real['train'].data.labels[datasets_real['train'].data.labels == class_to_idx[c]])) for c in classes
        ]
    else:
        raise NotImplementedError

    # Synthetic data.
    weights = util_ensemble.initialize_ensemble_weights(init_w=init_w, gan_list=sel_folders)
    dataset_synth = EnsembleDataset(folders=sel_folders, weights=weights)
    datasets = {
        'train': dataset_synth,
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
        print(f"Training {idx_train + 1}/{n_train}...")

        print('==> Building and training model...')
        if cfg['MODEL']['name'] == 'resnet18':
            model = ResNet18(input_channels=cfg['DATA']['img_channels'], num_classes=cfg['DATA']['num_classes'])
        elif cfg['MODEL']['name'] == 'resnet50':
            model = ResNet50(input_channels=cfg['DATA']['img_channels'], num_classes=cfg['DATA']['num_classes'])
        else:
            raise NotImplementedError
        model = model.to(device)

        # Loss function.
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg['TRAINER']['optimizer']['lr'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['TRAINER']['scheduler']['mode'],  patience=cfg['TRAINER']['scheduler']['patience'])

        model, history = util_cnn.train_model(
            model=model,
            data_loaders=data_loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=cfg['TRAINER']['max_epochs'],
            early_stopping=cfg['TRAINER']['early_stopping'],
            warmup_epoch=cfg['TRAINER']['warmup_epoch'],
            model_dir=reports_dir,
            device=device,
            n_samples=n_samples,
            to_disk=False
        )

        # Plot Training.
        util_cnn.plot_training(history=history, plot_training_dir=reports_dir, plot_name_loss=f'Loss_training_{idx_train}', plot_name_acc=f'Acc_training_{idx_train}')

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
        reports_file = os.path.join(reports_dir, f'results_training_{idx_train}.xlsx')
        results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
        results_frame.to_excel(reports_file, index=False)

    # Compute mean and std.
    results_frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    results_frame.loc['mean'] = results_frame.mean()
    results_frame.loc['std'] = results_frame.std()

    # Save Results
    reports_file = os.path.join(reports_dir, 'results.xlsx')
    results_frame.to_excel(reports_file, index=True)

    print("May the force be with you.")