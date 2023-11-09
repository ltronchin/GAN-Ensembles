import sys
sys.path.extend([
    "./",
])
import numpy as np
import os
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import copy
import random

import pandas as pd
from itertools import product
from src.general_utils import util_path
from src.general_utils import util_toy

# For aspect ratio 4:3.
sns.set_context("paper")
sns.set_theme(style="ticks")

if __name__ == '__main__':

    report_dir = './reports/'
    exp_name = 'toy_gans_num_real_samples__100000-grid_size__5-step__2-std_dev__0.05'
    report_dir = os.path.join(report_dir, exp_name)
    util_path.create_dir(report_dir)

    # Parameters for the data
    grid_size = 5
    min_mux =4
    max_mux = min_mux + 2 * (grid_size - 1)
    min_muy = 4
    max_muy = min_muy + 2 * (grid_size - 1)
    num_real_tot = 10000
    num_synth_gan = 500
    mu_real = [(min_mux + 2 * i, min_muy + 2 * j) for i in range(grid_size) for j in range(grid_size)]

    std_real = 0.05
    n_trials = 300

    # Create real datasets.
    data_real_list, data_real = util_toy.generate_data(num_real_tot, mu_real, std_real)
    fig = plt.figure()
    plt.scatter(data_real[:, 0], data_real[:, 1], alpha=0.5, s=1, color='blue')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(report_dir, 'real_data.png'))
    #plt.show()

    # Load all the toy GANs on the disk.
    epoch_to_load = 'final'
    folders_list =  os.listdir(report_dir)
    data_synthetic_list = []
    for fname in folders_list:
        samples_dir = os.path.join(report_dir, fname)
        # Check if directory.
        if not os.path.isdir(samples_dir):
            continue
        if 'ensemble_search' in samples_dir:
            continue
         # Load final samples.
        samples = np.load(os.path.join(samples_dir, f'generated_samples_epoch_{epoch_to_load}.npy'))
        # Shuffle
        np.random.shuffle(samples)
        # Select the first num_synth_gan samples.
        samples = samples[:num_synth_gan]
        data_synthetic_list.append(samples)

    # Flatten the list.
    data_synthetic = np.vstack(data_synthetic_list)

    # Plot synthetic data.
    fig = plt.figure()
    plt.scatter(data_synthetic[:, 0], data_synthetic[:, 1], alpha=0.5, s=1, color='blue')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(report_dir, 'synthetic_data.png'))
    #plt.show()
    for idx, samples_synth in enumerate(data_synthetic_list):
        fig = plt.figure()
        plt.scatter(samples_synth[:, 0], samples_synth[:, 1], alpha=0.5, s=1, color='blue')
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(report_dir, f'synthetic_data_{idx}.png'))
        #plt.show()

    n_classes = 10
    seed_class = 42
    class_choice = 'random'  # 'random' or 'kmeans'
    os.path.join(report_dir, f'toy-seed_class__{seed_class}-n_classes__{n_classes}/')
    util_path.create_dir(report_dir)

    # Generate classes.
    if class_choice == 'random':
        np.random.seed(seed_class)
        random.seed(seed_class)
        random_class = np.array([[np.random.uniform(min_mux, max_mux), np.random.uniform(min_muy, max_muy)] for _ in range(n_classes)])
    elif class_choice == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_classes, random_state=seed_class, n_init='auto').fit(data_real)
        centroids = kmeans.cluster_centers_
        random_class = copy.deepcopy(centroids)
    else:
        raise NotImplementedError

    # Assign label to each point according to its vicinity to a class.
    # Real.
    distances_real = np.linalg.norm(data_real[:, np.newaxis] - random_class, axis=2)
    labels_real = np.argmin(distances_real, axis=1)

    # Synthetic.
    distances_synth = np.linalg.norm(data_synthetic[:, np.newaxis] - random_class, axis=2)
    labels_synth = np.argmin(distances_synth, axis=1)

    # Visualize the space according to the label.
    fig = plt.figure()
    plt.scatter(data_real[:, 0], data_real[:, 1], c=labels_real, marker='o', s=1, alpha=0.4, label='Points')
    plt.scatter(random_class[:, 0], random_class[:, 1], c=np.unique(labels_real), marker='*', s=50, label='Classes')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(report_dir, 'real_data_label.png'))
    #plt.show()

    fig = plt.figure()
    plt.scatter(data_synthetic[:, 0], data_synthetic[:, 1], c=labels_synth, marker='o', s=1, alpha=0.4, label='Points')
    plt.scatter(random_class[:, 0], random_class[:, 1], c=np.unique(labels_synth), marker='*', s=50, label='Classes')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(report_dir, 'synthetic_data_label.png'))
    #plt.show()

    # Create dataset.
    X_real = copy.deepcopy(data_real.reshape(-1, 2))
    y_real = copy.deepcopy(labels_real.reshape(-1, 1))
    X_naive = copy.deepcopy(data_synthetic)
    y_naive = copy.deepcopy(labels_synth.reshape(-1, 1))
    # Create y_gauss.
    y_gauss = []
    for center_idx, samples_real in enumerate(data_real_list):
        y_gauss.append(np.repeat(center_idx, len(samples_real)))
    y_gauss = np.hstack(y_gauss)
    y_gauss = y_gauss.reshape(-1, 1)

    # Concate y_real and y_gauss and save to xlsx.
    y_strat = np.hstack([y_real, y_gauss])
    df_strat = pd.DataFrame(y_strat)
    df_strat['strat'] = df_strat[0].astype(str) + '-' + df_strat[1].astype(str)

    # Split the training set.
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2,  stratify=df_strat, random_state=42, shuffle=True)

    # Create a dataframe for the results.
    if os.path.exists(os.path.join(report_dir, 'history.xlsx')):
        history = pd.read_excel(os.path.join(report_dir, 'history.xlsx'))
    else:
        history = pd.DataFrame(columns=['seed_class', 'n_classes', 'n_obj', 'obj_name', 'aggregation', 'exp_name', 'n_gan_ens', 'accuracy', 'precision', 'recall', 'specificity', 'f1-score', 'g-mean',  'rec_modes', 'high_quality_samples'])

    # Real.
    clf = DecisionTreeClassifier()
    clf.fit(X_train_real, y_train_real)
    y_pred = clf.predict(X_test_real)
    clf_acc, clf_prec, clf_rec, clf_spec, clf_f1, clf_gmean = compute_metrics(y_test_real, y_pred, aggregation='mean')
    history = history.append({
        'seed_class': seed_class,
        'n_classes': n_classes,
        'n_obj': 'NA',
        'obj_name': 'NA',
        'aggregation': 'NA',
        'exp_name': 'baseline',
        'n_gan_ens': 'NA',
        'accuracy': clf_acc,
        'precision': clf_prec,
        'recall': clf_rec,
        'specificity': clf_spec,
        'f1-score': clf_f1,
        'g-mean': clf_gmean,
        'rec_modes': 'NA',
        'high_quality_samples': 'NA'
    }, ignore_index=True)

    # Naive.
    clf = DecisionTreeClassifier()
    clf.fit(X_naive, y_naive)
    y_pred = clf.predict(X_test_real)
    clf_acc, clf_prec, clf_rec, clf_spec, clf_f1, clf_gmean = util_toy.compute_metrics(y_test_real, y_pred, aggregation='mean')
    rec_modes = util_toy.recovered_mode(data=X_naive, mu_ref=mu_real, thres= 3 * std_real)
    high_quality_samples = util_toy.frac_points(data=X_naive, mu_ref=mu_real, thres=3 * std_real)
    history = history.append({
        'seed_class': seed_class,
        'n_classes': n_classes,
        'n_obj': 'NA',
        'obj_name': 'NA',
        'aggregation': 'NA',
        'exp_name': 'naive',
        'n_gan_ens': len(data_synthetic_list),
        'accuracy': clf_acc,
        'precision': clf_prec,
        'recall': clf_rec,
        'specificity': clf_spec,
        'f1-score': clf_f1,
        'g-mean': clf_gmean,
        'rec_modes': rec_modes,
        'high_quality_samples': high_quality_samples
    }, ignore_index=True)

    # Single Gaussians.
    mean_acc = []
    mean_precision = []
    mean_recall = []
    mean_specificity = []
    mean_f1_score = []
    mean_gmean = []

    mean_rec_modes = []
    mean_high_quality_samples = []
    for idx in range(len(data_synthetic_list)):
        X = X_naive[idx * num_synth_gan: (idx + 1) * num_synth_gan]
        y = y_naive[idx * num_synth_gan: (idx + 1) * num_synth_gan]
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        y_pred = clf.predict(X_test_real)
        clf_acc, clf_prec, clf_rec, clf_spec, clf_f1, clf_gmean = util_toy.compute_metrics(y_test_real, y_pred, aggregation='mean')
        mean_acc.append(clf_acc)
        mean_precision.append(clf_prec)
        mean_recall.append(clf_rec)
        mean_specificity.append(clf_spec)
        mean_f1_score.append(clf_f1)
        mean_gmean.append(clf_gmean)
        mean_rec_modes.append(util_toy.recovered_mode(data=X, mu_ref=mu_real, thres=3 * std_real))
        mean_high_quality_samples.append(util_toy.frac_points(data=X, mu_ref=mu_real, thres=3 * std_real))

    history = history.append({
        'seed_class': seed_class,
        'n_classes': n_classes,
        'n_obj': 'NA',
        'obj_name': 'NA',
        'aggregation': 'NA',
        'exp_name': 'mean gan',
        'n_gan_ens': 'NA',
        'accuracy': np.mean(mean_acc),
        'precision': np.mean(mean_precision),
        'recall': np.mean(mean_recall),
        'specificity': np.mean(mean_specificity),
        'f1-score': np.mean(mean_f1_score),
        'g-mean': np.mean(mean_gmean),
        'rec_modes': np.mean(mean_rec_modes),
        'high_quality_samples': np.mean(mean_high_quality_samples)

    }, ignore_index=True)

    # Ensemble search.
    name_obj_list =   ['pr_inter', 'pr_intra', 'dc_inter', 'dc_intra', 'fid_inter', 'fid_intra', 'pr_inter__pr_intra', 'dc_inter__dc_intra', 'fid_inter__fid_intra'] # ['pr_inter', 'pr_intra', 'pr_inter__pr_intra', 'dc_inter', 'dc_intra', 'dc_inter__dc_intra', 'fid_inter', 'fid_intra', 'fid_inter__fid_intra']
    aggregation = ['inter', 'intra', 'ratio', 'NA']

    opts = list(product(name_obj_list, aggregation))

    for name_obj, aggregation in opts:

        n_obj = len(name_obj.split('__'))
        if n_obj > 1:
            obj_first_position = name_obj.split('___')[0]
            assert 'inter' in obj_first_position
            if aggregation == 'NA':
                continue
        else:
            if aggregation != 'NA':
                continue

        directions = []
        for i in range(n_obj):
            if 'pr_inter' in name_obj:
                directions.append('maximize')
            elif 'pr_intra' in name_obj:
                directions.append('minimize')
            elif 'dc_inter' in name_obj:
                directions.append('maximize')
            elif 'dc_intra' in name_obj:
                directions.append('minimize')
            elif 'fid_inter' in name_obj:
                directions.append('minimize')
            elif 'fid_intra' in name_obj:
                directions.append('maximize')
            elif 'size' in name_obj:
                directions.append('minimize')
            else:
                raise NotImplementedError

        ensemble_dir = os.path.join(report_dir,f'ensemble_search-n_obj__{n_obj}-name_obj__{name_obj}-aggregation__{aggregation}-n_trials__{n_trials}')
        util_path.create_dir(ensemble_dir)

        study = optuna.create_study(directions=directions, sampler=optuna.samplers.TPESampler(multivariate=True))

        func = lambda trial: util_toy.obj_optuna(
            trial=trial,
            samples_real=data_real_list,
            samples_naive_list=data_synthetic_list,
            n_obj=n_obj,
            name_obj=name_obj,
        )
        study.optimize(func, n_trials=n_trials)
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

        # Drop not complete trials and save to disk.
        df = df[df['state'] == "COMPLETE"]
        df.reset_index(inplace=True, drop=True)
        df.to_excel(os.path.join(ensemble_dir, 'optuna_study.xlsx'), index=False)

        if n_obj==1:
            # Best trials analysis.
            if name_obj == 'pr_inter':
                best_trials = max(study.best_trials, key=lambda t: t.values[0])
            elif name_obj == 'pr_intra':
                best_trials = min(study.best_trials, key=lambda t: t.values[0])
            elif name_obj == 'dc_inter':
                best_trials = max(study.best_trials, key=lambda t: t.values[0])
            elif name_obj == 'dc_intra':
                best_trials = min(study.best_trials, key=lambda t: t.values[0])
            elif name_obj == 'fid_inter':
                best_trials = min(study.best_trials, key=lambda t: t.values[0])
            elif name_obj == 'fid_intra':
                best_trials = max(study.best_trials, key=lambda t: t.values[0])
            else:
                raise NotImplementedError

            with open(os.path.join(ensemble_dir, f"best_trials.txt"), 'w') as f:
                f.write(f"\nNumber: {best_trials.number}")
                f.write(f"\nParams: {best_trials.params}")
                f.write(f"\nValues: {best_trials.values}")

            best_trials_gans = [x for x in best_trials.params if best_trials.params[x] == 1]

            best_trials_gans_indexes = sorted([int(x.split('G')[-1]) for x in best_trials_gans])
            best_trials_gans = [f'G{x}' for x in best_trials_gans_indexes]
            sel_idx = [0] * len(data_synthetic_list)
            for idx in best_trials_gans_indexes:
                sel_idx[idx] = 1

        elif n_obj > 1:
            best_trials = copy.deepcopy(study.best_trials)
            # Selet only best trials.
            best_trials_number = [trial.number for trial in best_trials]
            best_trials_df = df[df['number'].isin(best_trials_number)]

            if aggregation == 'inter':
                best_trials_df['cost'] = copy.deepcopy(best_trials_df['values_0'])
            elif aggregation == 'intra':
                best_trials_df['cost'] = copy.deepcopy(best_trials_df['values_1'])
            elif aggregation == 'ratio':
                best_trials_df['cost'] = best_trials_df['values_0'] / best_trials_df['values_1']

            if 'pr' in name_obj or 'dc' in name_obj:
                best_trials_cost = best_trials_df[best_trials_df['cost'] == best_trials_df['cost'].max()].iloc[0]
            elif 'fid' in name_obj:
                best_trials_cost = best_trials_df[best_trials_df['cost'] == best_trials_df['cost'].min()].iloc[0]
            else:
                raise NotImplementedError

            # Save to disk.
            best_trials_df.reset_index(inplace=True, drop=True)
            best_trials_df.to_excel(os.path.join(ensemble_dir, 'optuna_study_best.xlsx'), index=False)

            best_trials_gans = best_trials_cost[best_trials_cost == 1]
            best_trials_gans = list(best_trials_gans.index)
            # Remove 'number' from the list.
            best_trials_gans = [x for x in best_trials_gans if x != 'number']
            with open(os.path.join(ensemble_dir, f"best_gans.txt"), 'w') as f:
                f.write(f"\n{best_trials_gans}")

            best_trials_gans_indexes = sorted([int(x.split('G')[-1]) for x in best_trials_gans])
            best_trials_gans = [f'G{x}' for x in best_trials_gans_indexes]
            sel_idx = [0] * len(data_synthetic_list)
            for idx in best_trials_gans_indexes:
                sel_idx[idx] = 1
        else:
            raise NotImplementedError

        # Create ensemble dataset.
        data_ensemble_list = [data_synthetic_list[i] for i in range(len(data_synthetic_list)) if sel_idx[i] == 1]
        data_ensemble = np.vstack(data_ensemble_list)

        # Label the data.
        distances_ensemble = np.linalg.norm(data_ensemble[:, np.newaxis] - random_class, axis=2)
        labels_ensemble = np.argmin(distances_ensemble, axis=1)

        # Create the training dataset.
        X_ens = copy.deepcopy(data_ensemble.reshape(-1, 2))
        y_ens = copy.deepcopy(labels_ensemble.reshape(-1, 1))

        # Ensemble.
        clf = DecisionTreeClassifier()
        clf.fit(X_ens, y_ens)
        y_pred = clf.predict(X_test_real)
        clf_acc, clf_prec, clf_rec, clf_spec, clf_f1, clf_gmean = util_toy.compute_metrics(y_test_real, y_pred, aggregation='mean')
        rec_modes = util_toy.recovered_mode(data=X_ens, mu_ref=mu_real, thres=3 * std_real)
        high_quality_samples = util_toy.frac_points(data=X_ens, mu_ref=mu_real, thres=3 * std_real)
        history = history.append({
            'seed_class': seed_class,
            'n_classes': n_classes,
            'n_obj': n_obj,
            'obj_name': name_obj,
            'aggregation': aggregation,
            'exp_name': 'ensemble',
            'n_gan_ens': len(data_ensemble_list),
            'accuracy': clf_acc,
            'precision': clf_prec,
            'recall': clf_rec,
            'specificity': clf_spec,
            'f1-score': clf_f1,
            'g-mean': clf_gmean,
            'rec_modes': rec_modes,
            'high_quality_samples': high_quality_samples
        }, ignore_index=True)

        # Save to the disk.
        history.to_excel(os.path.join(report_dir, 'history.xlsx'), index=False)

    print('May the force be with you.')