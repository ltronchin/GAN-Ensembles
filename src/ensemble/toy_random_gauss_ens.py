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

def plot_all(outdir, outname, real, synth, mu_real, mu_synth):
    # Google Material Design palette colors
    google_blue = '#4285F4'
    google_red = '#EA4335'

    # Visualize all.
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(real[:, 0], real[:, 1], alpha=0.5, color=google_blue, edgecolors='none', label='Real', marker='o', s=70)
    plt.scatter(synth[:, 0], synth[:, 1], alpha=0.5, color=google_red, edgecolors='none',  label='Synthetic', marker='*', s=70)

    # Mark the centers of each distribution with 'X'
    for c in mu_real:
        plt.scatter(c[0], c[1], color=google_blue, edgecolors='black', marker='o', s=100)

    for c in mu_synth:
        plt.scatter(c[0], c[1], color=google_red, edgecolors='black', marker='*', s=100)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, outname), dpi=300)
    #plt.show()

if __name__ == '__main__':

    # Define the space setting a limit for mux and muy.
    num_real_gauss = 10
    num_synth_gauss = 30
    num_samples = 100
    min_mux = 4
    min_muy = 4
    max_mux = 12
    max_muy = 12
    cov_real_max = 0.5
    cov_synth_list = [0.5] # [1.5, 0.5, 1.0]
    n_classes_list = [3] # [5, 4, 3, 2]
    # Optuna.
    n_trials = 300
    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    seed_data_list = [72, 103, 33, 101]
    seed_class_list = [33]
    pairwise_intra = True

    history_seed = pd.DataFrame(columns=['seed_data', 'seed_class', 'n_classes', 'cov_synth_max', 'n_obj', 'obj_name', 'aggregation', 'exp_name', 'n_gan_ens', 'accuracy', 'precision', 'specificity', 'recall', 'f1-score', 'g-mean','rec_modes', 'high_quality_samples'])

    opts_data = list(product(seed_data_list, seed_class_list, cov_synth_list, n_classes_list))
    class_choice = 'kmeans' # 'random' or 'kmeans'
    outdir = f'./reports/toy_gaussians-class_choice__{class_choice}-pairwise_intra__{pairwise_intra}'

    #for seed_data in seed_data_list:
    for seed_data, seed_class, cov_synth_max, n_classes in opts_data:

        report_dir = os.path.join(outdir, f'toy-seed_data__{seed_data}-seed_class__{seed_class}-num_real__{num_real_gauss}-cov_real_max__{cov_real_max}-cov_synth_max__{cov_synth_max}-num_synth__{num_synth_gauss}-n_samples__{num_samples}-n_classes__{n_classes}')
        util_path.create_dir(report_dir)

        # Generate data.
        data_real_list, data_real, mu_real, cov_real = util_toy.generate_gaussian_data(num_real_gauss, num_samples, max_covx=cov_real_max, max_covy=cov_real_max, min_mux=min_mux, min_muy=min_muy, max_mux=max_mux, max_muy=max_muy, seed=seed_data)
        data_synthetic_list, data_synthetic, mu_synth, cov_synth = util_toy.generate_gaussian_data(num_synth_gauss, num_samples, max_covx=cov_synth_max, max_covy=cov_synth_max, min_mux=min_mux, min_muy=min_muy, max_mux=max_mux, max_muy=max_muy, seed=seed_data+1)
        plot_all(outdir=report_dir, outname='guassians_real_all', real=data_real, synth=data_synthetic, mu_real=mu_real, mu_synth=mu_synth)

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

        labels_real = [util_toy.closest_point(mu, random_class) for mu in mu_real]
        labels_synth = [util_toy.closest_point(mu, random_class) for mu in mu_synth]

        # Visualize the space according to the label.
        fig = plt.figure(figsize=(10, 8))
        if n_classes == 2:
            colors = ['green', 'orange']
        elif n_classes == 3:
            colors = ['green', 'orange', 'purple']
        elif n_classes == 4:
            colors = ['green', 'orange', 'purple', 'brown']
        elif n_classes == 5:
            colors = ['green', 'orange', 'purple', 'brown', 'pink']
        else:
            raise NotImplementedError
        for idx, data in enumerate(data_real_list):
            plt.scatter(data[:, 0], data[:, 1],  alpha=0.5, color=colors[labels_real[idx]], edgecolors='none', marker='o', s=70)
        for idx, data in enumerate(data_synthetic_list):
            plt.scatter(data[:, 0], data[:, 1], alpha=0.5, color=colors[labels_synth[idx]],  edgecolors='none', marker='*', s=70)
            # Mark the centers of each distribution with 'X'
        for idx, center in enumerate(mu_real):
            plt.scatter(center[0], center[1], color=colors[labels_real[idx]], edgecolors='black', marker='o', s=100)
        for idx, center in enumerate(mu_synth):
            plt.scatter(center[0], center[1], color=colors[labels_synth[idx]], edgecolors='black', marker='*', s=100)
        # Plot the random points.
        class_idx = 0
        for point, color in zip(random_class, colors):
            plt.scatter(point[0], point[1], color=color, s=100, marker='X', edgecolors='black', label=f'Class {class_idx}')
            class_idx += 1
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        fig.savefig(os.path.join(report_dir, 'gaussians_class.png'), dpi=300)
        #plt.show()

        # Create dataset.
        X_real, y_real, y_gauss = util_toy.create_dataset(data_real_list, labels_real)
        X_naive, y_naive, _ = util_toy.create_dataset(data_synthetic_list, labels_synth)

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
            history = pd.DataFrame(columns=['seed_data', 'seed_class', 'n_classes', 'cov_synth_max', 'n_obj', 'obj_name', 'aggregation', 'exp_name', 'n_gan_ens', 'accuracy', 'precision', 'recall', 'specificity', 'f1-score', 'g-mean', 'rec_modes', 'high_quality_samples'])

        # Real.
        clf = DecisionTreeClassifier()
        clf.fit(X_train_real, y_train_real)
        y_pred = clf.predict(X_test_real)
        clf_acc, clf_prec, clf_rec, clf_spec, clf_f1, clf_gmean = util_toy.compute_metrics(y_test_real, y_pred, aggregation='mean')
        history = history.append({
            'seed_data': seed_data,
            'seed_class': seed_class,
            'n_classes': n_classes,
            'cov_synth_max': cov_synth_max,
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
        rec_modes = util_toy.recovered_mode(data=X_naive, mu_ref=mu_real, mu_synth=mu_synth, thres=3 *np.nanmax(np.mean(cov_real, axis=0)))
        high_quality_samples = util_toy.frac_points(data=X_naive, mu_ref=mu_real, thres=3 *np.nanmax(np.mean(cov_real, axis=0)))
        history = history.append({
            'seed_data': seed_data,
            'seed_class': seed_class,
            'n_classes': n_classes,
            'cov_synth_max': cov_synth_max,
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

        # Ensemble search.
        #name_obj_list =   ['pr_inter', 'pr_intra', 'dc_inter', 'dc_intra', 'fid_inter', 'fid_intra', 'pr_inter__pr_intra', 'dc_inter__dc_intra', 'fid_inter__fid_intra'] # ['pr_inter', 'pr_intra', 'pr_inter__pr_intra', 'dc_inter', 'dc_intra', 'dc_inter__dc_intra', 'fid_inter', 'fid_intra', 'fid_inter__fid_intra']
        #aggregation = ['inter', 'intra', 'ratio', 'NA']
        name_obj_list =   ['pr_inter', 'dc_inter', 'fid_inter', 'pr_intra','dc_intra', 'fid_intra', 'pr_inter__pr_intra', 'dc_inter__dc_intra', 'fid_inter__fid_intra'] # ['pr_inter', 'pr_intra', 'pr_inter__pr_intra', 'dc_inter', 'dc_intra', 'dc_inter__dc_intra', 'fid_inter', 'fid_intra', 'fid_inter__fid_intra']
        aggregation =  ['n_gans'] # ['inter', 'n_gans', 'NA']

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
            for i in name_obj.split('__'):
                if i == 'pr_inter':
                    directions.append('maximize')
                elif i == 'pr_intra':
                    directions.append('minimize')
                elif i == 'dc_inter':
                    directions.append('maximize')
                elif i == 'dc_intra':
                    directions.append('minimize')
                elif i == 'fid_inter':
                    directions.append('minimize')
                elif i == 'fid_intra':
                    directions.append('maximize')
                else:
                    raise NotImplementedError

            ensemble_dir = os.path.join(report_dir, f'n_obj__{n_obj}-name_obj__{name_obj}-aggregation__{aggregation}-n_trials__{n_trials}')
            util_path.create_dir(ensemble_dir)

            study = optuna.create_study(directions=directions, sampler=optuna.samplers.TPESampler(multivariate=True))

            func = lambda trial: util_toy.obj_optuna(
                trial=trial,
                samples_real=data_real_list,
                samples_naive_list=data_synthetic_list,
                n_obj=n_obj,
                name_obj=name_obj,
                pairwise=pairwise_intra
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

                # Save to disk.
                best_trials_df.reset_index(inplace=True, drop=True)
                best_trials_df.to_excel(os.path.join(ensemble_dir, 'optuna_study_best.xlsx'), index=False)

                if aggregation == 'inter':
                    best_trials_df['cost'] = copy.deepcopy(best_trials_df['values_0'])
                elif aggregation == 'intra':
                    best_trials_df['cost'] = copy.deepcopy(best_trials_df['values_1'])
                elif aggregation == 'ratio':
                    best_trials_df['cost'] = best_trials_df['values_0'] / best_trials_df['values_1']
                elif aggregation == 'n_gans':
                    temp = best_trials_df.drop(columns=['number', 'values_0', 'values_1', 'state'])
                    best_trials_df['cost'] = copy.deepcopy(temp.sum(axis=1))
                if aggregation == 'n_gans':
                    best_trials_cost = best_trials_df[best_trials_df['cost'] == best_trials_df['cost'].min()].iloc[0]
                else:
                    if 'pr' in name_obj or 'dc' in name_obj:
                        best_trials_cost = best_trials_df[best_trials_df['cost'] == best_trials_df['cost'].max()].iloc[0]
                    elif 'fid' in name_obj:
                        best_trials_cost = best_trials_df[best_trials_df['cost'] == best_trials_df['cost'].min()].iloc[0]
                    else:
                        raise NotImplementedError

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
            mu_ensemble = [mu_synth[i] for i in range(len(mu_synth)) if sel_idx[i] == 1]
            cov_ensemble = [cov_synth[i] for i in range(len(cov_synth)) if sel_idx[i] == 1]
            # Label the data.
            labels_ensemble = [util_toy.closest_point(mu, random_class) for mu in mu_ensemble]

            # Plot only the ensemble vs real.
            plot_all(outdir=ensemble_dir, outname='guassians_real_ensemble', real=data_real, synth=data_ensemble, mu_real=mu_real, mu_synth=mu_ensemble)

            # Create the training dataset.
            X_ens,y_ens, _ = util_toy.create_dataset(data_ensemble_list, labels_ensemble)

            # Ensemble.
            clf = DecisionTreeClassifier()
            clf.fit(X_ens, y_ens)
            y_pred = clf.predict(X_test_real)
            clf_acc, clf_prec, clf_rec, clf_spec, clf_f1, clf_gmean = util_toy.compute_metrics(y_test_real, y_pred, aggregation='mean')
            rec_modes = util_toy.recovered_mode(data=X_ens, mu_ref=mu_real, mu_synth=mu_ensemble, thres=3 * np.nanmax(np.mean(cov_real, axis=0)))
            high_quality_samples = util_toy.frac_points(data=X_ens, mu_ref=mu_real, thres=3 * np.nanmax(np.mean(cov_real, axis=0)))
            history = history.append({
                'seed_data': seed_data,
                'seed_class': seed_class,
                'n_classes': n_classes,
                'cov_synth_max': cov_synth_max,
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

        # Append all the results in history in history_seed.
        history_seed = pd.concat([history_seed, history], ignore_index=True)

    history_seed.to_excel(os.path.join(outdir, 'history_seed.xlsx'), index=False)

    print('May the force be with you.')