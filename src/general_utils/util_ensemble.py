import numpy as np
from itertools import combinations
from tqdm import tqdm
import copy
import optuna
import os
import pandas as pd

from src.general_utils import util_general
from src.general_utils import util_metric

#eps = np.finfo(float).eps
eps = 1e-6
def divide_samples(num_sample, num_class):
    return [(num_sample // num_class) + (i < num_sample % num_class) for i in range(num_class)]


def gan_list_epochs(gan_list, gan_epochs):
    x = []
    for gan in gan_list:
        for epochs in gan_epochs:
            x.append(f"{gan}_{epochs}")

    return x

def extract_best_trial(ensemble_dir, name_obj, study):
    if name_obj == 'pr_inter' or name_obj == 'pr_inter__pr_intra':
        best_trials = max(study.best_trials, key=lambda t: t.values[0])
    elif name_obj == 'pr_intra':
        best_trials = min(study.best_trials, key=lambda t: t.values[0])
    elif name_obj == 'dc_inter' or name_obj == 'dc_inter__dc_intra':
        best_trials = max(study.best_trials, key=lambda t: t.values[0])
    elif name_obj == 'dc_intra':
        best_trials = min(study.best_trials, key=lambda t: t.values[0])
    elif name_obj == 'fid_inter' or name_obj == 'fid_inter__fid_intra':
        best_trials = min(study.best_trials, key=lambda t: t.values[0])
    elif name_obj == 'fid_intra':
        best_trials = max(study.best_trials, key=lambda t: t.values[0])
    else:
        raise NotImplementedError

    # Convert best_trials to Dataframe and save.
    best_trials_df = study.trials_dataframe()
    best_trials_df.to_excel(os.path.join(ensemble_dir, f"best_trials.xlsx"), index=False)

    with open(os.path.join(ensemble_dir, f"best_trials.txt"), 'w') as f:
        f.write(f"\nNumber: {best_trials.number}")
        f.write(f"\nParams: {[x for x in best_trials.params if best_trials.params[x] == 1]}")
        f.write(f"\nParams: {list(best_trials.params.values())}")
        f.write(f"\nValues: {best_trials.values}")

    best_trials_name = [x for x in best_trials.params if best_trials.params[x] == 1]
    best_trials_idx = list(best_trials.params.values())

    return best_trials_idx, best_trials_name

def initialize_ensemble_weights(init_w, gan_list):

    if init_w == 'uniform':
        weights = np.array([1] * len(gan_list))
        weights = weights / weights.sum()
    elif init_w == 'random':
        weights = np.random.uniform(0, 1, len(gan_list))
        weights = softmax(weights)
    elif init_w == 'fid':
        raise NotImplementedError
    else:
        raise ValueError(init_w)

    return weights

def softmax(arr):
    """
    Calculate the softmax of a numpy array.
    """
    exp_arr = np.exp(arr - np.max(arr))
    return exp_arr / np.sum(exp_arr, axis=0)

def get_inter_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs):

    df_real = kwargs['data_real']
    df_real = df_real[
        (df_real['gan0'] + "_" + df_real['step0'].astype(str)).isin(sel_gans)
    ]
    df_real.reset_index(drop=True, inplace=True)
    n_inter = len(sel_gans)
    assert n_inter == len(df_real)

    values = list(df_real[fitness_name])

    if summary_flag == 'mean':
        fitness_value = np.mean(values)
    elif summary_flag == 'std':
        fitness_value = np.std(values)
    elif summary_flag == 'sem':
        fitness_value = np.std(values) / np.sqrt(n_inter)
    else:
        raise ValueError(f'Unknown summary flag: {summary_flag}')

    return fitness_value

def get_intra_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs):

    df_synth = kwargs['data_synth']
    df_synth = df_synth[
        (df_synth['gan0'] + "_" + df_synth['step0'].astype(str)).isin(sel_gans) &
        (df_synth['gan1'] + "_" + df_synth['step1'].astype(str)).isin(sel_gans)
        ]
    df_synth.reset_index(drop=True, inplace=True)
    n_intra =  len([x for x in combinations(sel_gans, 2)])
    assert n_intra == len(df_synth)

    values = list(df_synth[fitness_name])

    if summary_flag == 'mean':
        fitness_value = np.mean(values)
    elif summary_flag == 'std':
        fitness_value = np.std(values)
    elif summary_flag == 'sem':
        fitness_value = np.std(values) / np.sqrt(n_intra)
    else:
        raise ValueError(f'Unknown summary flag: {summary_flag}')

    return fitness_value

def objective_pairwise(trial, gan_models, gan_steps, obj_name, fitness_name, summary_flag, **kwargs):

    search_space = [f"{x}_{y}" for x in gan_models for y in gan_steps]
    sel_idx = [trial.suggest_int(x, 0, 1) for x in search_space]
    sel_gans = [x for i, x in enumerate(search_space) if sel_idx[i] == 1]
    n_search_space = len(sel_gans)

    if len(sel_gans) < 2:
        raise optuna.TrialPruned("Number of selected GANs is less than 2.")

    if obj_name == 'inter__intra':
        intra_metric = get_intra_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        inter_metric = get_inter_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        return inter_metric, intra_metric
    elif obj_name == 'intra':
        intra_metric = get_intra_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        return intra_metric
    elif obj_name == 'inter':
        inter_metric = get_inter_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        return inter_metric
    else:
        raise NotImplementedError

def backward_objective_pairwise(sel_gans, obj_name, fitness_name, summary_flag, **kwargs):

    if obj_name == 'inter__intra':
        intra_metric = get_intra_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        inter_metric = get_inter_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)

        obj = inter_metric / intra_metric
        return obj

    elif obj_name == 'intra':
        intra_metric = get_intra_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        return intra_metric
    elif obj_name == 'inter':
        inter_metric = get_inter_metric_pairwise(sel_gans, fitness_name, summary_flag, **kwargs)
        return inter_metric
    else:
        raise NotImplementedError

def compute_fid(real, ens):
    mu_ens, cov_ens = util_metric.compute_statistics(ens)
    mu_real, cov_real = util_metric.compute_statistics(real)
    fid = util_metric.FID(mu_ens, cov_ens, mu_real, cov_real)
    return fid

def compute_pr(real, ens):
    precision, recall = util_metric.compute_pr(real, ens, nearest_k=5)
    pr = 2 * (precision * recall) / ((precision + recall) + eps)
    return pr

def compute_dc(real, ens):
    density, coverage = util_metric.compute_dc(real, ens, nearest_k=5)
    dc = 2 * (density * coverage) / ((density + coverage) + eps)
    return dc

def compute_pairwise(samples_list, metric_function, metric_name, sel_ens, sel_gans, **kwargs):
    metric_list = []
    try:
        df_intra = kwargs['df_intra']
        df_intra = df_intra[
            (df_intra['gan0'] + "__" + df_intra['step0'].astype(str)).isin(sel_gans) &
            (df_intra['gan1'] + "__" + df_intra['step1'].astype(str)).isin(sel_gans)
            ]
        df_intra.reset_index(drop=True, inplace=True)
        n_intra = len([x for x in combinations(sel_gans, 2)])
        assert n_intra == len(df_intra)
        metric_list = list(df_intra[f'{metric_name}'])
    except KeyError:
        idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]
        for idxs_comb in combinations(idxs_sel_ens, 2):
            ens1 = samples_list[idxs_comb[0]]
            ens2 = samples_list[idxs_comb[1]]
            metric_list.append(metric_function(ens2, ens1))
    return np.mean(metric_list)


def objective(trial, samples_real, samples_naive_list, search_space, name_obj='fid_inter', pairwise=True, **kwargs):
    """Objective function to minimize for ensemble S."""

    max_ens = len(samples_naive_list)
    sel_ens = [trial.suggest_int(f'{search_space[x]}', 0, 1) for x in range(len(samples_naive_list))]
    sel_gans = [x for i, x in enumerate(search_space) if sel_ens[i] == 1]

    n_ens = np.sum(sel_ens)
    if n_ens < 2:
        raise optuna.TrialPruned("Number of selected GANs is less than 2.")
    samples_ens_list = [samples_naive_list[j] for j in range(max_ens) if sel_ens[j] == 1]
    samples_ens = np.vstack(samples_ens_list)

    if name_obj == 'fid_inter':
        return compute_fid(samples_real, samples_ens)

    elif name_obj == 'fid_intra':
        if not pairwise:
            return compute_fid(np.vstack(samples_naive_list), samples_ens)
        else:
            return compute_pairwise(samples_naive_list, compute_fid, 'fid', sel_ens, sel_gans, **kwargs)

    elif name_obj == 'pr_inter':
        return compute_pr(samples_real, samples_ens)

    elif name_obj == 'pr_intra':
        if not pairwise:
            return compute_pr(np.vstack(samples_naive_list), samples_ens)
        else:
            return compute_pairwise(samples_naive_list, compute_pr, 'f1_prc_rec', sel_ens, sel_gans, **kwargs)

    elif name_obj == 'dc_inter':
        return compute_dc(samples_real, samples_ens)

    elif name_obj == 'dc_intra':
        if not pairwise:
            return compute_dc(np.vstack(samples_naive_list), samples_ens)
        else:
            return compute_pairwise(samples_naive_list, compute_dc, 'f1_dns_cvg', sel_ens, sel_gans, **kwargs)

    elif name_obj == 'fid_inter__fid_intra':
        fid_inter = compute_fid(samples_real, samples_ens)
        if not pairwise:
            fid_intra = compute_fid(np.vstack(samples_naive_list), samples_ens)
        else:
            fid_intra = compute_pairwise(samples_naive_list, compute_fid, 'fid', sel_ens, sel_gans, **kwargs)
        return fid_inter, fid_intra

    elif name_obj == 'pr_inter__pr_intra':
        pr_inter = compute_pr(samples_real, samples_ens)
        if not pairwise:
            pr_intra = compute_pr( np.vstack(samples_naive_list), samples_ens)
        else:
            pr_intra = compute_pairwise(samples_naive_list, compute_pr, 'f1_prc_rec', sel_ens, sel_gans, **kwargs)
        return pr_inter, pr_intra
    elif name_obj == 'dc_inter__dc_intra':
        dc_inter = compute_dc(samples_real, samples_ens)
        if not pairwise:
            dc_intra = compute_dc(np.vstack(samples_naive_list), samples_ens)
        else:
            dc_intra = compute_pairwise(samples_naive_list, compute_dc, 'f1_dns_cvg', sel_ens, sel_gans, **kwargs)
        return dc_inter, dc_intra
    else:
        raise NotImplementedError
