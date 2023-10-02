import numpy as np
from itertools import combinations
from tqdm import tqdm
import copy
import optuna

from src.general_utils import util_general

def divide_samples(num_sample, num_class):
    return [(num_sample // num_class) + (i < num_sample % num_class) for i in range(num_class)]


def gan_list_epochs(gan_list, gan_epochs):
    x = []
    for gan in gan_list:
        for epochs in gan_epochs:
            x.append(f"{gan}_{epochs}")

    return x

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
def get_inter_metric(sel_gans, fitness_name, summary_flag, **kwargs):

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

def get_intra_metric(sel_gans, fitness_name, summary_flag, **kwargs):

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

def objective(trial, gan_models, gan_steps, obj_name, fitness_name, summary_flag, **kwargs):

    search_space = [f"{x}_{y}" for x in gan_models for y in gan_steps]
    sel_idx = [trial.suggest_int(x, 0, 1) for x in search_space]
    sel_gans = [x for i, x in enumerate(search_space) if sel_idx[i] == 1]
    n_search_space = len(sel_gans)

    if len(sel_gans) < 2:
        raise optuna.TrialPruned("Number of selected GANs is less than 2.")

    if obj_name == 'intra_inter':
        intra_metric = get_intra_metric(sel_gans, fitness_name, summary_flag, **kwargs)
        inter_metric = get_inter_metric(sel_gans, fitness_name, summary_flag, **kwargs)
        return intra_metric, inter_metric
    elif obj_name == 'intra':
        intra_metric = get_intra_metric(sel_gans, fitness_name, summary_flag, **kwargs)
        return intra_metric
    elif obj_name == 'inter':
        inter_metric = get_inter_metric(sel_gans, fitness_name, summary_flag, **kwargs)
        return inter_metric
    else:
        raise NotImplementedError

def backward_objective(ensemble, fitness_name, summary_flag, cost_name, **kwargs):

    intra_metric = get_intra_metric(search_space=ensemble, fitness_name=fitness_name, summary_flag=summary_flag, **kwargs)  # to maximize
    inter_metric = get_inter_metric(search_space=ensemble, fitness_name=fitness_name, summary_flag=summary_flag, **kwargs)  # to minimize
    print(f'Intra metric: {intra_metric}')
    print(f'Inter metric: {inter_metric}')


    if cost_name == 'diff':
        obj = inter_metric - intra_metric
    elif cost_name == 'ratio':
        obj = inter_metric / intra_metric # todo check cost
    elif cost_name == 'intra':
        obj = -intra_metric
    elif cost_name == 'inter':
        obj = inter_metric
    else:
        raise ValueError(cost_name)

    return obj