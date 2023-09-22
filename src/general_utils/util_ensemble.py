import numpy as np
from itertools import combinations
from tqdm import tqdm

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


def pho_div(models, k):

    # gan_div list of all couples.
    # k number of gans.

    div = 2 / (k * (k - 1))
    return div * sum(models)
def get_inter_metric(search_space, fitness_name, summary_flag, **kwargs):

    #print(f"Compute Inter {fitness_name}")
    data = kwargs['data_real']
    k = len(search_space)

    history = {
        fitness_name: [],
    }

    #with tqdm(total=k) as pbar:
    for gan in search_space:
        gan_name, step = gan.split('_')
        step = int(step)

        df = data[(data['gan0'] == gan_name) & (data['step0'] == step)]
        assert len(df) != 0

        history[fitness_name].append(df[fitness_name].values[0])
        #pbar.update(1)

    if summary_flag == 'mean':
        fitness_value = np.mean(history[fitness_name])
    elif summary_flag == 'div':
        fitness_value = pho_div(history[fitness_name], k)
    elif summary_flag == 'std':
        fitness_value = np.std(history[fitness_name])
    elif summary_flag == 'sem':
        fitness_value = np.std(history[fitness_name]) / np.sqrt(k)
    else:
        raise ValueError(f'Unknown summary flag: {summary_flag}')

    return fitness_value

def get_intra_metric(search_space, fitness_name, summary_flag, **kwargs):

    #print(f"Compute Intra {fitness_name}")
    data = kwargs['data_synth']

    k = len(search_space)

    history = {
        fitness_name: [],
    }
    tot_comb =len(list(combinations(search_space, 2)))
    #with tqdm(total=tot_comb) as pbar:
    for gan_comb in combinations(search_space, 2):
        gan_comb = util_general.parse_separated_list_comma(list(gan_comb))
        parts = gan_comb.split(',')
        gan0, step0 = parts[0].split('_')
        gan1, step1 = parts[1].split('_')
        step0 = int(step0)
        step1 = int(step1)

        # Filter data for current gan couple.

        df_couple = data[(data['gan0'] == gan0) & (data['step0'] == step0) &  (data['gan1'] == gan1) & (data['step1'] == step1)]
        if len(df_couple) == 0:
            df_couple = data[(data['gan0'] == gan1) & (data['step0'] == step1) &  (data['gan1'] == gan0) & (data['step1'] == step0)]
        assert len(df_couple) != 0, f'No data for {gan_comb}'

        history[fitness_name].append(df_couple[fitness_name].values[0])
        #pbar.update(1)

    # The fitness is the mean fid.
    if summary_flag == 'mean':
        fitness_value = np.mean(history[fitness_name])
    elif summary_flag == 'div':
        fitness_value = pho_div(history[fitness_name], k)
    elif summary_flag == 'std':
        fitness_value = np.std(history[fitness_name])
    elif summary_flag == 'sem':
        fitness_value = np.std(history[fitness_name]) / np.sqrt(k)
    else:
        raise ValueError(f'Unknown summary flag: {summary_flag}')

    return fitness_value

def objective(trial, search_space, fitness_name, summary_flag, cost_name, **kwargs):

    ensemble = trial.suggest_categorical('ensemble', search_space)
    ensemble = util_general.parse_comma_separated_list(ensemble)

    print('\n')
    print('Ensemble')
    print(ensemble)

    intra_metric = get_intra_metric(search_space=ensemble, fitness_name=fitness_name, summary_flag=summary_flag, **kwargs) # to maximize
    inter_metric = get_inter_metric(search_space=ensemble, fitness_name=fitness_name, summary_flag=summary_flag, **kwargs) # to minimize

    print(f'Intra metric: {intra_metric}')
    print(f'Inter metric: {inter_metric}')

    if cost_name == 'diff':
        obj = inter_metric - intra_metric
    elif cost_name == 'ratio':
        obj = inter_metric / intra_metric
    elif cost_name == 'intra':
        obj = -intra_metric
    elif cost_name == 'inter':
        obj = inter_metric
    else:
        raise ValueError(cost_name)

    return obj

def backward_objective(ensemble, fitness_name, summary_flag, cost_name, **kwargs):

    intra_metric = get_intra_metric(search_space=ensemble, fitness_name=fitness_name, summary_flag=summary_flag, **kwargs)  # to maximize
    inter_metric = get_inter_metric(search_space=ensemble, fitness_name=fitness_name, summary_flag=summary_flag, **kwargs)  # to minimize
    print(f'Intra metric: {intra_metric}')
    print(f'Inter metric: {inter_metric}')


    if cost_name == 'diff':
        obj = inter_metric - intra_metric
    elif cost_name == 'ratio':
        obj = inter_metric / intra_metric
    elif cost_name == 'intra':
        obj = -intra_metric
    elif cost_name == 'inter':
        obj = inter_metric
    else:
        raise ValueError(cost_name)

    return obj