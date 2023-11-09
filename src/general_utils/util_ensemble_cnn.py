import numpy as np
from itertools import combinations
from tqdm import tqdm
import copy
import optuna

from src.general_utils import util_general

def define_direction(n_obj, obj_name):
    directions = []
    for i in range(n_obj):
        if 'pr_inter' in obj_name:
            directions.append('maximize')
        elif 'pr_intra' in obj_name:
            directions.append('minimize')
        elif 'dc_inter' in obj_name:
            directions.append('maximize')
        elif 'dc_intra' in obj_name:
            directions.append('minimize')
        elif 'fid_inter' in obj_name:
            directions.append('minimize')
        elif 'fid_intra' in obj_name:
            directions.append('maximize')
        elif 'size' in obj_name:
            directions.append('minimize')
        else:
            raise NotImplementedError
    return directions

def FID(mu0, sigma0, mu1, sigma1):
    """Compute the Frechet Inception Distance."""

    m = np.square(mu1 - mu0).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma0), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma1 + sigma0 - s * 2))
    return fid

def compute_pairwise_distance(data_x, data_y=None):
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)
    return dists

def get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values
def compute_nearest_neighbour_distances(input_features, nearest_k):

    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii
def compute_pr(real_features, fake_features, nearest_k=5):

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
    recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

    return precision, recall
def compute_dc(real_features, fake_features, nearest_k=5):

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    density = (1. / float(nearest_k)) * (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()
    coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

    return density, coverage

def compute_statistics(x):

    """Compute the mean and covariance for the ensemble S."""
    mu_s = np.mean(x, axis=0)
    cov_s = np.cov(x, rowvar=False)

    return mu_s, cov_s

def obj(trial, samples_real, samples_naive_list, n_obj=1, name_obj='fid_inter'):
    """Objective function to minimize for ensemble S."""

    samples_real = np.vstack(samples_real)
    samples_naive = np.vstack(samples_naive_list)

    max_ens = len(samples_naive_list)
    sel_ens = [trial.suggest_int(f'G{x}', 0, 1) for x in range(len(samples_naive_list))]

    n_ens = np.sum(sel_ens)
    if n_ens < 1:
        raise optuna.TrialPruned("Number of selected GANs is less than 2.")
    samples_ens = np.vstack([samples_naive_list[i] for i in range(max_ens) if sel_ens[i] == 1])
    mu_ens, cov_ens = compute_statistics(samples_ens)

    if n_obj == 1:
        if name_obj == 'fid_inter':
            mu_real, cov_real = compute_statistics(samples_real)
            fid = FID(mu_ens, cov_ens, mu_real, cov_real)
            return fid
        elif name_obj == 'fid_intra':
            mu_naive, cov_naive = compute_statistics(samples_naive)
            fid = FID(mu_ens, cov_ens, mu_naive, cov_naive)
            return fid
        elif name_obj == 'pr_inter':
            precision, recall = compute_pr(samples_real, samples_ens, nearest_k=5)
            pr = 2 * (precision * recall) / (precision + recall)
            return pr
        elif name_obj == 'pr_intra':
            precision, recall = compute_pr(samples_naive, samples_ens, nearest_k=5)
            pr = 2 * (precision * recall) / (precision + recall)
            return pr
        elif name_obj == 'dc_inter':
            density, coverage = compute_dc(samples_real, samples_ens, nearest_k=5)
            dc = 2 * (density * coverage) / (density + coverage)
            return dc
        elif name_obj == 'dc_intra':
            density, coverage = compute_dc(samples_naive, samples_ens, nearest_k=5)
            dc = 2 * (density * coverage) / (density + coverage)
            return dc
        else:
            raise NotImplementedError

    elif n_obj == 2:
        if name_obj == 'pr_inter__pr_intra':
            precision_inter, recall_inter = compute_pr(samples_real, samples_ens, nearest_k=5)
            precision_intra, recall_intra = compute_pr(samples_naive, samples_ens, nearest_k=5)
            pr_inter = 2 * (precision_inter * recall_inter) / (precision_inter + recall_inter)
            pr_intra = 2 * (precision_intra * recall_intra) / (precision_intra + recall_intra)
            return pr_inter, pr_intra

        if name_obj == 'dc_inter__dc_intra':
            density_inter, coverage_inter = compute_dc(samples_real, samples_ens, nearest_k=5)
            density_intra, coverage_intra = compute_dc(samples_naive, samples_ens, nearest_k=5)
            dc_inter = 2 * (density_inter * coverage_inter) / (density_inter + coverage_inter)
            dc_intra = 2 * (density_intra * coverage_intra) / (density_intra + coverage_intra)
            return dc_inter, dc_intra

        elif name_obj == 'fid_inter__fid_intra':
            mu_real, cov_real = compute_statistics(samples_real)
            mu_naive, cov_naive = compute_statistics(samples_naive)
            fid_inter = FID(mu_ens, cov_ens, mu_real, cov_real)
            fid_intra = FID(mu_ens, cov_ens, mu_naive, cov_naive)
            return fid_inter, fid_intra
    else:
        raise NotImplementedError
