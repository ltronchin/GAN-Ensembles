import sys
import numpy as np
import scipy
import os
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sklearn
import copy
import random
from scipy.stats import gaussian_kde
import time

import pandas as pd
from itertools import product
from src.general_utils import util_path
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix
from itertools import combinations
import sys

from src.general_utils import util_metric

# For aspect ratio 4:3.
sns.set_context("paper")
sns.set_theme(style="ticks")

def recovered_mode(data, mu_ref, mu_synth=None, thres=0.1):

    rec_modes = 0
    if mu_synth is not None:
        mu_synth = np.array(mu_synth)
        penalty = len(mu_synth)
        for mu in mu_ref:
            d = np.linalg.norm(mu_synth - mu, axis=1)
            if (d < thres).any():
                rec_modes += 1

        #rec_modes /= penalty

    else:
        for mu in mu_ref:
            d = np.linalg.norm(data - mu, axis=1)
            if (d < thres).any():
                rec_modes += 1

    return rec_modes

def frac_points(data, mu_ref, thres=0.1):
    frac = 0
    for sample in data:
        d = np.linalg.norm(mu_ref - sample, axis=1)
        if (d < thres).any():
            frac += 1
    frac /= len(data)

    return frac

def compute_metrics(y_test_real, y_pred, aggregation='mean'):
    cm = confusion_matrix(y_test_real, y_pred)
    num_classes = cm.shape[0]

    accuracy = np.trace(cm) / float(np.sum(cm))

    precision_list = []
    recall_list = []
    f1_score_list = []
    specificity_list = []
    g_mean_list = []

    for j in range(num_classes):
        tp = cm[j, j]
        fp = np.sum(cm[:, j]) - tp
        fn = np.sum(cm[j, :]) - tp
        tn = np.sum(cm) - (fp +fn + tp)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        g_mean = np.sqrt(recall * specificity)

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_score_list.append(f1)
        g_mean_list.append(g_mean)

    if aggregation == 'mean':
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        specificity = np.mean(specificity_list)
        f1_score = np.mean(f1_score_list)
        g_mean = np.mean(g_mean_list)
    else:
        raise NotImplementedError

    return accuracy, precision, recall, specificity, f1_score, g_mean

def obj_optuna(trial, samples_real, samples_naive_list, n_obj=1, name_obj='fid_inter', pairwise=True):
    """Objective function to minimize for ensemble S."""

    samples_real = np.vstack(samples_real)
    samples_naive = np.vstack(samples_naive_list)

    max_ens = len(samples_naive_list)
    sel_ens = [trial.suggest_int(f'G{x}', 0, 1) for x in range(len(samples_naive_list))]

    n_ens = np.sum(sel_ens)
    if n_ens < 1:
        raise optuna.TrialPruned("Number of selected GANs is less than 2.")
    samples_ens_list = [samples_naive_list[j] for j in range(max_ens) if sel_ens[j] == 1]
    samples_ens = np.vstack(samples_ens_list)

    mu_ens, cov_ens = util_metric.compute_statistics(samples_ens)

    eps = np.finfo(float).eps
    if n_obj == 1:
        if name_obj == 'fid_inter':
            mu_real, cov_real = util_metric.compute_statistics(samples_real)
            fid = util_metric.FID(mu_ens, cov_ens, mu_real, cov_real)
            return fid
        elif name_obj == 'fid_intra':
            if not pairwise:
                mu_naive, cov_naive = util_metric.compute_statistics(samples_naive)
                fid = util_metric.FID(mu_ens, cov_ens, mu_naive, cov_naive)
            else:
                fid_list = []
                idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]
                for idxs_comb in combinations(idxs_sel_ens, 2):

                    gan0 = samples_naive_list[idxs_comb[0]]
                    gan1 = samples_naive_list[idxs_comb[1]]
                    gan0_mu, gan0_cov = util_metric.compute_statistics(gan0)
                    gan1_mu, gan1_cov = util_metric.compute_statistics(gan1)

                    fid_list.append(util_metric.FID(gan0_mu, gan0_cov, gan1_mu, gan1_cov))
                fid = np.mean(fid_list)
            return fid

        elif name_obj == 'pr_inter':
            precision, recall = util_metric.compute_pr(samples_real, samples_ens, nearest_k=5)
            pr = 2 * (precision * recall) / (precision + recall)
            return pr
        elif name_obj == 'pr_intra':
            if not pairwise:
                precision, recall = util_metric.compute_pr(samples_naive, samples_ens, nearest_k=5)
                pr = 2 * (precision * recall) / (precision + recall)
            else:
                pr_list = []
                idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]

                for idxs_comb in combinations(idxs_sel_ens, 2):

                    gan0 = samples_naive_list[idxs_comb[0]]
                    gan1 = samples_naive_list[idxs_comb[1]]

                    precision, recall = util_metric.compute_pr(gan0, gan1, nearest_k=5)
                    pr01 = 2 * (precision * recall) / ((precision + recall) + eps)
                    pr_list.append(pr01)

                pr = np.mean(pr_list)
            return pr
        elif name_obj == 'dc_inter':
            density, coverage = util_metric.compute_dc(samples_real, samples_ens, nearest_k=5)
            dc = 2 * (density * coverage) / (density + coverage)
            return dc
        elif name_obj == 'dc_intra':
            if not pairwise:
                density, coverage = util_metric.compute_dc(samples_naive, samples_ens, nearest_k=5)
                dc = 2 * (density * coverage) / (density + coverage)
            else:
                dc_list = []
                idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]
                eps = np.finfo(float).eps
                for idxs_comb in combinations(idxs_sel_ens, 2):
                    gan0 = samples_naive_list[idxs_comb[0]]
                    gan1 = samples_naive_list[idxs_comb[1]]

                    density, coverage = util_metric.compute_dc(gan0, gan1, nearest_k=5)
                    dc = 2 * (density * coverage) / ((density + coverage) + eps)
                    dc_list.append(dc)

                dc = np.mean(dc_list)
            return dc
        elif name_obj == 'size':
            size_term = (n_ens / max_ens)
            return size_term
        elif name_obj == 'entropy':
            entropy =  util_metric.compute_kde_entropy(samples_ens)
            return entropy
        elif name_obj == 'coverage_real':
            _, cov_real = util_metric.compute_statistics(samples_real)
            cov = np.linalg.det(cov_ens) / np.linalg.det(cov_real)
            return cov
        elif name_obj == 'coverage_naive':
            _, cov_naive = util_metric.compute_statistics(samples_naive)
            cov = np.linalg.det(cov_ens) / np.linalg.det(cov_naive)
            return cov

    elif n_obj == 2:
        if name_obj == 'pr_inter__pr_intra':
            precision_inter, recall_inter = util_metric.compute_pr(samples_real, samples_ens, nearest_k=5)
            pr_inter = 2 * (precision_inter * recall_inter) / (precision_inter + recall_inter)
            if not pairwise:
                precision_intra, recall_intra = util_metric.compute_pr(samples_naive, samples_ens, nearest_k=5)
                pr_intra = 2 * (precision_intra * recall_intra) / (precision_intra + recall_intra)
            else:
                pr_list = []
                idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]
                for idxs_comb in combinations(idxs_sel_ens, 2):

                    gan0 = samples_naive_list[idxs_comb[0]]
                    gan1 = samples_naive_list[idxs_comb[1]]

                    precision, recall = util_metric.compute_pr(gan0, gan1, nearest_k=5)
                    pr01 = 2 * (precision * recall) / ((precision + recall) + eps)
                    pr_list.append(pr01)

                pr_intra = np.mean(pr_list)
            return pr_inter, pr_intra

        if name_obj == 'dc_inter__dc_intra':
            density_inter, coverage_inter = util_metric.compute_dc(samples_real, samples_ens, nearest_k=5)
            dc_inter = 2 * (density_inter * coverage_inter) / (density_inter + coverage_inter)
            if not pairwise:
                density_intra, coverage_intra = util_metric.compute_dc(samples_naive, samples_ens, nearest_k=5)
                dc_intra = 2 * (density_intra * coverage_intra) / (density_intra + coverage_intra)
            else:
                dc_list = []
                idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]
                eps = np.finfo(float).eps
                for idxs_comb in combinations(idxs_sel_ens, 2):
                    gan0 = samples_naive_list[idxs_comb[0]]
                    gan1 = samples_naive_list[idxs_comb[1]]

                    density, coverage = util_metric.compute_dc(gan0, gan1, nearest_k=5)
                    dc = 2 * (density * coverage) / ((density + coverage) + eps)
                    dc_list.append(dc)

                dc_intra = np.mean(dc_list)
            return dc_inter, dc_intra

        elif name_obj == 'fid_inter__fid_intra':
            mu_real, cov_real = util_metric.compute_statistics(samples_real)
            fid_inter = util_metric.FID(mu_ens, cov_ens, mu_real, cov_real)
            if not pairwise:
                mu_naive, cov_naive = util_metric.compute_statistics(samples_naive)
                fid_intra = util_metric.FID(mu_ens, cov_ens, mu_naive, cov_naive)
            else:
                fid_list = []
                idxs_sel_ens = [j for j, val in enumerate(sel_ens) if val == 1]
                for idxs_comb in combinations(idxs_sel_ens, 2):
                    gan0 = samples_naive_list[idxs_comb[0]]
                    gan1 = samples_naive_list[idxs_comb[1]]
                    gan0_mu, gan0_cov = util_metric.compute_statistics(gan0)
                    gan1_mu, gan1_cov = util_metric.compute_statistics(gan1)

                    fid_list.append(util_metric.FID(gan0_mu, gan0_cov, gan1_mu, gan1_cov))
                fid_intra = np.mean(fid_list)

            return fid_inter, fid_intra
    else:
        raise NotImplementedError

def generate_data(n, mu, dev):
    data = []
    per_gaussian = n // len(mu)
    for mean in mu:
        samples = np.random.normal(mean, dev, (per_gaussian, 2))
        data.append(samples)
    return np.vstack(data)

# DATA GENERATION - GAUSSIANS #
def closest_point(mean, classes):
    d = [np.linalg.norm(np.array(mean) - c) for c in classes]
    return np.argmin(d)
def generate_gaussian_data(num_gaussians, num_samples, max_covx=0.1, max_covy=0.1, min_mux=4, min_muy=4, max_mux=12, max_muy=12,seed=42):
    data_list = []
    mu_list = []
    cov_list = []
    np.random.seed(seed)  # Set numpy pseudo-random generator at a fixed value
    random.seed(seed)  # Set python built-in pseudo-random generator at a fixed value
    for _ in range(num_gaussians):
        mu = [
            np.random.uniform(min_mux, max_mux),
            np.random.uniform(min_muy, max_muy)
        ]
        cov = np.diag(
            [np.random.uniform(0.05, max_covx),
             np.random.uniform(0.05, max_covy)]
        )

        data = np.random.multivariate_normal(mu, cov, num_samples)

        data_list.append(data)
        mu_list.append(mu)
        cov_list.append(cov)

    return data_list, np.vstack(data_list), mu_list, cov_list

def create_dataset(data, labels):
    x = []
    y = []
    gauss_idx = []
    for i, label in enumerate(labels):
        samples = data[i]
        lab = np.repeat(label, len(samples)).reshape(-1, 1)
        x.append(samples)
        y.append(lab)
        gauss_idx.append(np.repeat(i, len(samples)).reshape(-1, 1))

    return np.vstack(x), np.vstack(y), np.vstack(gauss_idx)