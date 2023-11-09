import numpy as np
import scipy
import seaborn as sns
import sklearn
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score
from multiprocessing import Pool

# For aspect ratio 4:3.
sns.set_context("paper")
sns.set_theme(style="ticks")

def compute_distances_chunk(args):
    data_x, data_y, i, chunk_size = args
    end_i = i + chunk_size
    batch_x = data_x[i:end_i]
    batch_dists = sklearn.metrics.pairwise_distances(batch_x, data_y, metric='euclidean', n_jobs=1)
    return i, batch_dists

def compute_pairwise_distance(data_x, data_y=None, in_chunks=False, chunk_size=1000, n_jobs=None):
    if data_y is None:
        data_y = data_x

    if in_chunks:
        # Initialize the distances array
        dists = np.zeros((data_x.shape[0], data_y.shape[0]), dtype=np.float32)
        # Define the number of jobs to run in parallel if n_jobs is None
        n_jobs = min(Pool()._processes, len(range(0, data_x.shape[0], chunk_size)))

        chunks_args = [(data_x, data_y, i, chunk_size) for i in range(0, data_x.shape[0], chunk_size)]
        with Pool(n_jobs) as pool:
            results = pool.map(compute_distances_chunk, chunks_args)

        for i, batch_dists in results:
            dists[i:i+chunk_size, :] = batch_dists
    else:
        dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=n_jobs)

    return dists

def compute_kde_entropy(data, n_samples=10000):

    kde = gaussian_kde(data.T)
    samples = kde.resample(n_samples)
    densities = kde(samples)
    entropy = -np.mean(np.log(densities + np.finfo(float).eps))

    return entropy
def FID(mu0, sigma0, mu1, sigma1):
    """Compute the Frechet Inception Distance."""

    m = np.square(mu1 - mu0).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma0), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma1 + sigma0 - s * 2))
    return fid

# def compute_pairwise_distance(data_x, data_y=None, in_chunks=False, chunk_size=1000):
#     if data_y is None:
#         data_y = data_x
#
#     if in_chunks:
#         dists = np.zeros((data_x.shape[0], data_y.shape[0]), dtype=np.float32)
#         for i in range(0, data_x.shape[0], chunk_size):
#
#             end_i = i + chunk_size
#             batch_x = data_x[i:end_i]
#
#             batch_dists = sklearn.metrics.pairwise_distances(batch_x, data_y, metric='euclidean', n_jobs=1)
#             dists[i:end_i, :] = batch_dists
#     else:
#         dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)
#     return dists

def get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values

def compute_nearest_neighbour_distances(input_features, nearest_k, in_chunks):

    distances = compute_pairwise_distance(data_x=input_features, in_chunks=in_chunks)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii
def compute_pr(real_features, fake_features, nearest_k=5, in_chunks=False):

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(input_features=real_features, nearest_k=nearest_k, in_chunks=in_chunks)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(input_features=fake_features, nearest_k=nearest_k, in_chunks=in_chunks)
    distance_real_fake = compute_pairwise_distance(data_x=real_features, data_y=fake_features, in_chunks=in_chunks)

    precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
    recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

    return precision, recall
def compute_dc(real_features, fake_features, nearest_k=5, in_chunks=False):

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(input_features=real_features, nearest_k=nearest_k, in_chunks=in_chunks)
    distance_real_fake = compute_pairwise_distance(data_x=real_features, data_y=fake_features, in_chunks=in_chunks)

    density = (1. / float(nearest_k)) * (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()
    coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

    return density, coverage

def compute_statistics(x):

    """Compute the mean and covariance for the ensemble S."""
    mu_s = np.mean(x, axis=0)
    cov_s = np.cov(x, rowvar=False)

    return mu_s, cov_s
