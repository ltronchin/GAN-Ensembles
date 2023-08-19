import sys
sys.path.extend([
    "./",
])

import numpy as np
import pickle
import torch
from tqdm import tqdm
import math
import time
import os
import hashlib
import uuid

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = pickle.load(f)
        obj = FeatureStats(capture_all=s['capture_all'], max_items=s['max_items'])
        obj.__dict__.update(s)
        return obj


def compute_feature(dataloader, eval_model, batch_size, quantize, device, cache_dir, max_items=None, **stats_kwargs):

    # Initialize.
    if max_items is None:
        max_items = len(dataloader.dataset)
    if max_items > 50000:
        max_items = 50000
    num_batches = math.ceil(float(max_items) / float(batch_size))

    # Try to lookup from cache.
    cache_file = None
    if cache_dir:
        # Choose cache file name.
        args = dict(eval_backbone=eval_model.eval_backbone, post_resizer=eval_model.post_resizer, max_items=max_items, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f"{max_items}-{eval_model.eval_backbone}-{eval_model.post_resizer}-{md5.hexdigest()}"
        cache_file = os.path.join(cache_dir, cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file)
        # Load.
        if flag:
            print("Using cache file found in {}".format(cache_dir))
            return FeatureStats.load(cache_file)


    stats = FeatureStats(max_items=max_items, **stats_kwargs)
    assert stats.max_items is not None

    eval_model.eval()
    data_iter = iter(dataloader)
    for _ in range(0, num_batches):
        try:
            images, labels  = next(data_iter)
            #if images.shape[1] != 3:
            #    images = images.repeat(1, 3, 1, 1)  # grayscale to RGB
        except StopIteration:
            break

        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features, _ = eval_model.get_outputs(images, quantize=quantize)
        stats.append_torch(features)

    # Save to cache.
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file)

    return stats

def compute_feature_real(self, max_items, **stats_kwargs):


    # Initialize.
    num_items = len(self.loader.dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = util_metric.FeatureStats(max_items=num_items, **stats_kwargs)

    # Main loop.
    for img, _ in self.loader:
        img = img.to(self.device)
        feat = self.detector(img)
        stats.append_torch(feat)

    # Save to cache.
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file)
    return stats


