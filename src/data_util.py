# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np

# Custom imports
import medmnist
from medmnist import INFO
import pandas as pd
from src.general_utils import util_data
from src.general_utils import util_path

resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size)) # not apply for squared images

    def __repr__(self):
        return self.__class__.__name__

class Dataset_(Dataset):
    def __init__(self,
                 data_name,
                 data_dir,
                 train,
                 crop_long_edge=False,
                 resize_size=None,
                 resizer="lanczos",
                 random_flip=False,
                 normalize=True,
                 hdf5_path=None,
                 load_data_in_memory=False,
                 # for additional arguments
                 cfgs = None,
                 split=None):
        super(Dataset_, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.train = train
        self.random_flip = random_flip
        self.normalize = normalize
        self.hdf5_path = hdf5_path
        self.load_data_in_memory = load_data_in_memory
        self.trsf_list = []
        # custom - manage single channel images
        self.cfgs = cfgs
        self.grayscale = True if data_name in ["pneumoniamnist", "breastmnist", "chestmnist", "organamnist", "AIforCOVID"] else False
        self.split = split

        if self.hdf5_path is None:
            if crop_long_edge:
                self.trsf_list += [CenterCropLongEdge()]
            if resize_size is not None and resizer != "wo_resize":
                self.trsf_list += [transforms.Resize(resize_size, interpolation=resizer_collection[resizer])]
        else:
            self.trsf_list += [transforms.ToPILImage()]

        if self.random_flip:
            self.trsf_list += [transforms.RandomHorizontalFlip()]

        if self.normalize:
            self.trsf_list += [transforms.ToTensor()]
            if self.grayscale:  # manage single channel images
                self.trsf_list += [transforms.Normalize([0.5], [0.5])]
            else:
                self.trsf_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        else:
            self.trsf_list += [transforms.PILToTensor()]

        self.trsf = transforms.Compose(self.trsf_list)
        print(self.trsf)

        self.load_dataset()

    def load_dataset(self):
        if self.split is None:
            mode = "train" if self.train == True else "val"
        else:
            mode = self.split

        if self.hdf5_path is not None:
            parts = self.hdf5_path.split("/")[-1].split("_")
            assert mode == parts[-1].split(".")[0]
            with h5.File(self.hdf5_path, "r") as f:
                if self.data_name == "AIforCOVID":
                    data, labels, ids = f["imgs"], f["labels"], f["ids"]
                else:
                    data, labels = f["imgs"], f["labels"]
                self.num_dataset = data.shape[0]
                if self.load_data_in_memory:
                    print("Load {path} into memory.".format(path=self.hdf5_path))
                    self.data = data[:]
                    self.labels = labels[:]
                    if self.data_name == "AIforCOVID":
                        self.ids = ids[:]
            return

        if self.data_name == "CIFAR10":
            self.data = CIFAR10(root=self.data_dir, train=self.train, download=True)

        elif self.data_name == "CIFAR100":
            self.data = CIFAR100(root=self.data_dir, train=self.train, download=True)
        # Custom codes for MedMnist datasets.
        elif self.data_name in ["chestmnist", "pneumoniamnist", "retinamnist", "breastmnist", "organamnist"]:
            if self.data_name == "chestmnist":
                info = INFO['chestmnist']
                info['label'] = {
                    '0': 'no_finding',
                    '1': 'finding'
                }


            elif self.data_name == "pneumoniamnist":
                info = INFO['pneumoniamnist']
            elif self.data_name == "retinamnist":
                info = INFO['retinamnist']
            elif self.data_name == "breastmnist":
                info = INFO['breastmnist']
            elif self.data_name == "organamnist":
                info = INFO['organamnist']
            else:
                raise NotImplementedError
            DataClass = getattr(medmnist, info['python_class'])
            util_path.create_dir(os.path.join(self.data_dir, self.data_name))
            self.data = DataClass(root=self.data_dir, split=mode, download=True)
            if self.data_name == "chestmnist":
                self.data.labels = np.array([x.any() for x in self.data.labels != 0], dtype='int64')
                self.data.labels = np.expand_dims(self.data.labels, axis=1)

        # Custom code for AERTS CT
        elif self.data_name == 'AERTS':
            pass
        # Custom code for CLARO CT
        elif self.data_name == 'CLARO':
            pass
        # Custom code for AI4Covid CXR
        elif self.data_name == "AIforCOVID":
            fold_data = pd.read_csv(os.path.join(self.cfgs.DATA.fold_dir, f'{mode}.txt'), delimiter=" ", index_col=0)
            self.data = util_data.AIforCOVIDImg(
                data=fold_data,
                classes=self.cfgs.DATA.classes,
                cfg_data=self.cfgs.DATA.modes['img'],
                img_size=self.cfgs.DATA.img_size,
                norm_range=self.cfgs.DATA.modes['img']['norm_range'],
                clahe=self.cfgs.DATA.clahe,
                resizer_library=self.cfgs.DATA.resizer_library
            )
        else:
            root = os.path.join(self.data_dir, mode)
            self.data = ImageFolder(root=root)
    def _get_hdf5(self, index):
        with h5.File(self.hdf5_path, "r") as f:
            return f["imgs"][index], f["labels"][index]

    def __len__(self):
        if self.hdf5_path is None:
            num_dataset = len(self.data)
        else:
            num_dataset = self.num_dataset
        return num_dataset

    def __getitem__(self, index):
        if self.hdf5_path is None:
            if self.data_name == 'AIforCOVID':
                img, label, _ = self.data[index]
                return img, int(label)
            img, label = self.data[index]
        else:
            if self.load_data_in_memory:
                img, label = self.data[index], self.labels[index]
                if self.data_name == 'AIforCOVID':
                    img = torch.Tensor(img)
                    return img, int(label)
            else:
                img, label = self._get_hdf5(index)
                if self.data_name == 'AIforCOVID':
                    img = torch.Tensor(img)
                    return img, int(label)
        return self.trsf(img), int(label)


