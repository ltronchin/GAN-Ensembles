import os

import cv2
import math
import numpy as np
import pydicom
import torch
from PIL import Image
import pandas as pd
from random import choices, randint
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder


def get_dataset(data_name, data_dir, cfgs=None, train=True):

    if data_name in  [
        "pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist", "tissuemnist", "organamnist"
    ]:
        mode = "train" if train == True else "valid"
        if data_name == "pathmnist":
            info = INFO['pathmnist']
        elif data_name == "chestmnist":
            info = INFO['chestmnist']
        elif data_name == "dermamnist":
            info = INFO['dermamnist']
        elif data_name == "octmnist":
            info = INFO['octmnist']
        elif data_name == "pneumoniamnist":
            info = INFO['pneumoniamnist']
        elif data_name == "retinamnist":
            info = INFO['retinamnist']
        elif data_name == "breastmnist":
            info = INFO['breastmnist']
        elif data_name == "bloodmnist":
            info = INFO['bloodmnist']
        elif data_name == "tissuemnist":
            info = INFO['tissuemnist']
        elif data_name == "organamnist":
            info = INFO['organamnist']
        else:
            raise NotImplementedError
        DataClass = getattr(medmnist, info['python_class'])
        util_path.create_dir(os.path.join(data_dir, data_name))
        dataset = DataClass(root=data_dir, split=mode, download=True)

    # Custom code for AERTS CT
    elif data_name == 'AERTS':
        raise NotImplementedError

    # Custom code for CLARO CT
    elif data_name == 'CLARO':
        raise NotImplementedError

    # Custom code for AI4Covid CXR
    elif data_name == "AIforCOVID":

        assert cfgs is not None

        mode = "train" if train == True else "val"
        fold_data = pd.read_csv(os.path.join(cfgs.DATA.fold_dir, f'{mode}.txt'), delimiter=" ", index_col=0)
        dataset = util_data.AIforCOVIDImg(
            data=fold_data,
            classes=cfgs.DATA.classes,
            cfg_data=cfgs.DATA.modes['img'],
            img_size=cfgs.DATA.img_size
        )
    else:
        mode = "train" if train == True else "valid"
        root = os.path.join(data_dir, mode)
        dataset = ImageFolder(root=root)

    return dataset

# ----------------------------------------------------------------------------------------------------------------------
# AiforCovid
def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    if extension == ".dcm":
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array.astype(float)
        photometric_interpretation = dicom.PhotometricInterpretation
    else:
        img = Image.open(img_path)
        img = np.array(img).astype(float)
        photometric_interpretation = None
    return img, photometric_interpretation
def get_img_loader(loader_name):
    if loader_name == "custom_preprocessing":
        return loader
    else:
        raise ValueError(loader_name)

def get_mask(img, mask, value=1):
    mask = mask != value
    img[~mask] = 0
    return img

def clahe_transform(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply((img * 255).astype(np.uint8)) / 255
    return img

def normalize(img,norm_range, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()
    img = (img - min_val) / (max_val - min_val)

    if norm_range == '-1,1':
        img = (img - 0.5) * 2  # Adjusts to -1 to 1 if desired

    return img

def get_box(img, box, perc_border=.0):
    # Sides
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    # Border
    diff_1 = math.ceil((abs(l_h - l_w) / 2))
    diff_2 = math.floor((abs(l_h - l_w) / 2))
    border = int(perc_border * diff_1)
    # Img dims
    img_h = img.shape[0]
    img_w = img.shape[1]
    if l_h > l_w:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-diff_1-border < 0:
            pad = 0-(box[1]-diff_1-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+diff_2+border > img_w:
            pad = (box[3]+diff_2+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-diff_1-border:box[3]+diff_2+border]
    elif l_w > l_h:
        if box[0]-diff_1-border < 0:
            pad = 0-(box[0]-diff_1-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+diff_2+border > img_h:
            pad = (box[2]+diff_2+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-diff_1-border:box[2]+diff_2+border, box[1]-border:box[3]+border]
    else:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-border:box[3]+border]
    return img


def loader(img_path, img_size, mask_path=None, box=None, norm_range='0,1', clahe=False):
    # Img
    img, photometric_interpretation = load_img(img_path)
    min_val, max_val = img.min(), img.max()
    # Pathometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
        min_val, max_val = img.min(), img.max()
    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)
    # Filter Mask
    if mask_path:
        mask, _ = load_img(mask_path)
        img = get_mask(img, mask, value=1)
    # Select Box Area
    if box:
        img = get_box(img, box, perc_border=0.5)
    # Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    # Normalize
    img = normalize(img,  norm_range=norm_range, min_val=min_val, max_val=max_val)
    # clahe
    if clahe:
        img = clahe_transform(img)
    # To Tensor
    img = torch.Tensor(img)
    img = torch.unsqueeze(img, dim=0) # add dimension

    return img

class AIforCOVIDImg(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch Dataloader to train images"""

    def __init__(self, data, classes, cfg_data, img_size):
        """Initialization"""
        self.img_dir = cfg_data['img_dir']
        self.data = data
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        # Mask (to select only the lungs pixels)
        if cfg_data['mask_dir']:
            self.masks = {id_patient: os.path.join(cfg_data['mask_dir'], '%s.tif' % id_patient) for id_patient in
                          data.index}
        else:
            self.masks = None
        # Box (to select only the box containing the lungs)
        if cfg_data['box_file']:
            box_data = pd.read_excel(cfg_data['box_file'], index_col="id", dtype=list)
            self.boxes = {row[0]: eval(row[1]["box"]) for row in box_data.iterrows()}
        else:
            self.boxes = None
        self.norm_range = cfg_data['norm_range']
        self.img_size = img_size
        self.loader = get_img_loader('custom_preprocessing')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        if self.masks:
            mask_path = self.masks[id]
        else:
            mask_path = None
        # load box
        if self.boxes:
            box = self.boxes[id]
        else:
            box = None
        # Load data and get label
        img_path = os.path.join(self.img_dir, '%s.dcm' % id)
        x = self.loader(img_path=img_path, img_size=self.img_size, mask_path=mask_path, box=box, norm_range=self.norm_range, clahe=False)
        y = row.label

        return x, self.class_to_idx[y], id


# ----------------------------------------------------------------------------------------------------------------------

def custom_pil_loader(path):
    #https://discuss.pytorch.org/t/loaded-grayscale-image-is-converted-into-3-channel/75862/5
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img

def custom_npy_loader(path):
    img = np.load(path)
    return img

class DiscDataset(Dataset):
    def __init__(self, root_dir, classes, max_samples_per_gan=None, pil_loader=False):

        # Define the loader.
        if pil_loader:
            self.loader = custom_pil_loader
            self.trsf_list = [transforms.PILToTensor()]
        else:
            self.loader = custom_npy_loader
            self.trsf_list = [torch.from_numpy]

        self.root_dir = root_dir
        self.classes = classes
        self.max_samples_per_gan = max_samples_per_gan
        self.samples = []

        self.trsf = transforms.Compose(self.trsf_list)
        self._prepare_dataset_gan()

    def _prepare_dataset_gan(self):

        for class_id, root_dir in enumerate(self.classes):
            sample_count = 0
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if self.max_samples_per_gan and sample_count >= self.max_samples_per_gan:
                        break
                    if file.endswith(".tiff") or file.endswith(".npy"):
                        image_path = os.path.join(root, file)
                        self.samples.append((image_path, class_id))
                        sample_count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = self.loader(image_path)

        # return self.trsf(image), label, image_path
        return self.trsf(image), label

class EnsembleDataset(Dataset):
    def __init__(self, folders, weights, pil_loader=False):

        assert len(folders) == len(weights)
        # assert sum(weights) == 1

        if pil_loader:
            self.loader = custom_pil_loader
            self.trsf_list = [transforms.PILToTensor()]
        else:
            self.loader = custom_npy_loader
            self.trsf_list = [torch.from_numpy]

        self.folders = folders
        self.weights = weights

        self.trsf = transforms.Compose(self.trsf_list)
        self.image_folders = []
        print('Create the ensemble dataset.')
        for folder in folders:
            print('Folder: ', folder)
            self.image_folders.append(DatasetFolder(root=folder, loader=self.loader, extensions=('.npy', '.tiff'))) #  self.image_folders.append(ImageFolder(root=folder, loader=self.loader))
        # self.image_folders = [ImageFolder(root=folder, loader=custom_pil_loader) for folder in folders]

    def __len__(self):
        return len(self.image_folders[0]) # sum([len(folder) for folder in self.image_folders])

    def __getitem__(self, _):

        folder_idx = choices(range(len(self.image_folders)), weights=self.weights)[0]
        sample_idx = randint(0, len(self.image_folders[folder_idx]) - 1)

        image, label = self.image_folders[folder_idx][sample_idx]
        # img_path = self.image_folders[folder_idx].imgs[sample_idx][0]

        # return self.trsf(image), label, img_path.split('fake')[0], img_path
        return self.trsf(image), label

class DummyDataset(Dataset):
    def __init__(self, folder, pil_loader=False):

        if pil_loader:
            self.loader = custom_pil_loader
            self.trsf_list = [transforms.PILToTensor()]
        else:
            self.loader = custom_npy_loader
            self.trsf_list = [torch.from_numpy]

        self.trsf = transforms.Compose(self.trsf_list)
        self.dataset = DatasetFolder(root=folder, loader=self.loader, extensions=('.npy', '.tiff')) # self.dataset = ImageFolder(root=folder, loader=self.loader)

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):

        image, label = self.dataset[idx]
        return self.trsf(image), label

class GANDataset(Dataset):
    def __init__(self, folder, pil_loader=False):

        if pil_loader:
            self.loader = custom_pil_loader
            self.trsf_list = [transforms.PILToTensor()]
        else:
            self.loader = custom_npy_loader
            self.trsf_list = [torch.from_numpy]

        self.trsf = transforms.Compose(self.trsf_list)
        self.dataset = DatasetFolder(root=folder, loader=self.loader, extensions=('.npy', '.tiff')) # self.dataset = ImageFolder(root=folder, loader=self.loader)

    def __len__(self):
        return len(self.dataset) # sum([len(folder) for folder in self.image_folders])

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        return self.trsf(image), label

class MergedDataset(Dataset):
    def __init__(self, dataset_synth, dataset_real):
        self.dataset_synth = dataset_synth
        self.dataset_real = dataset_real

        # Calculate the number of classes in dataset_synth
        self.classes_synth = len(set([label for _, label in dataset_synth.samples]))
        self.class_real = self.classes_synth

        self.total_classes = self.classes_synth + 1
        self.total_len = len(self.dataset_synth) + len(self.dataset_real)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # Check from which dataset the data should come
        if index < len(self.dataset_synth):
            image, label = self.dataset_synth[index]
        else:
            image, _ = self.dataset_real[index - len(self.dataset_synth)]
            label = self.class_real

        return image, label