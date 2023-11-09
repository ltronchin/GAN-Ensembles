import copy
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
from torchvision.transforms import InterpolationMode
resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}

import src.custom_metrics.resize as resize

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

def normalize(img, norm_range, min_val=None, max_val=None):
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

def loader(img_path, img_size, mask_path, box, norm_range='-1,1',clahe=False, resizer_library='CV2_LANCZOS'):
    # Img
    img, photometric_interpretation = load_img(img_path)

    # Photometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        min_val, max_val = img.min(), img.max()
        img = np.interp(img, (min_val, max_val), (max_val, min_val))

    lower_img, upper_img = np.percentile(img.flatten(),  [2, 98])
    img = np.clip(img, lower_img, upper_img)
    min_val, max_val = img.min(), img.max()

    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)

    # Filter Mask
    if mask_path:
        mask, _ = load_img(mask_path)
        img = get_mask(img, mask, value=0)

    # Select Box Area
    if box:
        img = get_box(img, box, perc_border=0.5)

    # Resize
    if resizer_library == 'CV2_LANCZOS':
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    elif resizer_library == 'PIL_LANCZOS':
        img = Image.fromarray(img)
        img = img.resize((img_size, img_size), resample=Image.LANCZOS)
        img = np.array(img)
    else:
        raise ValueError(resizer_library)

    # Normalize
    img = normalize(img,  norm_range=norm_range, min_val=min_val, max_val=max_val)

    # Clahe
    if clahe:
        img = clahe_transform(img)

    # To Tensor
    img = torch.Tensor(img)
    img = torch.unsqueeze(img, dim=0) # add dimension

    return img

class AIforCOVIDImg(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch Dataloader to train images"""

    def __init__(self, data, classes, cfg_data, img_size, norm_range, clahe, resizer_library,loader_name="custom_preprocessing"):
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
        self.norm_range = norm_range
        self.img_size = img_size
        self.resizer_library = resizer_library
        self.clahe = clahe
        if loader_name == "custom_preprocessing":
            self.loader = lambda img_path, mask_path, box: loader(img_path=img_path, img_size=self.img_size, mask_path=mask_path, box=box, norm_range=self.norm_range, clahe=self.clahe, resizer_library=self.resizer_library)
        else:
            raise ValueError(loader_name)

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
        x = self.loader(img_path=img_path, mask_path=mask_path, box=box)
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
    def __init__(self, root_dir, gan_models, gan_steps, split='train', max_samples_per_gan=None, pil_loader=False):

        # Define the loader.
        if pil_loader:
            self.loader = custom_pil_loader
            self.trsf_list = [transforms.PILToTensor()]
        else:
            self.loader = custom_npy_loader
            self.trsf_list = [torch.from_numpy]

        gan_aval = os.listdir(root_dir)
        gan_folders = [os.path.join(root_dir, x, f'fake__{split}') for x in gan_aval if any(f'{gan_model}-train-' in x for gan_model in gan_models)]
        gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]
        gan_folders = [x for x in gan_folders if self.has_npy_files(x)]
        self.classes = copy.deepcopy(gan_folders)
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

    @staticmethod
    def has_npy_files(dir_path):
        """
        Recursively checks if the directory contains any .npy files.
        """
        for root, dirs, files in os.walk(dir_path):
            if any(x.endswith('.npy') for x in files):
                return True
        return False

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
        #print('Create the ensemble dataset.')
        for folder in folders:
            #print('Folder: ', folder)
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
    def __init__(self, dataset_synth, dataset_real, post_resizer='bilinear', model_name=None):
        self.dataset_synth = dataset_synth
        self.dataset_real = dataset_real
        self.model_name = model_name
        if model_name in ['InceptionV3_torch', 'ResNet50_torch', 'ResNet18_torch']:
            self.transfer_learning = True
            res= 299 if model_name == 'InceptionV3_torch' else 224
            self.resize = transforms.Resize(res, interpolation=resizer_collection[post_resizer],antialias=True)
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.transfer_learning = False

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

        # Clamp images between [-1 1]
        image = torch.clamp(image, -1, 1)

        if self.transfer_learning:
            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)  # grayscale to RGB
            image = self.resize(image)
            image = (image + 1) / 2 # to [0 1]
            image = self.normalize(image)

        return image, label



class ImageNetDataset(Dataset):
    def __init__(self, dataset, post_resizer='bilinear', model_name=None):

        self.dataset = dataset
        self.model_name = model_name
        if model_name in ['InceptionV3_torch', 'ResNet50_torch', 'ResNet18_torch']:
            self.transfer_learning = True
            res= 299 if model_name == 'InceptionV3_torch' else 224
            self.resize = transforms.Resize(res, interpolation=resizer_collection[post_resizer], antialias=True)
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.transfer_learning = False

        # Copy the attributes.
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # Check from which dataset the data should come

        image, label = self.dataset[index]

        # Clamp images between [-1 1]
        image = torch.clamp(image, -1, 1)

        if self.transfer_learning:
            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)  # grayscale to RGB
            image = self.resize(image)
            image = (image + 1) / 2 # to [0 1]
            image = self.normalize(image)

        return image, label