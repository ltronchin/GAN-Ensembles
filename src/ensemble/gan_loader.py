import sys
sys.path.extend([
    "./",
    "./src/",
])
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from random import choices, randint
from collections import Counter
from PIL import Image

from src.general_utils import util_ensemble

def custom_pil_loader(path):
    #https://discuss.pytorch.org/t/loaded-grayscale-image-is-converted-into-3-channel/75862/5
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img

def custom_npy_loader(path):
    img = np.load(path)
    return img

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

if __name__ == '__main__':

    dataset_name = 'pneumoniamnist'
    reports_dir = '/home/lorenzo/GAN-Ensembles/reports/'
    source_dir = '/home/lorenzo/GAN-Ensembles/reports/'
    samples_dir = os.path.join(source_dir, dataset_name, 'samples')
    split='train'
    gan_aval = os.listdir(samples_dir)
    init_w = 'uniform'

    gan_steps = [
        '100000'
    ]
    gan_models = [
        'ACGAN-Mod-ADC',
        'SAGAN',
        'ReACGAN-ADA'
    ]

    # Create the entire path for each gan.
    gan_folders = [os.path.join(root_dir, x, f'fake__{split}') for x in gan_aval if any(f'{gan_model}-train-' in x for gan_model in gan_models)]
    gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]

    # Initialize weights.
    weights = util_ensemble.initialize_ensemble_weights(init_w=init_w, gan_list=gan_folders)
    dataset = EnsembleDataset(folders=gan_folders, weights=weights)

    # Dataloader.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=1)
    data_iter = iter(data_loader)
    # x, y, x_paths, x_full_paths = next(data_iter)

    # print(Counter(x_paths))
    x, y = next(data_iter)

    print("May be the force with you.")