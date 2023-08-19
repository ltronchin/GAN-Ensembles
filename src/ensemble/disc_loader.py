import sys
sys.path.extend([
    "./",
    "./src/",
])
from torch.utils.data import Dataset

import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset


def custom_pil_loader(path):
    #https://discuss.pytorch.org/t/loaded-grayscale-image-is-converted-into-3-channel/75862/5
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img

class DiscDataset(Dataset):
    def __init__(self, root_dir, classes=None, step='10000'):

        self.root_dir = root_dir
        self.step = step

        if classes is None:
            self.classes = os.listdir(self.root_dir)
        else:
            self.classes = classes

        self.trsf_list = [transforms.PILToTensor()]
        self.trsf = transforms.Compose(self.trsf_list)
        self.samples = []

        self._prepare_dataset()

    def _prepare_dataset(self):

        for class_id, class_name in enumerate(self.classes):
            step_path = os.path.join(self.root_dir, class_name, "fake", f"step={self.step}")
            for subfolder in os.listdir(step_path):
                subfolder_path = os.path.join(step_path, subfolder)
                for image_name in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image_name)
                    self.samples.append((image_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = custom_pil_loader(image_path)

        return self.trsf(image), label, image_path


if __name__ == '__main__':

    dataset_name = 'pneumoniamnist'
    reports_dir = '/home/lorenzo/GAN-Ensembles/reports/'
    source_dir = '/home/lorenzo/GAN-Ensembles/reports/'
    samples_dir = os.path.join(source_dir, dataset_name, 'samples')
    gan_aval = os.listdir(samples_dir)

    gan_steps = [
        '100000'
    ]
    gan_models = [
        'ACGAN-Mod-ADC',
        'SAGAN',
        'ReACGAN-ADA'
    ]

    # Create the path to gan exp.
    gan_folders = [x for x in gan_aval if any(y in x for y in gan_models)]

    # Initialize dataset.
    dataset = DiscDataset(root_dir=samples_dir, classes=None, step=gan_steps[0]) #

    # Dataloader.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    data_iter = iter(data_loader)

    x, y, img_paths = next(data_iter)
    print(y)
    print(img_paths)

    print("May be the force with you.")