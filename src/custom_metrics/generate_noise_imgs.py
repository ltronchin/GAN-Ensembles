import sys
sys.path.extend([
    "./",
])
import os
import torch
import argparse
import  yaml
import numpy as np

from src.data_util import Dataset_


def generate_noise_images(num_images, img_size, img_type, n_channels, min_val, max_val, save_path):
    """
    Generate a set of random noise images with specified properties and save as NumPy arrays.

    Parameters:
    - num_images: The number of images to generate.
    - img_size: Tuple of (width, height) for the desired image dimensions.
    - img_type: Data type of the image (like 'float32').
    - n_channels: Number of image channels (e.g., 3 for RGB).
    - min_val: Minimum pixel value.
    - max_val: Maximum pixel value.
    - save_path: Directory where the generated images will be saved.

    Returns:
    - A list of file paths for the saved images.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    saved_files = []

    for i in range(num_images):
        # Create random noise
        noise = np.random.uniform(min_val, max_val, (img_size[1], img_size[0], n_channels)).astype(img_type)
        # Channel first.
        noise = np.transpose(noise, (2, 0, 1))

        # Save the noise as a numpy array
        file_name = os.path.join(save_path, f"noise_{i}.npy")
        np.save(file_name, noise)
        saved_files.append(file_name)

    return saved_files

def get_parser():

    parser = argparse.ArgumentParser(description='Compute features.')
    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/")
    parser.add_argument('--source_dir', type=str, default='./reports/pneumoniamnist', help='Directory name to fake samples.') #/home/lorenzo/GAN-Ensembles/reports/
    parser.add_argument('--dataset_name', type=str, default='pneumoniamnist', choices=['pneumoniamnist', 'retinamnist', 'breastmnist'],  help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])

    return parser

# main
if __name__ == '__main__':

    parser = get_parser()
    args, unknown = parser.parse_known_args()

    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Directories.
    split= args.split
    source_dir = args.source_dir
    dataset_name = args.dataset_name
    samples_dir = os.path.join(source_dir, 'samples')

    # Parameters.
    batch_size = args.batch_size

    # Device.
    gpu_ids = int(args.gpu_ids)
    if gpu_ids >= 0:
        # Check if available.
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(gpu_ids))
    else:
        device = torch.device('cpu')


    real_dataset = Dataset_(data_name=cfg['DATA']['name'],
                            data_dir=cfg['DATA']['data_dir'],
                            train=True if split == 'train' else False,
                            split = split,
                            crop_long_edge=cfg['PRE']['crop_long_edge'],
                            resize_size=cfg['PRE']['resize_size'],
                            resizer=cfg['PRE']['pre_resizer'],
                            random_flip=cfg['PRE']['apply_rflip'],
                            normalize=cfg['PRE']['normalize'],
                            cfgs=cfg)

    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    # Iterator.
    real_iter = iter(real_loader)

    # Get image shapes and types.
    real_images, real_labels = next(real_iter)
    img = real_images[0].cpu().detach().numpy()
    n_channels = img.shape[0]
    img_size = img.shape[1]
    img_type = img.dtype

    filename = f'{dataset_name}_{split}-img_size_{img_size}-img_type_{img_type}-range_11-dummy_dataset'

    # Example usage:
    generate_noise_images(num_images=real_dataset.__len__(), img_size=(img_size, img_size), img_type=img_type, n_channels=n_channels, min_val=-1, max_val=1, save_path=os.path.join(samples_dir, filename, 'noise'))

    print("May the force be with you!")