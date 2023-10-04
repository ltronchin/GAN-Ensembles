import sys
sys.path.extend([
    "./",
])
import os
import torch
import argparse
import yaml

from src.general_utils import util_data
from src.general_utils import util_general
import src.custom_metrics.preparation as pp
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def get_parser():

    parser = argparse.ArgumentParser(description='Compute features.')

    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs_features/pneumoniamnist/feat_bin_real.yaml")
    parser.add_argument('--source_dir', type=str, default='/Users/valerioguarrasi/Downloads/pneumoniamnist/pneumoniamnist', help='Directory name to fake samples.') #todo: adapt to your path
    parser.add_argument('--dataset_name', type=str, default='pneumoniamnist', choices=['pneumoniamnist', 'retinamnist', 'breastmnist'],  help='The name of dataset')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  use -1 for CPU')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size')
    parser.add_argument('--gan_models', type=util_general.parse_comma_separated_list, default='ACGAN-Mod-ADC,SAGAN,ReACGAN-ADA', help='List of GANs to enable in the ensemble')
    parser.add_argument('--gan_steps', type=util_general.parse_comma_separated_list, default='100000', help='Iter or Iters to sample each GAN')
    parser.add_argument("--post_resizer", type=str, default="friendly", help="which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='InceptionV3_torch', help="InceptionV3_torch, SwAV_torch, resnet_ae_50_pneumoniamnist, disc_resnet_50_pneumoniamnist, cnn_resnet50_pneumoniamnist, resnet_ae_50_retinamnist, disc_resnet_50_retinamnist, cnn_resnet50_retinamnist, resnet_ae_50_breastmnist, disc_resnet_50_breastmnist, cnn_resnet50_breastmnist ")
    return parser


# main
if __name__ == '__main__':

    parser = get_parser()
    args, unknown = parser.parse_known_args()

    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Directories.
    source_dir = args.source_dir
    dataset_name = args.dataset_name
    samples_dir = os.path.join(source_dir, 'samples')
    cache_dir = os.path.join(source_dir, 'features')

    # Parameters.
    eval_backbone = args.eval_backbone
    post_resizer = args.post_resizer
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

    # GANs
    gan_models = args.gan_models
    gan_steps = args.gan_steps
    gan_aval = os.listdir(samples_dir)
    filename = f'gradcam__{eval_backbone}_{post_resizer}'

    # Create the entire path for each gan.
    gan_folders = [os.path.join(samples_dir, x, 'fake') for x in gan_aval if any(y in x for y in gan_models)]
    gan_folders = [os.path.join(x, f"step={y}") for x in gan_folders for y in gan_steps]

    eval_model = pp.LoadEvalModel(
        eval_backbone=eval_backbone,
        post_resizer=post_resizer,
        device=device
    )

    # Dataset
    dataset = util_data.EnsembleDataset(folders=[gan_folders[0]], weights=[1.0])
    num_batches = 1#len(dataset) // batch_size

    # Dataloder
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load fixed statistics.
    # mu_real
    # mu_synth
    # sigma_real
    # sigma_synth

    eval_model.eval()
    data_iter = iter(dataloader)
    for _ in range(0, num_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break

        images, labels = images.to(device), labels.to(device)
        images = torch.clamp(images, -1, 1) # Clip between -1 and 1

        features, _ = eval_model.get_outputs(images, quantize=True)

        # todo Spatial average

        # Update statistics.
        # todo N, mu_real, mu_synth, sigma_real, sigma_synth
        #sigma_synt_prime
        #mu_synth_prime
        # Compute updated mean and covariance, when reference image is added to set of gen. images.
        mean = ((num_images - 1) / num_images) * mean_gen + (1 / num_images) * features
        cov = ((num_images - 2) / (num_images - 1)) * cov_gen + \
              (1 / num_images) * torch.mm((features - mean_gen).T, (features - mean_gen))

        # Compute the fid.
        # todo transform to differentiable
        mean_term = torch.sum(torch.square(mean_reals - mean_gen.squeeze(0)))
        eigenvalues, _ = torch.eig(torch.matmul(cov_gen, cov_reals),
                                   eigenvectors=True)  # Eigenvalues shape: (D, 2) (real and imaginary parts).
        cov_term = torch.trace(cov_reals) + torch.trace(cov_gen) - 2 * torch.sum(torch.sqrt(eigenvalues[:, 0] + eps))
        wasserstein2 = mean_term + cov_term

        # Todo gradcam


    print("May the force be with you.")



