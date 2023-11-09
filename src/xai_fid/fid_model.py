import sys
sys.path.extend([
    "./",
])
import torch
import torch.nn as nn


def compute_fid(features, num_images, mu_real, mu_synth, sigma_real, sigma_synth):
    # Compute updated mean and covariance, when reference image is added to set of gen. images.
    mu_synth_prime = ((num_images - 1) / num_images) * mu_synth + (1 / num_images) * features
    sigma_synth_prime = ((num_images - 2) / (num_images - 1)) * sigma_synth + (1 / num_images) * torch.mm((features - mu_synth).T, (features - mu_synth))

    # Compute the fid
    eps = 1e-12
    mean_term = torch.sum(torch.square(mu_real - mu_synth_prime.squeeze(0)))
    eigenvalues, _ = torch.linalg.eig(torch.matmul(sigma_synth_prime, sigma_real))  # Eigenvalues shape: (D, 2) (real and imaginary parts).
    cov_term = torch.trace(sigma_real) + torch.trace(sigma_synth_prime) - 2 * torch.sum(torch.sqrt(eigenvalues + eps))
    wasserstein2 = mean_term + cov_term
    wasserstein2 = torch.real(wasserstein2)
    return wasserstein2.unsqueeze(0)


class FIDModel(nn.Module):
    def __init__(self, feature_extractor, num_images, mu_real, mu_synth, sigma_real, sigma_synth):
        super(FIDModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_images = num_images
        self.mu_real = mu_real
        self.mu_synth = mu_synth
        self.sigma_real = sigma_real
        self.sigma_synth = sigma_synth

    def forward(self, x):
        # Extract features
        features, _ = self.feature_extractor.get_outputs(x, quantize=True)
        # Compute FID
        fid = compute_fid(features=features, num_images=self.num_images, mu_real=self.mu_real, mu_synth=self.mu_synth, sigma_real=self.sigma_real, sigma_synth=self.sigma_synth)

        return fid



