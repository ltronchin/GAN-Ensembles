import copy
import sys
sys.path.extend([
    "./",
])
from itertools import product

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from src.general_utils import util_path
from src.general_utils import util_toy


# For aspect ratio 4:3.
sns.set_context("paper")
sns.set_theme(style="ticks")

# 1. Define the Generator and Discriminator
# Generator Network
class Generator(nn.Module):
    def __init__(self, out_dim=2, hidden_dim=128, n_layers=3):
        super(Generator, self).__init__()

        layers = [
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, n_layers=3):
        super(Discriminator, self).__init__()

        # Input layer
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)]

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':

    report_dir = './reports/'
    # Parameters for the data
    grid_size = 5
    num_samples = 100000
    step = 3 # 2
    mean_coordinates = [(4 + step * i, 4 + step * j) for i in range(grid_size) for j in range(grid_size)]
    std_dev = 0.05
    dataset =util_toy.generate_data(num_samples, mean_coordinates, std_dev)
    noise_synthetic = torch.randn(2000, 2)
    real_samples = copy.deepcopy(dataset)

    exp_name = f'toy_gans_num_real_samples__{num_samples}-grid_size__{grid_size}-step__{step}-std_dev__{std_dev}'
    report_dir = os.path.join(report_dir, exp_name)
    util_path.create_dir(report_dir)

    # Save noise_synthetic.
    np.save(os.path.join(report_dir, 'noise_synthetic.npy'), noise_synthetic)

    # Save real samples.
    np.save(os.path.join(report_dir, 'real_samples.npy'), real_samples)

    # Plot the grid
    fig = plt.figure()
    plt.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=1, color='blue')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Real Data")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(report_dir, 'real_data.png'))
    plt.show()

    dataset = torch.from_numpy(dataset).float()

    # Define parameters for GAN.
    checkpoints_interval = 40
    batch_size = 256
    beta1 = 0.5
    lr_list = [0.0002, 0.0001]
    num_epochs_list = [200]
    hidden_dim_list = [128, 256]
    netG_layers_list = [4, 5]
    netD_layers_list = [4, 5]

    opts = product(lr_list, num_epochs_list, hidden_dim_list, netG_layers_list, netD_layers_list)

    for lr, num_epochs, hidden_dim, netG_layers, netD_layers in opts:
        print("\n")
        print('lr: ', lr)
        print('num_epochs: ', num_epochs)
        print('hidden_dim: ', hidden_dim)
        print('netG_layers: ', netG_layers)
        print('netD_layers: ', netD_layers)

        filename = f'lr__{lr}-num_epochs__{num_epochs}-hidden_dim__{hidden_dim}-netG_layers__{netG_layers}-netD_layers__{netD_layers}'
        util_path.create_dir(os.path.join(report_dir, filename))

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Convert the dataset to PyTorch tensors
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Specify the loss function and optimizers
        netG = Generator(hidden_dim=hidden_dim, n_layers=netG_layers).to(device)
        netD = Discriminator(hidden_dim=hidden_dim, n_layers=netD_layers).to(device)

        netG.apply(weights_init)
        netD.apply(weights_init)

        criterion = nn.BCELoss()
        optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # 3. Train the GAN
        D_losses_epoch = []
        G_losses_epoch = []
        # real_data_iter = iter(train_loader)
        # real_data = next(real_data_iter)
        for epoch in range(num_epochs):
            D_losses = []
            G_losses = []
            for real_data in train_loader:

                # Train Discriminator
                netD.zero_grad()
                real_data = real_data.to(device)
                b_size = real_data.size(0)
                # Real data
                label_real = torch.ones(b_size, 1).to(device)
                output_real = netD(real_data)
                loss_real = criterion(output_real, label_real)

                # Generated data
                noise = torch.randn(b_size, 2).to(device)
                fake_data = netG(noise)
                label_fake = torch.zeros(b_size, 1).to(device)
                output_fake = netD(fake_data.detach())
                loss_fake = criterion(output_fake, label_fake)

                # Total loss and update
                loss_D = loss_real + loss_fake
                loss_D.backward()
                optimizer_D.step()

                # Train Generator
                netG.zero_grad()
                output_fake_G = netD(fake_data)
                loss_G = criterion(output_fake_G, label_real)
                loss_G.backward()
                optimizer_G.step()

                D_losses.append(loss_D.item())
                G_losses.append(loss_G.item())

            D_losses_epoch.append(np.mean(D_losses))
            G_losses_epoch.append(np.mean(G_losses))

            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, D Loss: {D_losses_epoch[-1]}, G Loss: {G_losses_epoch[-1]}")

            if epoch % checkpoints_interval == 0 and epoch != 0:
                # Save networks.
                torch.save(netG.state_dict(), os.path.join(report_dir, filename, f'netG_epoch_{epoch}.pth'))
                torch.save(netD.state_dict(), os.path.join(report_dir, filename, f'netD_epoch_{epoch}.pth'))

                with torch.no_grad():
                    generated_samples = netG(noise_synthetic.to(device)).cpu().numpy()

                # Save samples.
                np.save(os.path.join(report_dir, filename, f'generated_samples_epoch_{epoch}.npy'), generated_samples)

                fig = plt.figure()
                plt.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, color='blue', label="Real Data")
                plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, color='red', label="Generated Data")
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                fig.savefig(os.path.join(report_dir, filename, f'samples_epoch_{epoch}.png'))
                plt.show()

        # Final checkpoint.
        torch.save(netG.state_dict(), os.path.join(report_dir, filename, f'netG_epoch_final.pth'))
        torch.save(netD.state_dict(), os.path.join(report_dir, filename, f'netD_epoch_final.pth'))

        with torch.no_grad():
            generated_samples = netG(noise_synthetic.to(device)).cpu().numpy()

        np.save(os.path.join(report_dir, filename, f'generated_samples_epoch_final.npy'), generated_samples)

        fig = plt.figure()
        plt.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, color='blue', label="Real Data")
        plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, color='red', label="Generated Data")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(report_dir, filename, f'samples_epoch_final.png'))
        plt.show()

        # Plot final loss curve.
        fig = plt.figure(figsize=(10, 8))
        plt.plot(D_losses_epoch, label='Discriminator')
        plt.plot(G_losses_epoch, label='Generator')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        fig.savefig(os.path.join(report_dir, filename, f'loss_curve.png'))
        plt.show()

    print('May the force be with you.')