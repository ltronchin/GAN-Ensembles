import matplotlib.pyplot as plt
import time
import copy
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import cv2

from src.general_utils.util_components import *
from src.general_utils.util_resnet import *

def get_img_autoencoder(model_name, input_dim, h_dim=None, input_channels=1):
    if model_name == "conv":
        return AutoEncoderImgConv(input_dim, h_dim, input_channels)
    elif model_name == "fully_conv":
        return AutoEncoderImgFullyConv(input_dim, input_channels)
    elif model_name == "resnet_ae_50":
        return ResNetAE(downblock=Bottleneck, upblock=DeconvBottleneck, num_layers=[3, 4, 6, 3], input_channels=input_channels)
    elif model_name == "resnet_ae_101":
        return ResNetAE(downblock=Bottleneck, upblock=DeconvBottleneck, num_layers=[3, 4, 23, 2], input_channels=input_channels)
    else:
        raise ValueError(model_name)

class AutoEncoderImgFullyConv(nn.Module):
    def __init__(self, input_dim, input_channels=1):
        super().__init__()
        # Input: N, input_channels, 256, 256
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),  # N, 16, 128, 128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N, 32, 64, 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # N, 64, 32, 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # N, 128, 16, 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # N, 256, 8, 8
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # N, 512, 4, 4
        )

        # note that output_padding is only used to find output shape, but does not actually add zero-padding to output.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # N, 256, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # N, 128, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # N, 32, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 16, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),  # N, input_channels, 256, 256
            nn.Sigmoid() # to put the reconstructed image between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderImgConv(nn.Module):
    def __init__(self, input_dim, h_dim, input_channels=1):
        super().__init__()
        # Input: N, input_channels, 256, 256
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),  # N, 16, 128, 128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N, 32, 64, 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # N, 64, 32, 32
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # N, 128, 16, 16
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # N, 256, 8, 8
            nn.Flatten(),  # N, 16384
            nn.Linear(256 * 8 * 8, h_dim)  # N, 1024
        )

        # note that output_padding is only used to find output shape, but does not actually add zero-padding to output.
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 256*8*8), # N, 16384
            nn.Unflatten(1, (256, 8, 8)), # N, 256, 8, 8
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # N, 128, 16, 16
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 32, 32
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # N, 32, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 16, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),  # N, input_channels, 256, 256
            nn.Sigmoid()  # to put the reconstructed image between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class ResNetAE(nn.Module):
    def __init__(self, downblock, upblock, num_layers, input_channels):
        super(ResNetAE, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1], stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2], stride=2)
        self.layer4 = self._make_downlayer(downblock, 128, num_layers[3], stride=6)

        self.uplayer1 = self._make_up_block(upblock, 128, num_layers[3], stride=6)
        self.uplayer2 = self._make_up_block(upblock, 64, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 32, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 16, num_layers[0], stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 64, kernel_size=1, stride=2, bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, input_channels, kernel_size=1, stride=1, bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2, kernel_size=1, stride=stride, bias=False,
                                   output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))

        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def decode(self, x, image_size):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x, output_size=image_size)
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z, x.size())

        return out
def train_autoencoder(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, warmup_epoch, model_dir, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': []}

    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for input, labels in data_loaders[phase]:
                    input = input.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(input.float())

                        loss = criterion(outputs.float(), input.float())
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * input.size(0)

                    pbar.update(input.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                if epoch > warmup_epoch:
                    if epoch_loss < best_loss:
                        best_epoch = epoch
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= early_stopping:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_best_epoch_{best_epoch}.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history

def evaluate(model, data_loader, report_dir, split, device):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    psnr = PeakSignalNoiseRatio(data_range=(-1.0,1.0), reduction='none', dim=[1,2,3])
    ssim =StructuralSimilarityIndexMeasure(data_range=(-1.0,1.0), reduction='none')
    mae = torch.nn.L1Loss(reduction='none')

    model.eval()
    data =  {'PSNR': [], 'SSIM': [], 'MAE': []}
    for inputs, _ in tqdm(data_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs.float())

            psnr_score = psnr(outputs.float(), inputs.float()).cpu().detach().numpy()
            mae_score = mae(outputs.float(), inputs.float()).cpu().detach().numpy()
            ssim_score = ssim(outputs.float(), inputs.float()).cpu().detach().numpy()
            mae_score = np.mean(mae_score, axis=(1, 2, 3))

        # Append to list.
        data['PSNR'].append(psnr_score)
        data['SSIM'].append(ssim_score)
        data['MAE'].append(mae_score)

    # Flatten the lists and create a DataFrame
    df = pd.DataFrame({
        key: [item for x in value for item in x]
        for key, value in data.items()
    })
    # Compute and add the mean and standard deviation to the DataFrame
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()

    # Save dataframe history as excel.
    df.to_excel(os.path.join(report_dir, f'results_{split}.xlsx'), index=False)

    return df

def plot_evaluate_img_autoencoder(model, data_loader, plot_dir, device):

    model.eval()
    with torch.no_grad():
        data_iter = iter(data_loader)
        inputs, _ = next(data_iter)

        inputs = inputs.to(device)
        outputs = model(inputs.float())

        for index, (x, y) in enumerate(zip(inputs, outputs)):

            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            # From -1,1 to 0,255.
            x = ((x + 1) / 2) * 255
            y = ((y + 1) / 2) * 255

            # Clamp values outside 0 255.
            x = np.clip(x, 0, 255).astype(np.uint8)
            y = np.clip(y, 0, 255).astype(np.uint8)

            if x.shape[0] == 3:
                x = x.transpose(1, 2, 0)
                y = y.transpose(1, 2, 0)
            elif x.shape[0] == 1:
                x = np.squeeze(x, axis=0)
                y = np.squeeze(y, axis=0)

            xy = np.hstack((x, y))
            if x.shape[-1] == 3:
                cv2.imwrite(os.path.join(plot_dir, "{}.png".format(str(index))), cv2.cvtColor(xy, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(os.path.join(plot_dir, "{}.png".format(str(index))), xy)


def plot_training(history, plot_training_dir):

    # Training results Loss function
    if 'train_loss' in history.columns and 'val_loss' in history.columns:
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'val_loss']:
            plt.plot(history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        # Delete white space
        plt.tight_layout()
        plt.savefig(os.path.join(plot_training_dir, "Loss.pdf"),  dpi=400, format='pdf')
        plt.show()

    # Training results Accuracy
    if 'train_acc' in history.columns and 'val_acc' in history.columns:
        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'val_acc']:
            plt.plot(100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_training_dir, "Acc.pdf"),  dpi=400, format='pdf')
        plt.show()