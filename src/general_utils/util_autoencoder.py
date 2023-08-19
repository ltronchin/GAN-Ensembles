import matplotlib.pyplot as plt
import time
import copy
import pandas as pd
from tqdm import tqdm

from src.general_utils.util_components import *
from src.general_utils.util_resnet import *

def get_img_autoencoder(model_name, input_dim, h_dim=None, n_classes=1):
    if model_name == "conv":
        return AutoEncoderImgConv(input_dim, h_dim, n_classes)
    elif model_name == "fully_conv":
        return AutoEncoderImgFullyConv(input_dim, n_classes)
    elif model_name == "resnet_ae_50":
        return ResNetAE(downblock=Bottleneck, upblock=DeconvBottleneck, num_layers=[3, 4, 6, 3], n_classes=n_classes)
    elif model_name == "resnet_ae_101":
        return ResNetAE(downblock=Bottleneck, upblock=DeconvBottleneck, num_layers=[3, 4, 23, 2], n_classes=n_classes)
    else:
        raise ValueError(model_name)

class AutoEncoderImgFullyConv(nn.Module):
    def __init__(self, input_dim, n_classes=1):
        super().__init__()
        # Input: N, n_classes, 256, 256
        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 16, 3, stride=2, padding=1),  # N, 16, 128, 128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N, 32, 64, 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # N, 64, 32, 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # N, 128, 16, 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # N, 256, 8, 8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # N, 512, 4, 4
        )

        # note that output_padding is only used to find output shape, but does not actually add zero-padding to output.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # N, 256, 8, 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # N, 128, 16, 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 32, 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # N, 32, 56, 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 16, 128, 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, n_classes, 3, stride=2, padding=1, output_padding=1),  # N, n_classes, 256, 256
            nn.Sigmoid() # to put the reconstructed image between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderImgConv(nn.Module):
    def __init__(self, input_dim, h_dim, n_classes=1):
        super().__init__()
        # Input: N, n_classes, 256, 256
        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 16, 3, stride=2, padding=1),  # N, 16, 128, 128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N, 32, 64, 64
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 16, 128, 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, n_classes, 3, stride=2, padding=1, output_padding=1),  # N, n_classes, 256, 256
            nn.Sigmoid()  # to put the reconstructed image between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class ResNetAE(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNetAE, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

        self.conv1_1 = nn.ConvTranspose2d(64, n_classes, kernel_size=1, stride=1, bias=False)

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

    def decode(self, x, image_size=[1, 1, 224, 224]):
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
def train_autoencoder(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, model_dir, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': []}

    epochs_no_improve = 0
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

def plot_evaluate_img_autoencoder(model, data_loader, plot_dir, device):
    # Test loop
    model.eval()
    with torch.no_grad():
        # Sample one batch
        data_iter = iter(data_loader)
        inputs, labels = data_iter.next()

        inputs = inputs.to(device)
        labels = labels.to(device)
        # Prediction
        outputs = model(inputs.float())
        # Plot
        for index, (input, output) in enumerate(zip(inputs, outputs)):

            input = input.cpu().detach().numpy()[0]
            output = output.cpu().detach().numpy()[0]

            plt.figure()
            plt.gray()
            plt.subplot(1, 2, 1)
            plt.imshow(input)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(output)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "{}.png".format(str(index))), dpi=300,  bbox_inches='tight')
            plt.show()

def plot_evaluate_img_autoencoder_PIL(model, data_loader, plot_dir, device):
    # Test loop
    model.eval()
    with torch.no_grad():
        # Sample one batch
        data_iter = iter(data_loader)
        inputs, labels = data_iter.next()

        inputs = inputs.to(device)
        labels = labels.to(device)
        # Prediction
        outputs = model(inputs.float())
        # Plot
        for index, (input, output) in enumerate(zip(inputs, outputs)):

            input = input.cpu().detach().numpy()[0]
            output = output.cpu().detach().numpy()[0]

            plt.figure()
            plt.gray()
            plt.subplot(1, 2, 1)
            plt.imshow(input)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(output)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "{}.png".format(str(index))), dpi=300,  bbox_inches='tight')
            plt.show()
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