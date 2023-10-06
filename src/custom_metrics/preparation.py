# src/metrics/preparation.py

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np
import os
import torch
import torchvision.transforms as transforms

from src.general_utils import util_autoencoder
from src.cnn_models.models import ResNet50
import src.custom_metrics.resize as resize

model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                  "ResNet50_torch": "pytorch/vision:v0.10.0",
                  "SwAV_torch": "facebookresearch/swav:main"}

model_names = {"InceptionV3_torch": "inception_v3",
               "ResNet50_torch": "resnet50",
               "SwAV_torch": "resnet50"}

CUSTOM_BACKBONE_CONFIG = {
    # pneumoniamnist
    'resnet_ae_50_pneumoniamnist': {
        'source_dir': "./reports/pneumoniamnist/backbone",
        'model_name': 'resnet_ae_50',
        'res': 32,
        'input_dim': (1, 32, 32),
        'n_classes': None,
        'channels': 1,
        'model_path': "pneumoniamnist-resnet_ae_50-train-2023_08_06_07_26_06/model_best_epoch_17.pt"
    },
    'disc_resnet_50_pneumoniamnist': {
        'source_dir': "./reports/pneumoniamnist/backbone",
        'model_name': 'disc_resnet_50',
        'res': 32,
        'input_dim': (1, 32, 32),
        'channels': 1,
        'n_classes': 111,
        'model_path': "pneumoniamnist-disc_resnet_50-train-2023_08_08_11_32_43/model_best_epoch_2.pt"
    },
    'cnn_resnet_50_pneumoniamnist': {
        'source_dir': "./reports/pneumoniamnist/backbone",
        'model_name': 'cnn_resnet_50',
        'res': 32,
        'input_dim': (1, 32, 32),
        'channels': 1,
        'n_classes': 2,
        'model_path': "pneumoniamnist-cnn_resnet_50-train-2023_08_16_15_35_41/model_best_epoch_5.pt"
    },
    # retinamnist
    'resnet_ae_50_retinamnist': {
        'source_dir': "./reports/retinamnist/backbone",
        'model_name': 'resnet_ae_50',
        'res': 32,
        'input_dim': (3, 32, 32),
        'n_classes': None,
        'channels': 3,
        'model_path': "retinamnist-resnet_ae_50-train-2023_09_29_09_00_44/model_best_epoch_99.pt"
    },
    'disc_resnet_50_retinamnist': {
        'source_dir': "./reports/retinamnist/backbone",
        'model_name': 'disc_resnet_50',
        'res': 32,
        'input_dim': (3, 32, 32),
        'channels': 3,
        'n_classes': 111,
        'model_path': "retinamnist-disc_resnet_50-train-2023_09_29_08_45_23/model_best_epoch_29.pt"
    },
    'cnn_resnet_50_retinamnist': {
        'source_dir': "./reports/retinamnist/backbone",
        'model_name': 'cnn_resnet_50',
        'res': 32,
        'input_dim': (3, 32, 32),
        'channels': 3,
        'n_classes': 5,
        'model_path': "retinamnist-cnn_resnet_50-train-2023_09_29_08_54_25/model_best_epoch_27.pt"
    },
    # breastmnist
    'resnet_ae_50_breastmnist': {
        'source_dir': "./reports/breastmnist/backbone",
        'model_name': 'resnet_ae_50',
        'res': 32,
        'input_dim': (1, 32, 32),
        'n_classes': None,
        'channels': 1,
        'model_path': "breastmnist-resnet_ae_50-train-2023_09_29_09_00_46/model_best_epoch_97.pt"
    },
    'disc_resnet_50_breastmnist': {
        'source_dir': "./reports/breastmnist/backbone",
        'model_name': 'disc_resnet_50',
        'res': 32,
        'input_dim': (1, 32, 32),
        'channels': 1,
        'n_classes': 111,
        'model_path': "breastmnist-disc_resnet_50-train-2023_09_29_09_00_48/model_best_epoch_41.pt"
    },
    'cnn_resnet_50_breastmnist': {
        'source_dir': "./reports/breastmnist/backbone",
        'model_name': 'cnn_resnet_50',
        'res': 32,
        'input_dim': (1, 32, 32),
        'channels': 1,
        'n_classes': 2,
        'model_path': "breastmnist-cnn_resnet_50-train-2023_09_29_08_45_34/model_best_epoch_49.pt"
    }
}

SWAV_CLASSIFIER_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar"

# Metrics utils
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []

def quantize_images(x):
    x = (x + 1)/2
    x = (255.0 * x + 0.5).clamp(0.0, 255.0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x


def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.transpose((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x

class LoadEvalModel(object):
    def __init__(self, eval_backbone, post_resizer, device, preprocessing=True):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.post_resizer = post_resizer
        self.device = device
        self.save_output = SaveOutput()
        self.preprocessing = preprocessing

        if self.eval_backbone in ["InceptionV3_torch", "ResNet50_torch", "SwAV_torch"]:
            self.res = 299 if "InceptionV3" in self.eval_backbone else 224
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

            try:
                self.model = torch.hub.load(model_versions[self.eval_backbone], model_names[self.eval_backbone], pretrained=True)
            except ModuleNotFoundError: # https://pytorch.org/docs/stable/hub.html # custom code to manage cache error
                if self.eval_backbone == "SwAV_torch":
                    self.model = torch.hub.load("swav_main",  model_names[self.eval_backbone], pretrained=True, source='local')
                else:
                    raise ModuleNotFoundError

            if self.eval_backbone == "SwAV_torch":
                linear_state_dict = load_state_dict_from_url(SWAV_CLASSIFIER_URL, progress=True, map_location=self.device)["state_dict"]
                linear_state_dict = {k.replace("module.linear.", ""): v for k, v in linear_state_dict.items()}
                self.model.fc.load_state_dict(linear_state_dict, strict=True)
            self.model = self.model.to(self.device)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
            self.resizer = resize.build_resizer(resizer=self.post_resizer, backbone=self.eval_backbone, size=self.res)
            self.totensor = transforms.ToTensor()
            self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
            self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)

        else:
            config = CUSTOM_BACKBONE_CONFIG.get(self.eval_backbone)
            if config is None:
                raise NotImplementedError
            self.res = config['res']
            model_name = config['model_name']
            if model_name in ['disc_resnet_50', 'cnn_resnet_50']:
                self.model = ResNet50(input_channels=config['channels'], num_classes=config['n_classes'])
                self.model.load_state_dict(torch.load(os.path.join(config['source_dir'], config['model_path']), map_location=self.device))
            elif model_name in ['resnet_ae_50']:
                self.model = util_autoencoder.get_img_autoencoder(model_name=model_name, input_dim=config['input_dim'], h_dim=None, input_channels=config['channels'])
                self.model.load_state_dict(torch.load(os.path.join(config['source_dir'], config['model_path']), map_location=self.device))
            else:
                raise NotImplementedError
            self.model = self.model.to(self.device)

    def eval(self):
        self.model.eval()

    def get_outputs(self, x, quantize=False):

        # Get keys from dict.
        custom_backbones_keys = list(CUSTOM_BACKBONE_CONFIG.keys())

        if self.eval_backbone in custom_backbones_keys:
            config = CUSTOM_BACKBONE_CONFIG.get(self.eval_backbone)
            if config['model_name'] in ['resnet_ae_50']:
                repres = self.model.encode(x)  # BATCH_SIZE x n_feat x 1 x 1
                repres = repres.view(repres.size(0), -1) # BATCH_SIZE x n_feat
                return repres, None
            else:
                repres = self.model.extract_features(x)
                return repres, None

        if self.preprocessing:
            if x.shape[1] != 3:
                x = x.repeat(1, 3, 1, 1)  # grayscale to RGB

            if quantize:
                x = quantize_images(x)
            else:
                x = x.detach().cpu().numpy().astype(np.uint8)
            x = resize_images(x, self.resizer, self.totensor, self.mean, self.std, device=self.device)

        if self.eval_backbone in ["InceptionV3_torch", "ResNet50_torch", "SwAV_torch"]:
            logits = self.model(x)
            if len(self.save_output.outputs) > 1:
                repres = []
                for rank in range(len(self.save_output.outputs)):
                    repres.append(self.save_output.outputs[rank][0].detach().cpu())
                repres = torch.cat(repres, dim=0).to(self.device)
            else:
                repres = self.save_output.outputs[0][0].to(self.device)
            self.save_output.clear()

        else:
            raise NotImplementedError

        return repres, logits # repres -> BATCH_SIZE x 2048, tensor