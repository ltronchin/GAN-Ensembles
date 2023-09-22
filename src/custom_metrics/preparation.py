# src/metrics/preparation.py

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np
import torch
import torchvision.transforms as transforms

import src.custom_metrics.resize as resize

model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                  "ResNet50_torch": "pytorch/vision:v0.10.0",
                  "SwAV_torch": "facebookresearch/swav:main"}

model_names = {"InceptionV3_torch": "inception_v3",
               "ResNet50_torch": "resnet50",
               "SwAV_torch": "resnet50"}

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
    def __init__(self, eval_backbone, post_resizer, device):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.post_resizer = post_resizer
        self.device = device
        self.save_output = SaveOutput()

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
        elif self.eval_backbone == 'resnet_ae_50_pneumoniamnist':
            from src.general_utils import util_autoencoder
            import os
            source_dir = "./reports/pneumoniamnist/backbone"
            self.res = 32
            input_dim = (1, 32, 32)
            self.model = util_autoencoder.get_img_autoencoder(model_name='resnet_ae_50', input_dim=input_dim, h_dim=None, n_classes=1)
            self.model.load_state_dict(torch.load(os.path.join(source_dir, "pneumoniamnist-resnet_ae_50-train-2023_08_06_07_26_06", "model_best_epoch_17.pt"),map_location=self.device ))
            self.model = self.model.to(self.device)
        elif self.eval_backbone == 'disc_resnet_50_pneumoniamnist':
            from src.cnn_models.models import ResNet50
            import os
            source_dir = "./reports/pneumoniamnist/backbone"
            self.res = 32
            input_dim = (1, 32, 32)
            n_classes = 111
            self.model = ResNet50(in_channels=input_dim[0], num_classes=n_classes)
            self.model.load_state_dict(torch.load(os.path.join(source_dir, "pneumoniamnist-disc_resnet_50-train-2023_08_08_11_32_43", "model_best_epoch_2.pt"),map_location=self.device))
            self.model = self.model.to(self.device)
        elif self.eval_backbone == 'cnn_resnet_50_pneumoniamnist':
            from src.cnn_models.models import ResNet50
            import os
            source_dir = "./reports/pneumoniamnist/backbone"
            self.res = 32
            input_dim = (1, 32, 32)
            n_classes = 2
            self.model = ResNet50(in_channels=input_dim[0], num_classes=n_classes)
            self.model.load_state_dict(torch.load(os.path.join(source_dir, "pneumoniamnist-cnn_resnet_50-train-2023_08_16_15_35_41",  "model_best_epoch_5.pt"),map_location=self.device ))
            self.model = self.model.to(self.device)
        else:
            raise NotImplementedError

    def eval(self):
        self.model.eval()

    def get_outputs(self, x, quantize=False):

        if self.eval_backbone in "resnet_ae_50_pneumoniamnist":
            repres = self.model.encode(x)  # BATCH_SIZE x n_feat x 1 x 1
            repres = repres.view(repres.size(0), -1) # BATCH_SIZE x n_feat
            return repres, None
        elif self.eval_backbone == 'disc_resnet_50_pneumoniamnist':
            repres = self.model.extract_features(x)
            return repres, None
        elif self.eval_backbone == 'cnn_resnet_50_pneumoniamnist':
            repres = self.model.extract_features(x)
            return repres, None

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