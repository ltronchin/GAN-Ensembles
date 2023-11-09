
import os

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np


dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    }
}

def build_resizer(resizer, backbone, size):
    if resizer == "friendly":
        if backbone == "InceptionV3_tf":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "InceptionV3_torch":
            return make_resizer("PIL", "lanczos", (size, size))
        elif backbone in ["InceptionV3_torch__medical", "InceptionV3_torch__truefake"]:
            return make_resizer("PIL", "lanczos", (size, size))
        elif backbone == "ResNet50_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone in ["ResNet50_torch__medical", "ResNet50_torch__truefake"]:
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "SwAV_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        else:
            raise ValueError(f"Invalid resizer {resizer} specified")
    elif resizer == "clean":
        return make_resizer("PIL", "bicubic", (size, size))
    elif resizer == "legacy":
        return make_resizer("PyTorch", "bilinear", (size, size))


def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)
        def func(x):
            # x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = [resize_single_channel(x[:, :, idx]) for idx in range(x.shape[2])]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func
