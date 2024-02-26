"""
Utilities provided by Aasa, from her course on advanced medical image analysis.
"""

import numpy as np
import random
import torch


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def im_normalize(im: torch.Tensor) -> torch.Tensor:
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def tens2image(im: torch.Tensor) -> torch.Tensor:
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))
