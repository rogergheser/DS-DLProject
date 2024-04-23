import torch
from torchvision import transforms
import numpy as np


class ToUint8Transform:
    """Transform to convert images to uint8"""
    def __call__(self, tensor):
        return (tensor.mul(255)).byte()  # Use .byte() to convert to uint8


class AugMix(torch.nn.Module):
    def __init__(self, severity=3, width=3, depth=-1, alpha=1.):
        super(AugMix, self).__init__()
        self.severity = severity
        self.width = width
        self.depth = depth if depth > 0 else np.random.randint(1, 3)
        self.alpha = alpha
        self.augmentations = [transforms.ColorJitter(0.8*self.severity, 0.8*self.severity, 0.8*self.severity, 0.2*self.severity),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(size=224, padding=int(224*0.125), pad_if_needed=True)]