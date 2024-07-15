import torch
import torchvision.datasets as datasets
from torchvision.transforms import v2
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import py_vars
import random
from augmix import augmentations, post_augmentations
from PIL import Image


def load_imagenet_A(path:str, batch_size:int, preprocess:transforms.Compose, shuffle:bool=True, percentage:float=1.0):
    """
    Loads the images from the ImageNet-A dataset
    :param path: str: path to the ImageNet-A dataset
    :param batch_size: int: batch size
    :param preprocess: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    """
    imagenet_A = datasets.ImageFolder(root=path, transform=preprocess)
    if percentage < 1.0:
        imagenet_A_loader = torch.utils.data.DataLoader(imagenet_A, batch_size=batch_size, shuffle=shuffle, sampler=torch.utils.data.SubsetRandomSampler(np.random.choice(len(imagenet_A), int(percentage * len(imagenet_A)), replace=False)))
    else:
        imagenet_A_loader = torch.utils.data.DataLoader(imagenet_A, batch_size=batch_size, shuffle=shuffle)

    id2class = {imagenet_A.class_to_idx[c] : py_vars.num2class[c] for c in imagenet_A.classes}
    return imagenet_A_loader, id2class

def load_imagenet_v2(path:str, batch_size:int, preprocess:transforms.Compose, shuffle:bool=True, percentage:float=1.0):
    """
    Loads the images from the ImageNet-V2 dataset
    :param path: str: path to the ImageNet-V2 dataset
    :param batch_size: int: batch size
    :param preprocess: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    """
    imagenet_v2 = datasets.ImageFolder(root=path, transform=preprocess)
    
    if percentage < 1.0:
        imagenet_v2_loader = torch.utils.data.DataLoader(imagenet_v2, batch_size=batch_size, shuffle=shuffle, sampler=torch.utils.data.SubsetRandomSampler(np.random.choice(len(imagenet_v2), int(percentage * len(imagenet_v2)), replace=False)))
    else:
        imagenet_v2_loader = torch.utils.data.DataLoader(imagenet_v2, batch_size=batch_size, shuffle=shuffle)

    id2class = {imagenet_v2.class_to_idx[c] : py_vars.num2class_v2[int(c)] for c in imagenet_v2.classes}
    return imagenet_v2_loader, id2class

def load_cifar100(path:str, batch_size:int, preprocess:transforms.Compose, shuffle:bool=True, percentage:float=1.0):
    """
    Loads the images from the CIFAR-100 dataset
    :param path: str: path to the CIFAR-100 dataset
    :param batch_size: int: batch size
    :param preprocess: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    """
    cifar100 = datasets.CIFAR100(root=path, download=True, transform=preprocess)
    if percentage < 1.0:
        cifar100_loader = torch.utils.data.DataLoader(cifar100, batch_size=batch_size, shuffle=shuffle, sampler=torch.utils.data.SubsetRandomSampler(np.random.choice(len(cifar100), int(percentage * len(cifar100)), replace=False)))
    else:
        cifar100_loader = torch.utils.data.DataLoader(cifar100, batch_size=batch_size, shuffle=shuffle)
    id2class = {cifar100.class_to_idx[c] : c for c in cifar100.classes}
    return cifar100_loader, id2class

# class Augmixer():
#     def __init__(self, preprocess:transforms.Compose, batch_size:int=64, severity: int = 3, mixture_width: int = 5, chain_depth: int = 1, alpha: float = 0.3):
#         self.preprocess = preprocess
#         self.batch_size = batch_size
#         self.augmenter = v2.AugMix(severity=severity, mixture_width=mixture_width, chain_depth=chain_depth, alpha=alpha)
    
#     def __call__(self, img):
#         # img = self.preprocess(img)
#         if torch.is_tensor(img):
#             if not img.dtype == torch.uint8:
#                 img = img.mul(255).byte()
#             assert img.dtype == torch.uint8, "Image must be of type uint8"            

#         augmentations = [self.preprocess(self.augmenter(img)) for _ in range(self.batch_size-1)]
#         res = [self.preprocess(img)] + augmentations
#         return torch.stack(res).squeeze(0)

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def _augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix

class Augmixer(object):
    def __init__(self, preprocess, n_views=64, augmix=False, 
                    severity=1):
        self.preprocess = preprocess
        self.n_views = n_views-1
        self.aug_list = augmentations
        self.post_auglist = post_augmentations
        self.severity = severity
        self.augmix = augmix
        
    def __call__(self, x):
        if self.augmix:
            image = self.preprocess(x)
            views = [self.augment(x) for _ in range(self.n_views)]

        return torch.stack([image] + views, 0)
    
    def augment(self, x):
        if self.augmix:
            img = _augmix(x, self.preprocess, self.aug_list, self.severity)
        else:
            img = self.preprocess(x)
        for _ in range(random.randint(1,2)):
            img = random.choice(self.post_auglist)(img)

        return img
    

def load_pretrained_coop(backbone, _model, device="cuda"):
    """
    Loads coop pretrained context
    """
    # TODO Makes this function cleaner and more robust
    if backbone.lower() == "rn50":
        _backbone = "rn50"
    elif backbone.lower() == "rn101":
        _backbone = "rn101"
    elif backbone.lower() == "vit_b16" or backbone.lower() == "vit-b/16":
        _backbone = "vit_b16"
    elif backbone.lower() == "vit_b32" or backbone.lower() == "vit-b/32":
        _backbone = "vit_b32"
    else:
        raise ValueError(f"Unknown backbone {backbone}")

    path = f"bin/coop/{_backbone}_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"
    assert os.path.exists(path), f"Path {path} does not exist"

    pretrained_ctx = torch.load(path, device)['state_dict']['ctx']
    assert pretrained_ctx.size()[0] == _model.prompt_learner.n_ctx, f"Number of context tokens mismatch: {_model.prompt_learner.n_ctx} vs {pretrained_ctx.size()[0]}"
    with torch.no_grad():
        _model.prompt_learner.ctx.copy_(pretrained_ctx)
        _model.prompt_learner.ctx_init_state = pretrained_ctx
