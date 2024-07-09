import random
import torch
from torchvision import transforms
import numpy as np
import regex as re
import os
from matplotlib import pyplot as plt

def show_image(image, label):
    image = image.numpy()
    plt.title(f"Image of {label}")
    img = np.transpose((image * 255).astype(np.uint8), (1, 2, 0))
    plt.imshow(img)
    plt.show()

def entropy(p):
    """
    Given a tensor p representing a probability distribution, returns the entropy of the distribution
    """
    return -torch.sum(p * torch.log(p + 1e-8))

def get_index(path):
    """
    Given a directory path, returns the highest index of the files in the directory or zero
    """
    try:
        files = os.listdir(path)
        indices = [int(re.findall(r'\d+', file)[0]) for file in files]
        return max(indices) + 1
    except:
        return 0

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
        # Define a list of transformations; these can be adjusted as needed
        self.augmentations = [
            transforms.ColorJitter(0.8*self.severity, 0.8*self.severity, 0.8*self.severity, np.clip(0.2*self.severity, -0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224, padding=int(224*0.125), pad_if_needed=True)
        ]
        
    def forward(self, img):
        ws = np.float32(np.random.dirichlet([self.alpha]*self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        
        mix = torch.zeros_like(img)
        for i in range(self.width):
            image_aug = img.clone()
            for _ in range(self.depth):
                op = random.choice(self.augmentations)
                image_aug = op(image_aug)
            mix += ws[i] * image_aug
        
        mixed = (1 - m) * img + m * mix
        return mixed
    
def generate_augmented_batch(original_tensor, num_images, augmix_module):
    from torchvision.transforms import Compose, Resize, ToTensor
    batch = [original_tensor]  # Start with the original image

    # Generate num_images-1 augmented images
    for _ in range(num_images):
        augmented_image = augmix_module(original_tensor.unsqueeze(0)).squeeze(0)
        batch.append(augmented_image)

    # Convert list of tensors to a single tensor
    batch_tensor = torch.stack(batch)
    return batch_tensor

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)