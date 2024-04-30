import math
import os

import tqdm
from CLIP.clip import clip
import torch
import torchvision
import loaders
from torchvision.datasets import CIFAR100
from torchvision.transforms import AugMix
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import py_vars
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def process_batch(loader: torch.utils.data.DataLoader,
                    classes: list,
                    id2class: dict,
                    device: str = "cpu"):
    """
    Processes a batch of images and labels and then displays the results
    Mostly for testing purposes
    """
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images.to(device)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images.to(device))
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    for i in range(similarity.shape[0]):
        values, indices = similarity[i].topk(5)

        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{id2class[index.item()]:>16s}: {100 * value.item():.2f}%")
        print("\nTrue label:{}".format(id2class[labels[i].item()]))

        show_image(images[i], f"{id2class[labels[i].item()]} - {labels[i]}")

def eval(loader: torch.utils.data.DataLoader,
         classes: list,
         id2class: dict,
         device: str = "cpu",
         k: int = 5,
         augmix: int = -1):
    """
    Process dataset and compute top1 and topk accuracy
    """
    correct = 0
    total = 0
    topk_correct = 0
    topk_total = 0
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    dataset_name = loader.dataset.root.split("/")[-1]
    loop = tqdm.tqdm(loader, desc="Processing {}".format(dataset_name))
    text_features = None
    batch_size = loader.batch_size

    if augmix > 0:
        assert loader.batch_size == 1, "Augmix only works with batch size equal to 1"

    for images, labels in loop:
        labels.to(device)

        if augmix > 0:
            labels = labels.repeat(augmix + 1)
            augmix_transform = AugMix(severity=1, width=3, depth=-1, alpha=1.0)
            images = generate_augmented_batch(images[0], augmix, augmix_transform)

            # for _ in range(augmix):
            #     severity = 2
            #     mixture_width = np.random.randint(1, 4)
            #     augmix_transform = torchvision.transforms.AugMix(severity=severity, mixture_width=mixture_width, chain_depth=-1, alpha=1.0)
            #     augmented_image = augmix_transform(images[-1].to(torch.uint8))
            #     images = torch.cat([images, augmented_image.unsqueeze(0)])
        
        images.to(device)
        # for image in images:
        #     show_image(image, id2class[labels[0].item()])


        with torch.no_grad():
            image_features = model.encode_image(images.to(device))
            if text_features is None: # avoid unnecessary computation of text features
                text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        epsilon = 1e-8 # This value is added to log to avoid NaNs

        if augmix > 0:
            entropies = -torch.sum(similarity * torch.where(similarity>0, torch.log(similarity), similarity.new([0])), dim=1)
            entropies, indices = entropies.topk(1)
            final_similarity = sum([similarity[i] for i in indices])/5
            
            values, indices = final_similarity.topk(k)
            if labels[0] == indices[0]:
                correct += 1
            if labels[0] in indices:
                topk_correct += 1

        else:
            for i in range(similarity.shape[0]):
                values, indices = similarity[i].topk(k)

                if labels[i] == indices[0]:
                    correct += 1

                if labels[i] in indices:
                    topk_correct += 1
                print("\nTop predictions:\n")
                for value, index in zip(values, indices):
                    print(f"{id2class[index.item()]:>16s}: {100 * value.item():.2f}%")
                print("\nTrue label:{}".format(id2class[labels[i].item()]))
                show_image(images[i], f"{id2class[labels[i].item()]} - {labels[i]}")
                print("\n\n")
        total += batch_size
        topk_total += batch_size
        
        loop.set_postfix_str(f"@1={correct / total}, @{k}={topk_correct / topk_total}")
        
    return correct / total, topk_correct / topk_total

def show_image(image, label):
    image = image.numpy()
    plt.title(f"Image of {label}")
    img = np.transpose((image * 255).astype(np.uint8), (1, 2, 0))
    plt.imshow(img)
    plt.show()

all_wnids = py_vars.all_wnids

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
    # device = "cpu"
else:
    device = "cpu"


# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
no_transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

augmix_transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                ToUint8Transform(),
                                # AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)
        ])


model, preprocess = clip.load('ViT-B/16', device)

print("="*90)
imagenet_A_loader, id2class = loaders.load_imagenet_A('./data/imagenet-a', 256, preprocess)
try:
    top_1, top_5 = eval(imagenet_A_loader, py_vars.num2class.items(), id2class, device, augmix=0)
except KeyboardInterrupt:
    "Move on with next dataset"

print("="*90)
imagenet_v2_loader, id2class = loaders.load_imagenet_v2('./data/imagenetv2-matched-frequency-format-val', 1, preprocess, False)
try:
    top_1, top_5 = eval(imagenet_v2_loader, py_vars.num2class_v2.items(), id2class, device, augmix=0)
except KeyboardInterrupt:
    "Move on with next dataset"

print("="*90)
cifar100_loader, id2class = loaders.load_cifar100('./data/cifar100', 1, preprocess)
try:
    top_1, top_5 = eval(cifar100_loader, id2class.items(), id2class, device, augmix=63)
except KeyboardInterrupt:
    "Move on with next dataset"

# plot results