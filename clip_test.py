import os

import tqdm
from CLIP.clip import clip
import torch
import torchvision
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
         k: int = 5):
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

    for images, labels in loop:
        images.to(device)
        labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        for i in range(similarity.shape[0]):
            values, indices = similarity[i].topk(k)

            if labels[i] == indices[0]:
                correct += 1

            if labels[i] in indices:
                topk_correct += 1

            total += 1
            topk_total += 1
        
        loop.set_postfix_str(f"@1={correct / total}, @5={topk_correct / topk_total}")
        
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


model, preprocess = clip.load('ViT-B/32', device)

imagenet_a_wnids = os.listdir('./data/imagenet-a')
imagenet_a_wnids.remove('README.txt')
assert len(imagenet_a_wnids) == 200

imagenet_A = datasets.ImageFolder(root='./data/imagenet-a', transform=no_transform)
imagenet_A_loader = torch.utils.data.DataLoader(imagenet_A, batch_size=1024, shuffle=True)

id2class = {imagenet_A.class_to_idx[c] : py_vars.num2class[c] for c in imagenet_A.classes}
top_1, top_5 = eval(imagenet_A_loader, py_vars.num2class.items(), id2class, device)

imagenet_v2 = datasets.ImageFolder(root='./data/imagenetv2-matched-frequency-format-val', transform=no_transform)
imagenet_v2_loader = torch.utils.data.DataLoader(imagenet_v2, batch_size=1024, shuffle=True)

id2class = {imagenet_v2.class_to_idx[c] : py_vars.num2class_v2[int(c)] for c in imagenet_v2.classes}

top_1, top_5 = eval(imagenet_v2_loader, py_vars.num2class_v2.items(), id2class, device)