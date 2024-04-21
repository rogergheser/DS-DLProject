import os
from CLIP.clip import clip
import torch
import torchvision
from torchvision.datasets import CIFAR100
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import py_vars
import matplotlib.pyplot as plt
import numpy as np


def process_batch(loader: torch.utils.data.DataLoader,
                    classes: list,
                    id2class: dict,
                    device: str = "cpu"):
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
no_transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

model, preprocess = clip.load('ViT-B/32', device)

imagenet_a_wnids = os.listdir('./data/imagenet-a')
imagenet_a_wnids.remove('README.txt')
assert len(imagenet_a_wnids) == 200

print("Processing ImageNet-A")
imagenet_A = datasets.ImageFolder(root='./data/imagenet-a', transform=preprocess)
imagenet_A_loader = torch.utils.data.DataLoader(imagenet_A, batch_size=10, shuffle=True)

id2class = {imagenet_A.class_to_idx[c] : py_vars.num2class[c] for c in imagenet_A.classes}

process_batch(imagenet_A_loader, id2class.items(), id2class, device)

print("Processing ImageNet-V2")
imagenet_v2 = datasets.ImageFolder(root='./data/imagenetv2-matched-frequency-format-val', transform=no_transform)
imagenet_v2_loader = torch.utils.data.DataLoader(imagenet_v2, batch_size=10, shuffle=True)

id2class = {imagenet_v2.class_to_idx[c] : py_vars.num2class_v2[int(c)] for c in imagenet_v2.classes}

process_batch(imagenet_v2_loader, py_vars.num2class_v2.items(), id2class, device)