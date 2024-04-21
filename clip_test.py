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


def show_image(image, label):
    image = image.numpy()
    plt.title(f"Image of {label}")
    plt.imshow(np.transpose(image.astype(np.uint8), (1, 2, 0)))
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
# transform = transforms.Compose([transforms.Resize(256),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean, std)])

model, preprocess = clip.load('ViT-B/32', device)

imagenet_a_wnids = os.listdir('./data/imagenet-a')
imagenet_a_wnids.remove('README.txt')
assert sorted(imagenet_a_wnids) == sorted(py_vars.imagenet_a_wnids)

imagenet_a_mask = [wnid in set(imagenet_a_wnids) for wnid in all_wnids]

imagenet_A = datasets.ImageFolder(root='./data/imagenet-a', transform=preprocess)
imagenet_A_loader = torch.utils.data.DataLoader(imagenet_A, batch_size=10, shuffle=True)

dataiter = iter(imagenet_A_loader)


image, labels = next(dataiter)

image.to(device)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {py_vars.num2class[c]}") for c in imagenet_A.classes]).to(device)

with torch.no_grad():
    # image_features = model.encode_image(image)
    image_features = model.encode_image(image.to(device))
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

for i in range(similarity.shape[0]):
    values, indices = similarity[i].topk(5)

    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{py_vars.num2class[imagenet_A.classes[index]]:>16s}: {100 * value.item():.2f}%")
    print("\nTrue label:{}".format(py_vars.num2class[imagenet_A.classes[labels[i]]]))

    show_image(image[i], py_vars.num2class[imagenet_A.classes[labels[i]]])