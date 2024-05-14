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
import pickle
from utils import *
import stats

_datasets = {
    # "cifar100" : "./data/cifar100",
    "imagenet_A" : "./data/imagenet-a",
    "imagenet_v2" : "./data/imagenetv2-matched-frequency-format-val" 
}

_loaders = {
    "cifar100" : loaders.load_cifar100,
    "imagenet_A" : loaders.load_imagenet_A,
    "imagenet_v2" : loaders.load_imagenet_v2
}

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

    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in id2class.values()]).to(device)
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

    true_labels = []
    predicted_topk_labels = []
    predicted_topk_confidence = []

    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in id2class.values()]).to(device)

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

                true_labels.append(labels[i].item())
                predicted_topk_labels.append(indices)
                predicted_topk_confidence.append(values)

        total += batch_size
        topk_total += batch_size
        
        loop.set_postfix_str(f"@1={correct / total}, @{k}={topk_correct / topk_total}")

    true_labels = torch.tensor(true_labels).cpu()
    predicted_topk_labels = torch.stack([x.cpu() for x in predicted_topk_labels])
    predicted_topk_confidence = torch.stack([x.cpu() for x in predicted_topk_confidence])  
    
    return correct / total, topk_correct / topk_total, true_labels, predicted_topk_labels, predicted_topk_confidence

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

if __name__ == "__main__":
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

    for _dataset in _datasets:
        print("="*90)
        loader, id2class = _loaders[_dataset](_datasets[_dataset], 256, preprocess)
        
        try:
            top_1, top_5, true_labels, predicted_topk_labels, predicted_topk_confidence = eval(
                loader, list(py_vars.num2class.values()), id2class, device, augmix=0)
            true_labels.cpu()
            predicted_topk_labels.cpu()
            predicted_topk_confidence.cpu()
            print(f"Top 1 accuracy of {_dataset}: {top_1}")
            print(f"Top 5 accuracy of {_dataset}: {top_5}")
            
            idx = get_index(f"results/{_dataset}")
            if not os.path.exists(f"results/{_dataset}"):
                os.makedirs(f"results/{_dataset}")
            pickle.dump((true_labels, predicted_topk_labels, predicted_topk_confidence), open(f"results/{_dataset}/run{idx}.pkl", "wb"))
        except KeyboardInterrupt:
            f"Stopped dataset of {_dataset} evaluation earlier"

        predicted_label = [i[0].item() for i in predicted_topk_labels]
        fig, _ = stats.confusion_matrix(true_labels, predicted_label, list(py_vars.num2class.values()), f"results/{_dataset}/conf_mat")
        if not os.path.exists(f"results/{_dataset}/conf_mat"):
            os.makedirs(f"results/{_dataset}/conf_mat")
        idx = get_index(f"results/{_dataset}/conf_mat")
        fig.savefig(f"results/{_dataset}/conf_mat/confusion_matrix_{idx}.png")
        # class_average_error = stats.average_class_error(cm)


        # writer.add_figure(f"Confusion Matrix {_dataset}", fig)
        # writer.add_scalar(f"Top 1 accuracy {_dataset}", top_1)
        # writer.add_scalar(f"Top 5 accuracy {_dataset}", top_5)
        # writer.flush()