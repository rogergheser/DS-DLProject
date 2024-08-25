import random
import torch
import os
import py_vars
import shutil

from torch.utils import data
from torch.utils.data import random_split
from torchvision import datasets
from loaders import Augmixer
from typing import Optional

def my_collate(batch):
        # Unpack the batch
        images, label, path = zip(*batch)

        # Remove the extra dimension and stack the images and labels
        images = torch.stack([img.squeeze(0) for img in images]).squeeze(0)
        labels = torch.tensor(label[0]).unsqueeze(0)

        return images, labels, path

class AugmixFolder(datasets.ImageFolder):
    def __init__(self, root,transform):
        super(AugmixFolder, self).__init__(root, transform=transform)
        self.transform = transform
           
    def __getitem__(self, index):
        img, label = super(AugmixFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        if isinstance(self.transform, Augmixer):
            return img.squeeze(0), label, path
        return img, label, path

class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, indices, from_idx: int = 0):
        self.indices = indices[from_idx:]

    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
    

def get_data(dataset_name, batch_size, transform, shuffle=True, train_size=0.8, val_size=0.1, from_idx=0):
    """
    Loads the dataset and splits it into training, validation and test sets. Available datsets:
    ["cifar10", "cifar100", "imagenet_v2", "imagenet_a"]
    :param dataset_name: str: name of the dataset
    :param batch_size: int: batch size
    :param transform: function: preprocessing function
    :param shuffle: bool: shuffle the dataset
    :param train_size: float: proportion of the dataset to include in the training set
    :param val_size: float: proportion of the dataset to include in the validation set
    :return: tuple: training, validation and test dataloaders
    """
    if dataset_name == "cifar10":
        download = not (os.path.exists(os.path.join("data/cifar-10-python")))
        dataset = AugmixFolder(root="./data", download=download, transform=transform)
        id2class = {dataset.class_to_idx[c] : c for c in dataset.classes}
    elif dataset_name == "cifar100":
        download = not (os.path.exists(os.path.join("data/cifar-100-python")))
        dataset = AugmixFolder(root="./data", download=download, transform=transform)
        id2class = {dataset.class_to_idx[c] : c for c in dataset.classes}
    elif dataset_name == "imagenet_v2":
        root = "./data/imagenetv2-matched-frequency-format-val"
        filtered_root = "./data/imagenetv2-random-partition"
        if not os.path.exists(filtered_root):
            os.makedirs(filtered_root, exist_ok=True)
            all_subfolders = [d.name for d in os.scandir(root) if d.is_dir()]
            selected_subfolders = random.sample(all_subfolders, 500)

            for folder in selected_subfolders:
                src_folder = os.path.join(root, folder)
                dest_folder = os.path.join(filtered_root, folder)
                if not os.path.exists(dest_folder):
                    shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True) 
        else:
            print("Using existing partition of ImagenetV2, ensure you're using a checkpoint")

        dataset = AugmixFolder(root=filtered_root, transform=transform)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
        id2class = {dataset.class_to_idx[c] : py_vars.num2class_v2[int(c)] for c in dataset.classes}
    elif dataset_name == "imagenet_a":
        dataset = AugmixFolder(root="./data/imagenet-a", transform=transform)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
        id2class = {dataset.class_to_idx[c] : py_vars.num2class[c] for c in dataset.classes}
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    n = len(dataset)
    n_train = int(train_size * n)
    n_val = int(val_size * n)
    n_test = n - n_train - n_val
    
    if(n_train + n_val == 0):
        train_loader, val_loader = None, None
        if batch_size == 1:
            test_loader = data.DataLoader(dataset, batch_size=batch_size, 
                                          sampler=CustomSampler(range(n), from_idx=from_idx), collate_fn=my_collate)
        else:
            test_loader = data.DataLoader(dataset, batch_size=batch_size,
                                          sampler=CustomSampler(range(n), from_idx=from_idx))
    else:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate)

    return train_loader, val_loader, test_loader, list(id2class.values()), id2class
