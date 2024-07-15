import torch
import os
import io
import random
import numpy as np
import regex as re
from datetime import datetime
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image

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

def batch_report(inputs, outputs, final_prediction, targets, id2classes, batch_n):
    probabilities, predictions = outputs.cpu().topk(5)
    probabilities = probabilities.detach().numpy()
    predictions = predictions.detach()

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean = torch.tensor(clip_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(clip_std).reshape(1, 3, 1, 1)

    # Denormalize the batch of images
    # denormalized_images = inputs.cpu() * std + mean
    # denormalized_images = denormalized_images.numpy().astype('uint8')
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    denormalized_images = unnormalize(inputs)

    # Visualise the input using matplotlib
    images = [image.numpy().transpose(1, 2, 0) for image in denormalized_images.cpu()] # Convert to numpy and transpose to (H, W, C)

    # Visualise the input using matplotlib
    label = id2classes[targets[0].item()]
    plt.figure(figsize=(16,16))
    plt.title(f"Image batch of {label} - min entropy 10 samples selected")
    plt.axis('off')

    for i, image in enumerate(images[:10]):
        plt.subplot(6,4, 2*i+1)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(6,4, 2*i+2)
        y = np.arange(probabilities.shape[-1])
        plt.grid()
        plt.barh(y, probabilities[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [id2classes[pred] for pred in predictions[i].numpy()])
        plt.xlabel("probability")
    
    avg_prob, avg_pred = final_prediction.cpu().topk(5)
    avg_prob = avg_prob.detach().numpy()
    avg_pred = avg_pred.detach()
    plt.subplot(6,4,22)
    y = np.arange(avg_prob.shape[-1])
    plt.grid()
    plt.barh(y, avg_prob[0])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [id2classes[index] for index in avg_pred[0].numpy()])
    plt.xlabel("Final prediction (avg entropy)")    

    plt.savefig(f"batch_reports/Batch{batch_n}.png")
    plt.close()

def make_histogram(no_tpt_acc: dict, tpt_acc: dict, no_tpt_label: str, tpt_label: str, save_path:str=None)-> Image:
    """
    Creates histogram for class accuracies and log it with tensorboard and saves the plot
    """
    classes = list(no_tpt_acc.keys())
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(dpi=500)
    ax.bar(x - width/2, no_tpt_acc.values(), width, color='b', label=no_tpt_label)
    ax.bar(x + width/2, tpt_acc.values(), width, color='r', label=tpt_label)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Class accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=-90, fontsize=2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image = np.array(image)

    if save_path:
        plt.savefig(save_path)

    return image


def batch_report(inputs:torch.Tensor, outputs: torch.Tensor, final_prediction:torch.Tensor,
                 target:torch.Tensor, id2classes: dict, batch_n:int):
    """
    Creates a report in the batch_report/ dir showing augmentation images and their confidence
    Then shows the average confidence prediction
    :param: inputs: torch.Tensor: batch of images
    :param: outputs: torch.Tensor: batch of outputs
    :param: final_prediction: torch.Tensor: average prediction
    :param: target: torch.Tensor: batch with target label
    :param: id2classes: dict: mapping from class index to class name
    :param: batch_n: int: batch number

    TODO: Fix spacing in between images
    """
    from matplotlib import pyplot as plt
    max_plots = 10

    probabilities, predictions = outputs.cpu().topk(5)
    probabilities = probabilities.detach().numpy()
    predictions = predictions.detach()

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean = torch.tensor(clip_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(clip_std).reshape(1, 3, 1, 1)

    # Denormalize the batch of images
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    denormalized_images = unnormalize(inputs)

    # Visualise the input using matplotlib
    images = [image.numpy().transpose(1, 2, 0) for image in denormalized_images.cpu()] # Convert to numpy and transpose to (H, W, C)

    # Visualise the input using matplotlib
    label = id2classes[target[0].item()]

    plt.figure(figsize=(16,16))
    plt.title(f"Image batch of {label} - min entropy {max_plots} percentile selected\n{datetime.now()}")
    plt.axis('off')

    for i, image in enumerate(images[:max_plots]):
        plt.subplot(6,4, 2*i+1)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(6,4, 2*i+2)
        y = np.arange(probabilities.shape[-1])
        plt.grid()
        plt.barh(y, probabilities[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [id2classes[pred] for pred in predictions[i].numpy()])
        plt.xlabel("probability")
    # Original image
    plt.subplot(6,4, 22)
    plt.imshow(images[0])
    plt.axis('off')
    plt.xlabel("Original image")

    # Final prediction
    plt.subplot(6,4, 2*i+1)
    plt.imshow(image)
    plt.axis('off')
    avg_prob, avg_pred = final_prediction.cpu().topk(5)
    avg_prob = avg_prob.detach().numpy()
    avg_pred = avg_pred.detach()
    plt.subplot(6,4,23)
    y = np.arange(avg_prob.shape[-1])
    plt.grid()
    plt.barh(y, avg_prob[0])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [id2classes[index] for index in avg_pred[0].numpy()])
    plt.xlabel("Final prediction (avg entropy)")    

    plt.savefig(f"batch_reports/Batch{batch_n}.png")
    plt.close()

def make_histogram(no_tpt_acc: dict, tpt_acc: dict, no_tpt_label: str, tpt_label: str, save_path:str=None)-> Image:
    """
    Creates histogram for class accuracies and log it with tensorboard to save the plot
    :param: no_tpt_acc: dict: class accuracies before TPT
    :param: tpt_acc: dict: class accuracies after TPT
    :param: no_tpt_label: str: label for the no_tpt_acc
    :param: tpt_label: str: label for the tpt_acc
    :param: save_path: str: path to save the plot. If None, the plot is not saved

    :return: PIL.Image: image of the plot
    """
    classes = list(no_tpt_acc.keys())
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(dpi=500)
    ax.bar(x - width/2, no_tpt_acc.values(), width, color='b', label=no_tpt_label)
    ax.bar(x + width/2, tpt_acc.values(), width, color='r', label=tpt_label)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Class accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=-90, fontsize=2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image = np.array(image)

    if save_path:
        plt.savefig(save_path)

    return image

def report_predictions(idx:int, predictions:str, values:float, target:str):
    """
    Saves the predictions to a file in the batch_predictions/ directory
    :param: idx: int: index of the batch
    :param: predictions: str: list of predictions
    :param: values: float: list of probabilities for each prediction
    :param: target: str: target class
    """
    dir = 'batch_predictions/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}batch_{idx}.txt", 'w') as f:
        f.write(f"Target: {target}\n")
        for pred, value in zip(predictions, values[0]):
            f.write(f"\t{pred}: {value:.2f}\n")
        f.write(f"{datetime.now()}")

def compute_accuracies(id2classes:dict, no_tpt_class_acc:dict, tpt_class_acc:dict):
    """
    Computes the average accuracy for each class before and after TPT
    :param: id2classes: dict: mapping from class index to class name
    :param: no_tpt_class_acc: dict: class accuracies before TPT
    :param: tpt_class_acc: dict: class accuracies after TPT

    :return: dict, dict: no_tpt_accuracies, accuracies
    """
    no_tpt_accuracies = {}
    accuracies = {}

    for c in id2classes.values():
        if len(no_tpt_class_acc[c]) == 0 or len(tpt_class_acc[c]) == 0:
            continue
        no_tpt_accuracies[c] = sum(no_tpt_class_acc[c]) / len(no_tpt_class_acc[c])
        accuracies[c] = sum(tpt_class_acc[c]) / len(tpt_class_acc[c])

    return no_tpt_accuracies, accuracies

def filter_on_entropy(inputs:torch.Tensor, outputs:torch.Tensor, p_threshold:int=10, return_original:bool=False):
    """
    Return all inputs and outputs where prediction entropy is in the 'p' percentile
    :param: inputs: torch.Tensor: batch of inputs
    :param: outputs: torch.Tensor: batch of outputs
    :param: p_threshold: int: percentile threshold
    :param: return_original: bool: return the original image of the batch
    """
    entropies = [entropy(t).item() for t in outputs]
    entropies = [0 if val > p_threshold else 1 for val in entropies]
    indices = torch.nonzero(torch.tensor(entropies)).squeeze(1)

    if return_original and 0 not in indices:
        indices.insert(0,0)

    return inputs[indices], outputs[indices]

def caption_report(images, label, outputs, id2class, idx):
    import matplotlib.pyplot as plt

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean = torch.tensor(clip_mean).reshape(1, 3, 1, 1)
    std = torch.tensor(clip_std).reshape(1, 3, 1, 1)

    # Denormalize the batch of images
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    denormalized_images = unnormalize(images)

    # Visualise the input using matplotlib
    images = [image.numpy().transpose(1, 2, 0) for image in denormalized_images.cpu()] # Convert to numpy and transpose to (H, W, C)
    label = [lab.item() for lab in label.cpu()] if label.shape[0] > 1 else label.item()

    plt.figure(figsize=(16, 16), dpi=300)
    plt.title(f"Captions generated from the {idx}th batch") if isinstance(label, list) else plt.title(f"Caption for {id2class[label]} class")
    plt.axis('off')

    for i, image in enumerate(images[:9]):
        plt.subplot(3,3, i+1)
        plt.title(id2class[label[i]]) if isinstance(label, list) else id2class[label]
        plt.xlabel(outputs[i])
        plt.imshow(image)

    plt.savefig(f"caption_reports/batch_{idx}.png")
    plt.close()
