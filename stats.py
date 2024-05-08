###################################################
# Insert functions to compute model statistics here
#   - Confusion matrices
#   - Average class error
###################################################

# !TODO: Implement the following functions
# !TODO: Find other ways to neatly represent results
# !TODO: Find graph to describe/measure the confidence of the predictions/classes
# Common statistics used to evaluate models with classification?

from typing import Optional
from sklearn.metrics import confusion_matrix as confusion_matrix_comp
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_index
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch.utils.tensorboard import SummaryWriter
LOG_DIR = "tmp"

def confusion_matrix(true_labels: list[int], predicted_labels: list[int], class_names: list[str], save_path:str=Optional[str])->tuple:
    """
    Computes and returns the confusion matrix given a batch of images and labels
    Then plots the results
    :param true_labels: list: ground truth labels
    :param predicted_labels: list: predicted labels
    :param class_names: list: class names
    :return: tuple: confusion matrix and figure
    """
    assert len(true_labels) == len(predicted_labels), "Differing prediction and ground truth size" 

    cm = confusion_matrix_comp(true_labels, predicted_labels)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), #d is decimal format: the values will be formatted as integers
                    ha="center", va="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig, cm


def average_class_error(cm, class_names: list[str], k:int = 1, save_path: Optional[str]=None):
    """
    Computes and returns the average class error given a batch of images and labels
    Then plots the results
    :param cm: np.array: confusion matrix
    :param class_names: list: class names
    :param k: int: top k classes to consider
    :param save_path: str: path to save the results
    """
    # Compute class-wise error
    class_wise_error = 1 - np.diag(cm) / np.sum(cm, axis=1)
    
    # Sort class error and class names based on error rates
    sorted_indices = np.argsort(class_wise_error)  # Sort in descending order
    sorted_class_error = class_wise_error[sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_class_names)), sorted_class_error, color='blue')
    plt.ylabel('Classes')
    plt.xlabel('Error Rate')
    plt.title('Error Rate per Class')
    plt.yticks(range(len(sorted_class_names)), sorted_class_names)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return class_wise_error

if __name__ == "__main__":
    writer = SummaryWriter(LOG_DIR)
    true_labels = [0, 1, 2, 0, 1, 2, 0, 2]
    predicted_labels = [0, 1, 1, 0, 1, 2, 0, 2]
    classes = ['Class 0', 'Class 1', 'Class 2']
    fig, cm = confusion_matrix(true_labels, predicted_labels, classes)
    writer.add_figure("Confusion Matrix", fig)
    idx = get_index(f"{LOG_DIR}/conf_mat")
    fig.savefig(f"{LOG_DIR}/conf_mat/confusion_matrix_{idx}.png")
    class_wise_error = average_class_error(cm, classes)
    print("Average Class Error:", class_wise_error)