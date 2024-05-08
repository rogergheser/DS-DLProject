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
from torch.utils.tensorboard import SummaryWriter
from utils import get_index

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
LOG_DIR = "tmp/"

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
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), #d is decimal format: the values will be formatted as integers
                    ha="center", va="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig, cm


def average_class_error(cm, k:int = 1, save_path=Optional[str]):
    """
    Computes and returns the average class error given a batch of images and labels
    Then plots the results
    :param k: int: top k classes to consider
    :param save_path: str: path to save the results
    """
    # Compute class-wise error
    class_error = 1 - np.diag(cm) / np.sum(cm, axis=1)
    
    class_average_error = np.mean(class_error)
    return class_average_error

if __name__ == "__main__":
    writer = SummaryWriter(LOG_DIR)
    true_labels = [0, 1, 2, 0, 1, 2, 0, 2]
    predicted_labels = [0, 1, 1, 0, 1, 2, 0, 2]
    classes = ['Class 0', 'Class 1', 'Class 2']
    fig, cm = confusion_matrix(true_labels, predicted_labels, classes)
    writer.add_figure("Confusion Matrix", fig)
    idx = get_index(f"{LOG_DIR}/conf_mat")
    fig.savefig(f"{LOG_DIR}/conf_mat/confusion_matrix_{idx}.png")
    class_average_error = average_class_error(cm)
    print("Average Class Error:", class_average_error)