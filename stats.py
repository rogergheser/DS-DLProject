###################################################
# Insert functions to compute model statistics here
#   - Confusion matrices
#   - Average class error
###################################################

# !TODO: Implement the following functions
# !TODO: Find other ways to neatly represent results
# Common statistics used to evaluate models with classification?

from typing import Optional
from sklearn.metrics import confusion_matrix as confusion_matrix_comp
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

LOG_DIR = "tmp/conf_mat"

def confusion_matrix(true_labels: list[int], predicted_labels: list[int], class_names: list[str], save_path:str=Optional[str]):
    assert len(true_labels) == len(predicted_labels), "Differing prediction and ground truth size" 

    cm = confusion_matrix_comp(true_labels, predicted_labels)
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
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

    plt.show()
    return cm


# def confusion_matrix(cm: np.ndarray, class_names: str=[], batch_size:int = 10, save_path:str=Optional[str]):
#     """
#     Computes and returns/plots the confusion matrix given a batch of images and labels
#     """
    
#     figure = plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)
    
#     # Normalize the confusion matrix.
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
#     # Use white text if squares are dark; otherwise black.
#     threshold = cm.max() / 2.
    
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         color = "white" if cm[i, j] > threshold else "black"
#         plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
#     buf = io.BytesIO()
    
#     # Use plt.savefig to save the plot to a PNG in memory.
#     plt.savefig(buf, format='png')
    
#     # Closing the figure prevents it from being displayed directly inside
#     # the notebook.
#     plt.close(figure)
#     buf.seek(0)
    
#     # Use tf.image.decode_png to convert the PNG buffer
#     # to a TF image. Make sure you use 4 channels.
#     figure = tf.image.decode_png(buf.getvalue(), channels=4)
    
#     # Use tf.expand_dims to add the batch dimension
#     figure = tf.expand_dims(figure, 0)
    
#     return figure


def average_class_error(k:int = 1, save_path=Optional[str]):
    """
    Computes and returns the average class error given a batch of images and labels
    Then plots the results
    :param k: int: top k classes to consider
    :param save_path: str: path to save the results
    """
    pass

if __name__ == "__main__":
    true_labels = [0, 1, 2, 0, 1, 2, 0, 2]
    predicted_labels = [0, 1, 1, 0, 1, 2, 0, 2]
    classes = ['Class 0', 'Class 1', 'Class 2']
    cm=confusion_matrix(true_labels, predicted_labels, classes)
    plt.imsave(LOG_DIR, cm)
    
    
