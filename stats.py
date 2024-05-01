###################################################
# Insert functions to compute model statistics here
#   - Confusion matrices
#   - Average class error
###################################################

# !TODO: Implement the following functions
# !TODO: Find other ways to neatly represent results
# Common statistics used to evaluate models with classification?

from typing import Optional

def confusion_matrix(batch_size:int = 10, save_path:str=Optional[str]):
    """
    Computes and returns/plots the confusion matrix given a batch of images and labels
    """
    pass

def average_class_error(k:int = 1, save_path=Optional[str]):
    """
    Computes and returns the average class error given a batch of images and labels
    Then plots the results
    :param k: int: top k classes to consider
    :param save_path: str: path to save the results
    """
    pass

