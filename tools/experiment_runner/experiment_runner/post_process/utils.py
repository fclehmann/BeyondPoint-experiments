from pathlib import Path
import re
import numpy as np
import tomlkit


def macro_f1_score(confusion_matrix: np.ndarray):
    # Number of classes
    num_classes = confusion_matrix.shape[0]

    # Initialize F1 scores for each class
    f1_scores = []

    for i in range(num_classes):
        # True positives (TP)
        tp = confusion_matrix[i, i]

        # False positives (FP) - Sum of column i, excluding the diagonal element
        fp = np.sum(confusion_matrix[:, i]) - tp

        # False negatives (FN) - Sum of row i, excluding the diagonal element
        fn = np.sum(confusion_matrix[i, :]) - tp

        # Precision and Recall for the class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 score for the class
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1)

    # Macro F1 score
    macro_f1 = np.mean(f1_scores)
    return macro_f1

def macro_precision(confusion_matrix):
    # Number of classes
    num_classes = confusion_matrix.shape[0]

    xs = []
    for i in range(num_classes):
        # True positives (TP)
        tp = confusion_matrix[i, i]

        # False positives (FP) - Sum of column i, excluding the diagonal element
        fp = np.sum(confusion_matrix[:, i]) - tp

        # Precision and Recall for the class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        xs.append(precision)
    
    return np.mean(xs)

def macro_recall(confusion_matrix):
    # Number of classes
    num_classes = confusion_matrix.shape[0]

    xs = []
    for i in range(num_classes):
        # True positives (TP)
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        # Precision and Recall for the class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        xs.append(recall)
    
    return np.mean(xs)

def string_to_numpy_array(s):
    # Remove any extra characters, including line breaks and brackets
    cleaned_str = re.sub(r'[\[\]\n]', '', s)

    # Split the cleaned string and convert it to a NumPy array of integers
    data_list = [int(x.strip()) for x in cleaned_str.split()]
    rows = len(s.split('\n'))
    cols = len(data_list) // rows
    np_array = np.array(data_list).reshape(rows, cols)

    return np_array

def read_ec(exp_path: Path) -> tomlkit.TOMLDocument:
    return tomlkit.parse(next(exp_path.glob("*.toml")).read_text())

