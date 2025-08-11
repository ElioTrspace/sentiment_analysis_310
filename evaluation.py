from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from typing import List, Optional

def evaluate_predictions(y_true: List[int], y_pred: List[int], label_names: Optional[List[str]] = None):
    """
    Evaluate classification performance of the semi-supervised model.

    Parameters:
        y_true (List[int]): True labels for the labeled data.
        y_pred (List[int]): Predicted labels (same length as y_true).
        label_names (List[str], optional): Class names for reporting.
    """
    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names = label_names))

    print("\n[INFO] Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def print_cluster_examples(texts: List[str], labels: List[int], num_examples=3):
    """
    Print a few sample texts from each cluster.

    Parameters:
        texts (List[str]): List of original texts.
        labels (List[int]): Predicted cluster labels.
        num_examples (int): Number of examples to print per cluster.
    """
    from collections import defaultdict
    import random

    cluster_map = defaultdict(list)
    for text, label in zip(texts, labels):
        cluster_map[label].append(text)

    for label, examples in cluster_map.items():
        print(f"\n[Cluster {label}] ({len(examples)} samples)")
        print("-" * 30)
        for ex in random.sample(examples, min(num_examples, len(examples))):
            print(f"- {ex}...")