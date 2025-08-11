"""
    Filename: cluster.py
    Title: Perform semi-supervised KNN algorithm to classify the embedded words (which are now vectors)
    Note***: May need the PCA model in the pipeline of the model (to improve efficiency)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import torch
import matplotlib.pyplot as plt
import joblib

def cluster_embeddings(embeddings: np.ndarray, labeled_indices = None, labels_for_labeled = None,  
                       n_components = 50, random_state = 42, k_max = 20, k_min = 2,
                       save_model_path = None, plot_scores = True, num_folds = 5):
    '''
        Semi-supervision using KNN (base estimator) with hyperparameter tuning \\
        
        Parameters: \\
        labeled_indices: list or np.array
            Indices of labeled samples\\
        labels_for_labeled: list or np.array
            Corresponding labels for labeled samples \\
        save_model_path: str or None
            Path to save best model using joblib (optional)\\
        plot_scores: bool
            Whether to plot score vs k

        Returns:\\
        best_labels: np.array
            Final predicted labels for all samples from best model\\
        best_model: sklearn Pipeline
            Trained model pipeline with best k\\
        best_k: int
            Best k value
    '''
    ### TODO: Might think of dimensionality reduction to increase speed and efficiency
    ### Can try with other algorithms (Gaussian Mixture Model is a great example)
    X = embeddings
    y = -1 * np.ones(X.shape[0], dtype=int) ### the labels initialised
    if labeled_indices is not None and labels_for_labeled is not None:
        y[labeled_indices] = labels_for_labeled

    X_labeled = X[labeled_indices]
    y_labeled = np.array(labels_for_labeled)

    scores = []
    best_score = -np.inf
    best_model = None
    best_k = None
    best_labels = None

    skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = random_state)

    for k in range(k_min, k_max + 1):
        fold_scores = []
        for train_idx, val_idx in skf.split(X_labeled, y_labeled):
            X_train_fold = X_labeled[train_idx]
            y_train_fold = y_labeled[train_idx]
            X_val_fold = X_labeled[val_idx]
            y_val_fold = y_labeled[val_idx]

            base_model = make_pipeline(
                PCA(n_components = n_components, random_state = random_state),
                KNeighborsClassifier(n_neighbors = k)
            )
            selftrainingknn = SelfTrainingClassifier(base_model, threshold = 0.95)
            
            y_temp = -1 * np.ones(X.shape[0], dtype=int)
            labeled_indices = np.array(labeled_indices)
            temp_indices = labeled_indices[train_idx]
            y_temp[temp_indices] = y_train_fold

            selftrainingknn.fit(X, y_temp)

            fold_score = selftrainingknn.score(X_val_fold, y_val_fold)
            fold_scores.append(fold_score)

        avg_score = np.mean(fold_scores)
        scores.append((k, avg_score))
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = selftrainingknn
            best_labels = selftrainingknn.predict(X)
            best_k = k

    if plot_scores and scores:
        ks, s = zip(*scores)
        plt.figure(figsize=(10, 8))
        plt.plot(ks, s, marker='o')
        plt.title("KNN Self-Training: Cross-Validation Accuracy Score vs k")
        plt.xlabel("k (neighbors)")
        plt.ylabel("Avg Accuracy on Labeled Fold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tuning.svg')

    if best_model is None:
        print("[ERROR] No valid model found. Please check labeled data or k-range.")
        return None, None, None

    if save_model_path and best_model is not None:
        joblib.dump(best_model, save_model_path)
        print(f"[INFO] Best model saved to {save_model_path}")

    return best_labels, best_model, best_k