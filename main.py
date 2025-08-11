import sys
from cleaning import preprocess_dataset
from text2Vec import embed_texts
from cluster import cluster_embeddings
from labeled_vs_unlabeled import get_weak_labels
from evaluation import evaluate_predictions, print_cluster_examples
from visualize_data import plot_tsne
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
from joblib import Parallel, delayed

def main(input_file):
    print("[INFO] Preprocessing dataset...")
    data_frame = preprocess_dataset(input_file)
    data_frame = data_frame.dropna(subset = ['subreddit'])
    texts = data_frame['body'].tolist()
    subs = data_frame['subreddit'].tolist()

    print("[INFO] Creating weak labels...")
    labeled_indices, labels_for_labeled = get_weak_labels(texts, subs)
    print(f"[INFO] Labeled samples: {len(labeled_indices)} / {len(texts)}")
    
    embeddings_file = "embeddings.pt"
    if os.path.exists(embeddings_file):
        print("[INFO] Loading cached embeddings...")
        embeddings = torch.load(embeddings_file)
    else:
        print("[INFO] Converting text to embeddings...")
        embeddings = embed_texts(texts)
        torch.save(embeddings, embeddings_file)

    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings

    print("[INFO] Training semi-supervised model with KNN...")

    X_labeled = embeddings_np[labeled_indices]
    y_labeled = np.array(labels_for_labeled)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X_labeled, y_labeled, 
                                                                          labeled_indices, 
                                                                          test_size = 0.3, 
                                                                          stratify = y_labeled, 
                                                                          random_state = 127)
    
    labels, model, best_k = cluster_embeddings(
        embeddings_np,
        labeled_indices = idx_train,
        labels_for_labeled = y_train,
        n_components = 50,
        plot_scores = True,
        save_model_path = "best_selftraining_knn.pkl"
    )

    y_pred = model.predict(X_val)
    print(f"[INFO] The best k value after tuning the hyperparamter is: {best_k}")

    print("[INFO] Evaluating clusters and showing examples...")
    evaluate_predictions(y_true = y_val, y_pred = y_pred)
    print_cluster_examples(texts, labels)
    
    print("[INFO] Visualizing clusters...")
    plot_tsne(embeddings.cpu().numpy(), labels, texts)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <path_to_data.json.gz>")
        sys.exit(1)

    input_path = sys.argv[1]
    main(input_path)