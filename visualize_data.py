import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

def plot_tsne(embeddings, labels, texts=None, sample_size=500):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hls", len(set(labels)))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette=palette, s = 30, legend = "full")

    # annotate some points
    if texts is not None:
        for i in np.random.choice(len(texts), min(sample_size, len(texts)), replace = False):
            plt.text(X_2d[i, 0], X_2d[i, 1], texts[i][:15], fontsize = 6, alpha = 0.7)

    plt.title("t-SNE visualization of clusters")
    plt.tight_layout()
    plt.savefig("tsne_clusters.svg")
    # plt.show()
