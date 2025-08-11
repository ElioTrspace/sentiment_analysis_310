# Sexual Harassment Content Classification Pipeline

This project implements a machine learning pipeline for detecting potential sexual harassment content from Reddit comments and posts. It supports semi-supervised learning to leverage both labeled and unlabeled data, and includes clustering, vectorization, and evaluation modules.

---

## Features

- **Text Cleaning** (`cleaning.py`)  
  Removes unwanted characters, punctuation, stopwords, and normalizes text for processing. Also adds 
  a layer to filter out all bot comments.

- **Text Vectorization** (`text2Vec.py`)  
  Converts cleaned text into numerical embeddings using pre-trained models BERT (`embeddings.pt`).
  We used bert-base-uncase model, i.e., texts must be in lowercase first before the embeddings happen.

- **Clustering** (`cluster.py`)  
  In this file, the base classifier model is KNN, (the built-in KNeighborsClassifier class in scikit-learn
  is used). The embeddings dimensionality is reduced to 50 for computational efficiency. Then, the base
  classifier is used for semi-supervision, we used the SelfTrainingClassifier in scikit-learn in this project.
  We also implemented cross-validation with 5 folds for the KNN to tune the hyperparameter k (considering
  k from 2 to 20).

- **Semi-labeling** (`labeled_vs_unlabeled.py`)  
  Automate the labeling process. We only labeled some data in this project for semi-supervision. 

- **Visualization** (`visualize_data.py`)  
  Creates visual summaries of the dataset, clusters, and model performance.

- **Evaluation** (`evaluation.py`)  
  Evaluates model performance using precision, recall, F1-score, and accuracy.

- **Main Pipeline** (`main.py`)  
  Orchestrates all components into a cohesive end-to-end workflow.

- **Interactive Demo** (`tryout.py`)
  A demo run on GUI for users to interact with the model, but mostly for us to test some edge cases.
  Please create the file `embeddings.pt` first using `main.py` before running the try-out.
---

## Project Structure

```plaintext
.
├── cleaning.py               # Text preprocessing
├── cluster.py                # Clustering for unsupervised grouping
├── evaluation.py              # Model evaluation metrics
├── labeled_vs_unlabeled.py    # Semi-supervised pseudo-labeling
├── text2Vec.py                # Vectorization of text data
├── tryout.py                  # Experimental scripts
├── visualize_data.py          # Data visualization tools
├── main.py                    # Main execution script
├── embeddings.pt              # Pre-trained text embeddings
└── README.md                  # Project documentation
```

---

## Pipeline Flow

```       ┌───────────────────────────┐
          │ raw texts data from Reddit|
          |API (joined_data.json.gz)  │
          └───────────────┬───────────┘
                          │
                          ▼
                    ┌────────────┐
                    │  cleaning  │
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │  text2Vec  │
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐
                    │  cluster   │
                    └─────┬──────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │ labeled_vs_unlabeled.py   │
            └───────────────┬───────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │  evaluation    │
                    └────────────────┘
                            │
                            ▼
                ┌──────────────────┐
                │ visualize_data   │
                └──────────────────┘
```

---

## Getting Started
### 0. Prerequisite
#### For Linux
Python 3.9-3.12 is generally installed by default on any of PyTorch supported Linux distributions. 
#### For Windows
Currently, PyTorch on Windows only supports Python 3.9-3.12; Python 2.x is not supported.

#### For Transformers
Transformers works with Python 3.9+ PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

### 1. Install Dependencies (supported for Windows)
```bash
pip3 install numpy
pip3 install pandas
pip3 install torch torchvision
pip3 install scikit-learn
pip3 install seaborn
pip3 install transformers
pip3 install matplotlib
```

### 2. Prepare the Data
Place your dataset (Reddit comments/posts) in the same directory as the code.

### 3. Run the Pipeline
```bash
python3 main.py joined_data.json.gz
```
### 4. Run the Interactive Try-out (make sure you have the embeddings.pt file first)
```bash
python3 tryout.py
```
---

## Built With

- Python 3.11.9
- NumPy, Pandas - Data manipulation
- scikit-learn - Clustering & evaluation
- PyTorch - Embedding generation
- Matplotlib / Seaborn - Visualization
- Tkinter - For the Demo
---

