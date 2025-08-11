"""
    Filename: text2Vec.py
    Title: Embed words (inputs) into numerical vectors that the machine can understand
           Using BERT
"""
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from tqdm import tqdm

def embed_texts(texts: List[str], model_name = "bert-base-uncased", batch_size = 128) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            batch = [" ".join(x) if isinstance(x, list) else str(x) for x in batch] 
            inputs = tokenizer(batch, padding = True, truncation = True, 
                               return_tensors = "pt", max_length = 128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu())

    return torch.cat(embeddings, dim=0)