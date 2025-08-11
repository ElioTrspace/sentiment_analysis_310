import tkinter as tk
from tkinter import messagebox
import joblib
import torch
from cleaning import clean_text
from text2Vec import embed_texts 

def predict_text(model, text):
    cleaned = clean_text(text)
    embedding_tensor = embed_texts([cleaned])
    embedding = embedding_tensor.numpy()

    prediction = model.predict(embedding)
    likelihood = model.predict_proba(embedding)[0]

    if prediction == 1:
        label_as_str = "Sexual Harassment"
    elif prediction == 0:
        label_as_str = "Normal"
    else:
        label_as_str = "[ERROR]"

    confidence = likelihood[prediction].item() * 100
    return label_as_str, prediction, confidence

def run_gui(model_path):
    model = joblib.load(model_path)

    def on_predict():
        user_input = text_box.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Warning", "Please enter some text.")
            return
        label, pred_class, confidence = predict_text(model, user_input)
        messagebox.showinfo("Prediction",
                            f"Prediction: {label} (class {pred_class})\nConfidence: {confidence:.2f}%")

    root = tk.Tk()
    root.title("Interactive Tryout")

    tk.Label(root, text = "Enter your message:").pack(pady = 5)

    text_box = tk.Text(root, wrap = tk.WORD, width = 60, height = 8)
    text_box.pack(pady = 5)

    tk.Button(root, text = "Predict", command = on_predict).pack(pady = 10)

    root.mainloop()
    

if __name__ == "__main__":
    run_gui("best_selftraining_knn.pkl")