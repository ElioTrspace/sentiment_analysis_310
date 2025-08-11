"""
    Filename: cleaning.py
    Title: Return a cleaned data frame (no bots' comments, or comments with just
            some random links in it only, etc.)
    Date: July 2025
"""
import numpy as np
import re
from typing import List
import sys
import pandas as pd

def clean_text(txt) -> str:
    '''
        Clean up:
         - Input: a string (column 'body' in the data frame)
         - Ouput: cleaned up string 
        by removing weird boilerplates, links, markdown links all kind and transform abbreviations
    '''
    txt = str(txt).lower()
    txt = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", txt)
    txt = re.sub(r"what's", "what is ", txt)
    txt = re.sub(r"\'s", " ", txt)
    txt = re.sub(r"\'ve", " have ", txt)
    txt = re.sub(r"can't", "cannot ", txt)
    txt = re.sub(r"n't", " not ", txt)
    txt = re.sub(r"i'm", "i am ", txt)
    txt = re.sub(r"\'re", " are ", txt)
    txt = re.sub(r"\'d", " would ", txt)
    txt = re.sub(r"\'ll", " will ", txt)
    txt = re.sub(r",", " ", txt)
    txt = re.sub(r"\.", " ", txt)
    txt = re.sub(r"!", " ! ", txt)
    txt = re.sub(r"\/", " ", txt)
    txt = re.sub(r"\^", " ^ ", txt)
    txt = re.sub(r"\+", " + ", txt)
    txt = re.sub(r"\-", " - ", txt)
    txt = re.sub(r"\=", " = ", txt)
    txt = re.sub(r"'", " ", txt)
    txt = re.sub(r"(\d+)(k)", r"\g<1>000", txt)
    txt = re.sub(r":", " : ", txt)
    txt = re.sub(r" e g ", " eg ", txt)
    txt = re.sub(r" b g ", " bg ", txt)
    txt = re.sub(r" u s ", " american ", txt)
    txt = re.sub(r"\0s", "0", txt)
    txt = re.sub(r" 9 11 ", "911", txt)
    txt = re.sub(r"e - mail", "email", txt)
    txt = re.sub(r"j k", "jk", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    txt = re.sub(r'http\S+', '', txt)
    txt = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', txt) # for markdown links

    txt = txt.split()

    return " ".join(txt)

def is_valid_text(row):
    # since I discovered a lot of bots are distracting the model
    text = row['body']
    author = str(row['author']).lower()

    if (author == 'automoderator') or ('bot' in author) or (author == 'floodassistant'):
        return False

    if not isinstance(text, str) or (len(text.strip()) < 10):
        return False
    if ('[removed]' in text) or ('[deleted]' in text):
        return False

    return True

def preprocess_dataset(input_path: str) -> pd.DataFrame:
    data = pd.read_json(input_path, orient = 'records', convert_dates = True, lines = True)
    ### doing some hardcoding here, #TODO: modify this
    data = data.dropna(subset = ['body'])
    data['body'] = data['body'].astype(str)
    data = data[data['body'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    data = data[data.apply(is_valid_text, axis = 1)]

    data['body'] = data['body'].apply(clean_text)
    
    if data.empty:
        print("[WARNING] No usable text found in 'body' column.")
        return []
    return data