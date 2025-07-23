"""
    Filename: cleaning.py
    Title: Return a cleaned data frame (no more duplicates, bots' comments, or comments with just
            some random links in it only, etc.)
    Date: July 2025
"""

import re
from typing import List

def clean_text(txt: str) -> str:
    '''
        Clean up:
         - Input: a string (column 'body' in the data frame)
         - Ouput: cleaned up string 
        by removing...
    '''
    ### ... URLs
    txt = re.sub(r'http\\S+', '', txt)
    ### ...markdown links but keep visible text: something like [text](link) will turn to [text]
    txt = re.sub(r'\\[(.*?)\\]\\((.*?)\\)', r'\\1', txt)
    ### ...spacing and lowercases the text
    txt = re.sub(r'\\s+', ' ', txt).strip().lower()
    ### ...boilerplates
    txt = re.sub(r'(?i)(edit|tl;dr|tldr)[:\\s]', '', txt)
    ### ...punctuations (but not emojis)
    txt = re.sub(r'[\"""\'\\^&*_+=~`|\\[\\]{}<>]', '', txt)

def preprocess_dataset(input_path: str) -> List[str]:
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [clean_text(line) for line in lines if line.strip()]