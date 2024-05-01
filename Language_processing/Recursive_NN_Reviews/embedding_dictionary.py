import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.lm import Vocabulary
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

class EmbeddingDictionary:
    "Reads a cvs file with multiple texst and creates a dictionary with the words and their embeddings."
    def __init__(self, text_path: str, unk_cutoff: int):
        self.data = self.read_data(text_path)
        self.unk_cutoff = unk_cutoff
        
    def read_data(self, text_path: str):
        return (
            pd.read_csv(text_path, encoding="utf-8")
            ["Review Text"]
            .dropna()
            .reset_index(drop=True)
        )
        
    def create_embedding_dict(self):
        all_words = (" ".join(self.data.to_list())).lower()
        tokens = self.text_preprocessing(all_words)
        vocab = Vocabulary(tokens, unk_cutoff=self.unk_cutoff, unk_label="unk")
        voc_w2v = {word: idx for idx, word in enumerate(vocab)}
        self.dictionary_size = len(voc_w2v)
        return voc_w2v
    
    def create_stopwords(self):
        stop_words = stopwords.words('english')
        stop_words = list(stop_words)
        stop_words = [i for i in stop_words if i not in 
            ["don't", "doesn't", "aren't", "isn't", "hadn't", "won't", "couldn't",
            "not", "no", "nor", "while", "very", "against", "won't", "few", "off"]
        ]
        stop_words = set(stop_words)
        return stop_words
        
    def text_preprocessing(self, text: str):
        lemmatizer = WordNetLemmatizer()
        stop_words = self.create_stopwords()
        tokenizer = RegexpTokenizer(r'[\w]+')
        tokens = tokenizer.tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        return tokens
