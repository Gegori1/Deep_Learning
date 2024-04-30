import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.lm import Vocabulary

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
        
    def text_preprocessing(self, text: str):
        stop_words = stopwords.words('english')
        tokenizer = RegexpTokenizer(r'[\w]+')
        tokens = tokenizer.tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [token for token in tokens if token.isalpha()]
        return tokens
