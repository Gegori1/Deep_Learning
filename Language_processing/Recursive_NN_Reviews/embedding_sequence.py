import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class SequenceFromText:
    """
    Create sequence of words from text file for sequential model model.
    Arguments:
        - text_path: str, path to the text file.
        - word_dictionary: dict, dictionary with words and their embeddings.
        - series_length: int, length of the series.
    """
    def __init__(self, text_path: str, word_dictionary: dict, series_length: int) -> None:
        self.data = self.read_data(text_path)
        self.word_dictionary = word_dictionary
        self.series_length = series_length
        
        
    def read_data(self, text_path: str):
        return (
            pd.read_csv(text_path, encoding="utf-8")
            [["Review Text", "Rating"]]
            .dropna()
            .reset_index(drop=True)
        )                 
        
    def words2seq(self, words, voc_w2v):
        return [voc_w2v[word] for word in words]
    
    def text_preprocessing(self, text: str):
        length = self.series_length
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and token in self.word_dictionary and token.isalpha()
        ]
        if len(tokens) > length:
            tokens = tokens[:length]
            
        if len(tokens) < length:
            tokens = tokens + ['unk'] * (length - len(tokens))
            
        tokens = self.words2seq(tokens, self.word_dictionary)
            
        return tokens
    
    def create_stopwords(self):
        stop_words = stopwords.words('english')
        stop_words = list(stop_words)
        stop_words = [i for i in stop_words if i not in 
            ["don't", "doesn't", "aren't", "isn't", "hadn't", "won't", "couldn't",
            "not", "no", "nor", "while", "very", "against", "won't", "few", "off"]
        ]
        stop_words = set(stop_words)
        return stop_words
    
    def create_sequence(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self.create_stopwords()
        self.tokenizer = RegexpTokenizer(r'\w+')
        X_all = []
        for text in self.data["Review Text"]:
            X = self.text_preprocessing(text)
            X_all.append(X)
        
        Y_all = (self.data.Rating - 1).to_list()
    
        return X_all, Y_all
    
# %%
