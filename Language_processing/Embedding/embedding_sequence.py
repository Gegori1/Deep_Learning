import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


class SequenceFromText:
    """
    Create sequence of words from text file for CBOW model.
    Arguments:
        - text_path: str, path to the text file.
        - word_dictionary: dict, dictionary with words and their embeddings.
        - hwc: int, number of words to the left and right of the target word.
    """
    def __init__(self, text_path: str, word_dictionary: dict, half_word_context: int) -> None:
        self.data = self.read_data(text_path)
        self.word_dictionary = word_dictionary
        self.hwc = half_word_context
        
        
    def read_data(self, text_path: str):
        return (
            pd.read_csv(text_path, encoding="utf-8")
            ["Review Text"]
            .dropna()
            .reset_index(drop=True)
        )
        
    def create_examples(self, words, voc_w2v, hcw):
        X, Y = [], []
        pad = ['unk'] * hcw
        lp_words = [*pad, *words]
        lrp_words = [*lp_words, *pad]
        for target_index in range(hcw, len(lrp_words) - hcw):
            context = lrp_words[target_index - hcw: target_index] + lrp_words[target_index + 1: target_index + hcw + 1]
            target = lrp_words[target_index]
            X.append(self.words2seq(context, voc_w2v))
            Y.append(voc_w2v[target])
        return X, Y
    
    def words2seq(self, words, voc_w2v):
        return [voc_w2v[word] for word in words]
    
    def text_preprocessing(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        return [
            token for token in tokens 
            if token not in self.stop_words and token in self.word_dictionary and token.isalpha()
        ]
    
    def create_sequence(self):
        self.stop_words = stopwords.words('english')
        self.tokenizer = RegexpTokenizer(r'[\w]+')
        X_all, Y_all = [], []
        for text in self.data:
            words = self.text_preprocessing(text)
            X, Y = self.create_examples(words, self.word_dictionary, self.hwc)
            X_all.extend(X)
            Y_all.extend(Y)
    
        return X_all, Y_all