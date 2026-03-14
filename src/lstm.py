import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

print(torch.__version__)
print("CUDA:", torch.cuda.is_available())

nltk.download('punkt_tab')
nltk.download('stopwords')

# 2️⃣ Define the class
class LazyHeadlineVectorizer:
    def __init__(self, parquet_path, col_name="Article_title", vector_size=128, window=5, min_count=1):
        self.parquet_path = parquet_path
        self.col_name = col_name
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

        self.lf = None
        self.model = None

    def load_headlines(self, n=None):
        self.lf = pl.scan_parquet(self.parquet_path)
        if n:
            self.lf = self.lf.head(n)
        return self.lf

    # Tokenizer for a single headline
    def clean_tokenize(self, headline):
        tokens = word_tokenize(headline.lower())
        tokens = [t for t in tokens if t not in self.stop_words
                  and t not in self.punctuation
                  and not t.isnumeric()]
        return tokens

    # Iterator for Word2Vec
    class HeadlinesIterator:
        def __init__(self, lf, col_name, tokenizer):
            self.lf = lf
            self.col_name = col_name
            self.tokenizer = tokenizer

        def __iter__(self):
            # streaming=True avoids loading all rows into memory
            for batch in self.lf.collect(streaming=True).iter_rows(named=True):
                yield self.tokenizer(batch[self.col_name])

    # Train Word2Vec lazily
    def train_word2vec(self):
        sentences = self.HeadlinesIterator(self.lf, self.col_name, self.clean_tokenize)
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count
        )
        return self.model

    # TODO: Do we need to combine the indiv. word vectors into one sentence vector here?
    # Or does LSTM only need word for word vectors?
    # Average of word vectors
    def headline_to_vector(self, headline):
        tokens = self.clean_tokenize(headline)
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if len(vectors) == 0:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    # Add lazy column of embedded sentences
    def add_embedded_column(self):
        self.lf = self.lf.with_columns(
            embedded_sentence=pl.col(self.col_name).map_elements(
                self.headline_to_vector,
                return_dtype=pl.List(pl.Float64)
            )
        )
        return self.lf

l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet")

l.load_headlines(n=5)
# print(l.lf.collect())
l.train_word2vec()
l.add_embedded_column()
print(l.lf.collect())

# # ===== LSTM Encoder =====
# class LSTM_Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         # hidden_dim = dim of h_t
#         # https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

#     def forward(self, x: torch.Tensor, h_in=None, mem_in=None):
#         x = x.reshape(x.shape[0], -1)
#         out, (h_out, c_out) = self.lstm(x, (h_in, mem_in))
#         return out, h_out, c_out


