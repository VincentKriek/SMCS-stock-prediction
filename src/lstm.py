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
        self.max_headline_len = None
        self.word2id = {}


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
    
    def tokenize_lf(self):
        self.lf = self.lf.with_columns(
                tokenized_headline=pl.col(self.col_name).map_elements(
                l.clean_tokenize,
                return_dtype=pl.List(pl.String)
            )
        ).with_columns(
            headline_len=pl.col("tokenized_headline").list.len()
        )

        # Set the max allowed healdine length for lstm to the 95th percentile
        self.max_headline_len = int((
            self.lf
            .select("headline_len")
            .quantile(0.95, interpolation="linear")
            .collect()
        ).item())

    # Iterator for Word2Vec
    class HeadlinesIterator:
        def __init__(self, lf):
            self.lf: pl.LazyFrame = lf

        def __iter__(self):
            for row in self.lf.select("tokenized_headline").collect(streaming=True).iter_rows():
                yield row[0]

    # Train Word2Vec lazily
    def train_word2vec(self):
        sentences = self.HeadlinesIterator(self.lf)
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count
        )
        return self.model

    def build_vocab_id(self):
        self.word2id = {w: i + 1 for i, w in enumerate(self.model.wv.index_to_key)}
        self.word2id["<PAD>"] = 0

    def add_embedded_column(self):
        def process(tokens):
            ids = [self.word2id.get(t, 0) for t in tokens] # convert token to int

            # LSTM needs equal length input sequences, add padding to equal sentence lengths
            if len(ids) < self.max_headline_len:
                ids = ids + [0] * (self.max_headline_len - len(ids))
            else:
                ids = ids[:self.max_headline_len]

            return ids

        self.lf = self.lf.with_columns(
            embedded_headline=pl.col("tokenized_headline").map_elements(
                process,
                return_dtype=pl.List(pl.Int64)
            )
        )

        return self.lf
    
    def run(self, n):
        self.load_headlines(n)
        self.tokenize_lf()
        self.train_word2vec()
        self.build_vocab_id()
        self.add_embedded_column()

l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet")

# l.load_headlines(n=5)
# # print(l.lf.collect())

# l.tokenize_lf()

# l.train_word2vec()
# l.build_vocab_id()

# l.add_embedded_column()

l.run(n=5)

print(l.word2id)
e = l.lf.select(l.col_name, "embedded_headline").collect()
for row in e.iter_rows():
    print(row)

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


