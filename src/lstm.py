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
    
    def set_max_healdine_len(self):


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

    def build_vocab_id(self):
        self.word2id = {w: i + 1 for i, w in enumerate(self.model.wv.index_to_key)}
        self.word2id["<PAD>"] = 0

    def headline_to_ids(self, headline):
        tokens = self.clean_tokenize(headline)
        ids = [self.word2id[t] for t in tokens if t in self.word2id]
        return ids
    
    # LSTM needs equal length input sequences, add padding to equal sentence lengths
    def pad_headline(self, ids, max_headline_len):
        if len(ids) < max_headline_len:
            ids = ids + [0] * (max_headline_len - len(ids))
        else:
            ids = ids[:max_headline_len]
        return ids

    def headline_to_sequence(self, headline, max_len):
        ids = self.headline_to_ids(headline)
        padded = self.pad_headline(ids, max_len)
        return np.array(padded)
    

    # Add lazy column of embedded sentences
    def add_embedded_column(self):
        self.lf = self.lf.with_columns(
            embedded_headline=pl.col(self.col_name).map_elements(
                self.headline_to_sequence,
                return_dtype=pl.List(pl.Float64)
            )
        )
        return self.lf

l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet")

l.load_headlines(n=10_000)
# print(l.lf.collect())

# Set a value for the maximum headline len
headline_lens = l.lf.with_columns(
        len_headline=pl.col(l.col_name).map_elements(
        l.clean_tokenize,
        return_dtype=pl.List(pl.String)
    ).list.len()
)
print(headline_lens.collect())

max_len = headline_lens.select("len_headline").max().collect().item()
print(max_len)
len_p95 = (
    headline_lens
    .select("len_headline")
    .quantile(0.95, interpolation="linear")
    .collect()
)
len_p95_value = len_p95["len_headline"].item()
print(len_p95_value)



# l.train_word2vec()
# l.build_vocab_id()

# print(l.word2id)
# h = l.lf.select(l.col_name).collect()
# print(h)



# l.add_embedded_column()
# e = l.lf.select("embedded_headline").collect()
# for em in e:
#     print(e)

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


