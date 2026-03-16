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
            for row in self.lf.select("tokenized_headline").collect(engine="streaming").iter_rows():
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
    
    # Map the Word2Vec embedding to each token in the vocab
    def build_embedding_matrix(self):
        vocab_size = len(self.word2id)
        emb_dim = self.vector_size

        matrix = np.zeros((vocab_size, emb_dim))

        for word, idx in self.word2id.items():
            if word in self.model.wv:
                matrix[idx] = self.model.wv[word]

        self.embedding_matrix = matrix
    
    def run(self, n):
        self.load_headlines(n)
        self.tokenize_lf()
        self.train_word2vec()
        self.build_vocab_id()
        self.add_embedded_column()
        self.build_embedding_matrix()

l = LazyHeadlineVectorizer("../news_formatted_2018-01-01_2023-12-31.parquet")
l.run(n=5)

print(l.embedding_matrix.shape)

# print(l.word2id)
# e = l.lf.select(l.col_name, "embedded_headline").collect()
# for row in e.iter_rows():
#     print(row)

# ===== Attention Mechanism =====
# Based on code from: https://www.ijcai.org/proceedings/2020/0626.pdf
class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_dim):
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_dim, hidden_dim) # Matrix for memory
        self.w_2 = nn.Linear(hidden_dim, hidden_dim) # Matrix for query
        self.u = nn.Linear(hidden_dim, 1, bias=False) # scores how imporant h is

    # LINEAR ADDITIVE ATTENTION MECHANISM EQ. (2) AND (3)
    def forward(self, memory, query=None, mask=None):
        '''
        :param query:  (node, hidden)
        :param memory: (node, hidden)
        :param mask:
        :return:
        '''

        # h = W1*h_t + W2*x_s === Eq. (2)
        if query is None:
            h = torch.tanh(self.w_1(memory))  # shape: (node, hidden)
        else:
            h = torch.tanh(self.w_1(memory) + self.w_2(query)) # shape: (node, hidden)

        score = torch.squeeze(self.u(h), -1)  # score = u * h
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9) # set score = -infinity on masked places
        alpha = F.softmax(score, -1)  # shape: (node)
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2) # Eq. (3)
        return s # shape: (hidden_dim, 1)

# ===== LSTM Encoder =====
class LSTM_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix, layer_dim=1):
        super(LSTM_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix)) # Copy the Word2Vec embeddings

        # hidden_dim = dim of h_t
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.att = Attentive_Pooling(hidden_dim)

    def forward(self, x: torch.Tensor, stock_embedding=None, h_in=None, mem_in=None):
        # x.shape = (batch_size, seq_len)
        # x_emb.shape = (batch_size, seq_len, embed_dim)
        x_embedding = self.embedding(x) # lookup the embedding

        if h_in is None or mem_in is None:
            h_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            mem_in = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        out, (h_out, c_out) = self.lstm(x_embedding, (h_in, mem_in))
        # out.shape: (batch_size, seq_len, hidden_dim)
        # h_out.shape: (batch_size, 1, hidden_dim)

        # Apply attention mechanism
        # TODO: Get the current stock embedding for query from the Graph, using the 'Stock_symbol'?
        mask = (x != 0) # Mask to give no weight to the pad tokens in the attention
        h_att = self.att(out, query=stock_embedding, mask=mask)

        return h_att

lstm = LSTM_Encoder(len(l.word2id), l.vector_size, 128, l.embedding_matrix)
# print(lstm.embedding.weight)

print(l.lf.head(5).collect())
print(f"max_95th_len: {l.max_headline_len}")

test_headline = l.lf.select("embedded_headline").collect()[1].item()
batch = [test_headline]
test_headline = torch.tensor(batch, dtype=torch.long)

# get stock emb from the graph
stock_emb = None
h_att = lstm(test_headline, stock_embedding=stock_emb)
print(h_att)


# TODO: Next steps:
# 1. Add attention mechanism on top of the lstm
# 2. Train the LSTM on the 'train headlines'
# 3. Somehow store a output vector h_t in a Lazyframe
